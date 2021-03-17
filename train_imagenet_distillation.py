from __future__ import print_function
import argparse
import shutil
import torch.backends.cudnn as cudnn
import torch.nn as nn
from models import model_dict
from utils.utils import *
from distiller_zoo.AIN import transfer_conv,statm_loss
from dataset.imagenet import get_imagenet_dataloader
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='knowledge distillation')
# training hyper parameters
parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--workers', type=int, default=16, help='workers')
parser.add_argument('--lr', type=float, default=0.2, help='initial learning rate 0.2')
parser.add_argument('--epochs', type=int, default=100, help='number of total epochs')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay 1e-4')
# net and dataset choosen
parser.add_argument('--net_s', type=str, required=True, choices=['resnet18S', 'MobileNet'], help='')
parser.add_argument('--net_t', type=str, required=True, choices=['resnet34T', 'resnet50T'], help='')

# 0.5 for ce and 0.9 for kd
def main():
    global args
    args = parser.parse_args()
    cur_path = os.path.abspath(os.curdir)
    save_path = cur_path.replace('KD_SRRL', 'Results') + '/ImageNet_AB/' + \
                str(args.mode) + 'T:' + str(args.net_t) + 'S:' + str(args.net_s) + '_weight:' + str(args.weight)+'_lr'+str(args.lr)
    print(save_path)
    model_file = os.path.join(save_path, 'models')
    if not os.path.exists(model_file):
        os.makedirs(model_file)
    create_logger(os.path.join(save_path, 'logs'))
    shutil.copy2('train_imagenet_distillation.py', save_path)
    shutil.copy2('models/official_resnet.py', save_path)
    train_loader, test_loader = get_imagenet_dataloader(batch_size=args.batch_size, num_workers=args.workers)
    net_t = model_dict[args.net_t](num_class=1000)
    net_t = torch.nn.DataParallel(net_t)
    net_t = net_t.cuda()
    net_t.eval()
    for param in net_t.parameters():
        param.requires_grad = False

    net_s = model_dict[args.net_s](num_class=1000)
    student_params = sum(p.numel() for p in net_s.parameters())
    print('student_param:%d' %student_params)
    logging.info('student_param:%d' %student_params)

    net_s = torch.nn.DataParallel(net_s)
    net_s = net_s.cuda()

    trainable_list = nn.ModuleList([])
    trainable_list.append(net_s)
    conector = torch.nn.DataParallel(transfer_conv(net_s.module.fea_dim, net_t.module.fea_dim)).cuda()
    trainable_list.append(conector)
    optimizer = torch.optim.SGD(trainable_list.parameters(),  lr = args.lr, momentum = args.momentum,weight_decay = args.weight_decay)
    net_s, optimizer, last_epoch, best_epoch,best_top1,best_top5 = \
        load_checkpoints(net_s, optimizer, model_file)

    lr_scheduler = lr_step_policy(args.lr, [30, 60, 90], 0.1, 0)
    val_top1, val_top5 = test2(test_loader, net_t)
    print('net_t:%.2f,%.2f'%(val_top1,val_top5))
    logging.info('net_t:%.2f,%.2f'%(val_top1,val_top5))
    val_top1, val_top5 = test2(test_loader, net_s)
    print('epochs:%d net_s:%.2f,%.2f'%(args.epochs,val_top1,val_top5))
    logging.info('epochs:%d net_s:%.2f,%.2f'%(args.epochs,val_top1,val_top5))

    for epoch in range(last_epoch+1, args.epochs):
        lr_scheduler(optimizer, epoch)
        epoch_start_time = time.time()
        train(train_loader, net_t, net_s, optimizer, conector,epoch)
        epoch_time = time.time() - epoch_start_time
        print('one epoch time is {:02}h{:02}m{:02}s'.format(*transform_time(epoch_time)))
        print('testing the models......')
        test_start_time = time.time()
        val_top1, val_top5 = test2(test_loader, net_s)
        if val_top1 > best_top1:
            best_top1 = val_top1
            best_top5 = val_top5
            best_epoch = epoch
            model_save = os.path.join(model_file, 'net_best.pth')
            torch.save(net_s.state_dict(),model_save)
        test_time = time.time() - test_start_time
        print('testing time is {:02}h{:02}m{:02}s'.format(*transform_time(test_time)))
        print('lr:%.6f,epoch:%d,cur_top1:%.2f,cur_top5:%.2f,best_epoch:%d,best_top1:%.2f,best_top5:%.2f'%
              (optimizer.param_groups[0]['lr'],epoch,val_top1,val_top5,best_epoch,best_top1,best_top5))
        logging.info('lr:%.6f,epoch:%d,cur_top1:%.2f,cur_top5:%.2f,best_epoch:%d,best_top1:%.2f,best_top5:%.2f'%
              (optimizer.param_groups[0]['lr'],epoch,val_top1,val_top5,best_epoch,best_top1,best_top5))

def train(train_loader,net_t,net_s,optimizer, conector,epoch):
    batch_time = AverageMeter('Time',':.3f')
    data_time  = AverageMeter('Data',':.3f')
    losses     = AverageMeter('Loss',':.3f')
    losses_ce  = AverageMeter('ce', ':.3f')
    losses_kd  = AverageMeter('kd', ':.3f')
    top1       = AverageMeter('Acc@1', ':.2f')
    top5       = AverageMeter('Acc@5', ':.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time,losses, losses_ce, losses_kd, top1],
        prefix="Epoch: [{}]".format(epoch))

    net_s.train()
    conector.train()
    end = time.time()
    for idx, data in enumerate(train_loader):
        data_time.update(time.time() - end)
        img, target = data
        img = img.cuda()
        target = target.cuda()
        with torch.no_grad():
            feat_t, pred_t = net_t(img, is_adain=True)
        feat_s, pred_s = net_s(img, is_adain=True)
        feat_s = conector(feat_s)
        loss_stat = statm_loss(feat_s, feat_t.detach())
        pred_sc = net_t(x=None,feat_s=feat_s)
        loss_kd = loss_stat + F.mse_loss(pred_sc, pred_t)
        loss_ce = F.cross_entropy(pred_s, target)

        loss = loss_ce+loss_kd*args.weight
        prec1, prec5 = accuracy(pred_s, target, topk=(1,5))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_ce.update(loss_ce.detach().item(), img.size(0))
        losses_kd.update(loss_kd.detach().item(), img.size(0))
        losses.update(loss.detach().item(), img.size(0))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % args.print_freq == 0:
            progress.display(idx)
            if idx % (args.print_freq*5) == 0:
                logging.info('Epoch[{0}]:[{1:03}/{2:03}]'
                             'Time:{batch_time.val:.4f}'
                             'loss:{losses.val:.4f}({losses.avg:.4f})'
                             'ce:{losses_ce.val:.4f}({losses_ce.avg:.4f})'
                             'kd:{losses_kd.val:.4f}({losses_kd.avg:.4f})'
                             'prec@1:{top1.val:.2f}({top1.avg:.2f})'.format(
                    epoch, idx, len(train_loader), batch_time=batch_time,
                    losses=losses, losses_ce=losses_ce, losses_kd=losses_kd, top1=top1))

if __name__ == '__main__':
    main()