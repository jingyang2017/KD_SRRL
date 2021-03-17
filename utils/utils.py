from __future__ import print_function
import torch
import torch.nn.functional as F
import time
import logging
import numpy as np
import os
import json

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('|'.join(entries))

        # print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val'+self.fmt+'} ({avg'+self.fmt+'})'
        return fmtstr.format(**self.__dict__)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def transform_time(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return h,m,s

def create_logger(root_out_path):
    #set up logger
    if not os.path.exists(root_out_path):
        os.makedirs(root_out_path)
    assert os.path.exists(root_out_path), '{} does not exits'.format(root_out_path)
    log_file = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(root_out_path,log_file),format=head)
    print(os.path.join(root_out_path,log_file))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    return logger


def lr_policy(lr_fn):
    def _alr(optimizer, epoch):
        lr = lr_fn(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr


def lr_step_policy(base_lr, steps, decay_factor, warmup_length):
    def _lr_fn( epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            lr = base_lr
            for s in steps:
                if epoch >= s:
                    lr *= decay_factor
        return lr

    return lr_policy(_lr_fn)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred    = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def load_checkpoints(net, optimizer, model_path):
    latestpath = os.path.join(model_path, 'latest.pth.tar')
    if os.path.exists(latestpath):
        print('===================>loading the checkpoints from:', latestpath)
        latest = torch.load(latestpath)
        last_epoch = latest['epoch']
        best_epoch = latest['best_epoch']
        best_top1 = latest['best_top1']
        best_top5 = latest['best_top5']
        net.load_state_dict(latest['models'])
        optimizer.load_state_dict(latest['optim'])
        return net, optimizer, last_epoch,best_epoch,best_top1,best_top5
    else:
        print('====================>Train From Scratch')
        return net, optimizer, -1, 0, 0, 0

def save_checkpoints(net, optimizer, epoch,best_epoch,best_top1,best_top5, model_path):
    latest = {}
    latest['epoch'] = epoch
    latest['models'] = net.state_dict()
    latest['optim'] = optimizer.state_dict()
    latest['best_epoch'] = best_epoch
    latest['best_top1'] = best_top1
    latest['best_top5'] = best_top5
    torch.save(latest, os.path.join(model_path, 'latest.pth.tar'))

def test(test_loader, net):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    net.eval()
    for idx, (img, target, index) in enumerate(test_loader):
        img = img.cuda()
        target = target.cuda()
        with torch.no_grad():
            output = net(img)
        prec1, prec5 = accuracy(output, target, topk=(1,5))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))
    return top1.avg, top5.avg

def test2(test_loader, net):
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    net.eval()
    for idx, (img, target) in enumerate(test_loader):
        img = img.cuda()
        target = target.cuda()
        with torch.no_grad():
            output = net(img)
        prec1, prec5 = accuracy(output, target, topk=(1,5))
        top1.update(prec1.item(), img.size(0))
        top5.update(prec5.item(), img.size(0))
    return top1.avg, top5.avg

def mixup_data(x, y, alpha):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0
    return warpper

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__






