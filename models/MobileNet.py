'''MobileNet in PyTorch.
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileNet_(nn.Module):
    def __init__(self,num_class=1000):
        super(MobileNet_, self).__init__()
        self.fea_dim = 1024
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )
        self.model1 = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
        self.model2 = nn.Sequential(
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )
        self.pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024, num_class)

    def forward(self, x,is_adain=False):
        x = self.model1(x)
        x = self.model2(x)
        fea = x
        x = self.pool(x)
        x = x.view(-1, 1024)
        f1 = x
        x = self.fc(x)
        if is_adain:
            return fea,x
        else:
            return x

def MobileNet(num_class=1000):
    model = MobileNet_(num_class=num_class)
    return model

if __name__ == '__main__':
    net_G = MobileNet()
    sub_params = sum(p.numel() for p in net_G.parameters())
    print(sub_params)
    # net(torch.randn(2, 3, 224, 224))


