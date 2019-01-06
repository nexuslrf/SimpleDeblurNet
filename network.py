import torch
import torch.nn as nn
from layers import *


def conv5x5_relu(in_channels, out_channels, stride):
    return conv(in_channels, out_channels, 5, stride, 2, activation_fn=partial(nn.ReLU, inplace=True))


def deconv5x5_relu(in_channels, out_channels, stride, output_padding):
    return deconv(in_channels, out_channels, 5, stride, 2, output_padding=output_padding,
                  activation_fn=partial(nn.ReLU, inplace=True))


def resblock(in_channels):
    """Resblock without BN and the last activation
    """
    return BasicBlock(in_channels, out_channels=in_channels, kernel_size=5, stride=1, use_batchnorm=False,
                      activation_fn=partial(nn.ReLU, inplace=True), last_activation_fn=None)


class EBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, resblcoks=3):
        super(type(self), self).__init__()
        self.conv = conv5x5_relu(in_channels, out_channels, stride)
        resblock_list = []
        for i in range(resblcoks):
            resblock_list.append(resblock(out_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)

    def forward(self, x):
        x = self.conv(x)
        x = self.resblock_stack(x)
        return x


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, output_padding):
        super(type(self), self).__init__()
        resblock_list = []
        for i in range(3):
            resblock_list.append(resblock(in_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)
        self.deconv = deconv5x5_relu(in_channels, out_channels, stride, output_padding)

    def forward(self, x):
        x = self.resblock_stack(x)
        x = self.deconv(x)
        return x


class OutBlock(nn.Module):
    def __init__(self, in_channels):
        super(type(self), self).__init__()
        resblock_list = []
        for i in range(3):
            resblock_list.append(resblock(in_channels))
        self.resblock_stack = nn.Sequential(*resblock_list)
        self.conv = conv(in_channels, 3, 5, 1, 2, activation_fn=None)

    def forward(self, x):
        x = self.resblock_stack(x)
        x = self.conv(x)
        return x


class SRNDeblurNet(nn.Module):
    """SRN-DeblurNet
    examples:
        net = SRNDeblurNet()
        y = net( x1 , x2 , x3）#x3 is the coarsest image while x1 is the finest image
    """

    def __init__(self, upsample_fn=partial(torch.nn.functional.upsample, mode='bilinear'), xavier_init_all=True, whole_skip=False):
        super(type(self), self).__init__()
        self.upsample_fn = upsample_fn
        self.inblock = EBlock(3, 32, 1)
        self.eblock1 = EBlock(32, 64, 2)
        self.eblock2 = EBlock(64, 128, 2)
        self.convmid = nn.Conv2d(128, 128, 5, padding=2)
        self.dblock1 = DBlock(128, 64, 2, 1)
        self.dblock2 = DBlock(64, 32, 2, 1)
        self.outblock = OutBlock(32)
        self.skip = whole_skip
        # self.cblock = nn.Sequential(
        #                     EBlock(128, 128, 2, 1),
        #                     EBlock(128, 128, 2, 1),
        #                     EBlock(128, 128, 2, 1)
        #                 )
        # self.classifier = nn.Sequential(
        #     nn.Linear(128 * 8 * 8, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 1),
        # )
        # self.prob = nn.Sigmoid()

        self.input_padding = None
        if xavier_init_all:
            for name, m in self.named_modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    torch.nn.init.xavier_normal_(m.weight)
                    # print(name)

    def forward(self, x):
        e32 = self.inblock(x)
        e64 = self.eblock1(e32)
        e128 = self.eblock2(e64)
        h   = self.convmid(e128)
        # c1   = self.cblock(h)
        # c1 = c1.view(c1.size(0), -1)
        # co   = self.classifier(c1)
        # p   = self.prob(co/2)
        d64 = self.dblock1(h)
        d32 = self.dblock2(d64 + e64)
        d3 = self.outblock(d32 + e32)
        if self.skip:
            d3 = d3 + x
        # d3f = d3.view(d3.size()[0],-1)
        # out = d3f * (1-p) + x.view(x.size()[0],-1) * p
        # out = out.view(d3.size())
        return d3








