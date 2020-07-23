
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import coco, voc
from layers import *
from torch.autograd import Variable


class SSD(nn.Module):
    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.size = size
        #先验框生成
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = Variable(self.priorbox.forward())
        self.size = size

        self.vgg = nn.ModuleList(base)
        #conv4_3之后使用L2Norm
        self.L2Norm = L2Norm(512,20)
        self.extras = nn.ModuleList(extras)
        
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        # till conv4_3 relu 
        for k in range(23):
            x = self.vgg[k](x)
        
        s = self.L2Norm(x)
        sources.append(s)

        # till fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        #extra layeres
        for k,v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)


def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if(v == 'M'):
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif(v == 'C'):
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv = nn.conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True),
               conv7, nn.ReLU(inplace=True)]
    return layers

def add_extras(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    flag = False
    for k,v in enumerate(cfg):
        if(in_channels != 'S'):
            if (v == 'S'):
                layers += [nn.Conv2d(in_channels, cfg[k+1], kernel_size=(1, 3)[flag],
                           stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg(v).out_channels,
                       cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k] * 4,
                                 kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k] * num_classes,
                                  kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)

base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}

def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)