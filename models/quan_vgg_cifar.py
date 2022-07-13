import math

import torch.nn as nn
import torch.nn.init as init
from models.quantization import *

__all__ = [
    'VGG', 'vgg11_quan', 'vgg11_bn_quan', 'vgg13_quan', 'vgg13_bn_quan', 'vgg16_quan', 'vgg16_bn_quan',
    'vgg19_bn_quan', 'vgg19_quan',
]


class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, num_class=10, n_bits=8):
        super(VGG, self).__init__()
        self.features = features
        self.n_bits = n_bits
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # nn.Linear(512, 512),
            quan_Linear(512, 512, n_bits=self.n_bits),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(512, 512),
            quan_Linear(512, 512, n_bits=self.n_bits),
            nn.ReLU(True),
            # nn.Linear(512, num_class),
            quan_Linear(512, num_class, n_bits=self.n_bits),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGG_mid(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, features, num_class=10, n_bits=8):
        super(VGG_mid, self).__init__()
        self.features = features
        self.n_bits = n_bits
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # nn.Linear(512, 512),
            quan_Linear(512, 512, n_bits=self.n_bits),
            nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(512, 512),
            quan_Linear(512, 512, n_bits=self.n_bits),
            nn.ReLU(True),
            # nn.Linear(512, num_class),
            quan_Linear(512, num_class, n_bits=self.n_bits),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = self.classifier[3](x)
        x = self.classifier[4](x)
        x = self.classifier[5](x)
        return x


def make_layers(cfg, n_bits=8, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            # conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            conv2d = quan_Conv2d(in_channels, v, kernel_size=3, padding=1, n_bits=n_bits)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11_quan(class_num = 10, n_bits=8):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A'], n_bits), class_num, n_bits)


def vgg11_bn_quan(class_num = 10, n_bits=8):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], n_bits, batch_norm=True), class_num, n_bits)


def vgg13_quan(class_num = 10, n_bits=8):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B'], n_bits), class_num, n_bits)


def vgg13_bn_quan(class_num = 10, n_bits=8):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], n_bits, batch_norm=True), class_num, n_bits)


def vgg16_quan(class_num = 10, n_bits=8):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D'], n_bits), class_num, n_bits)


def vgg16_bn_quan(class_num = 10, n_bits=8):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], n_bits, batch_norm=True), class_num, n_bits)


def vgg16_bn_quan_mid(class_num = 10, n_bits=8):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG_mid(make_layers(cfg['D'], n_bits, batch_norm=True), class_num, n_bits)


def vgg19_quan(class_num = 10, n_bits=8):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E'], n_bits), class_num, n_bits)


def vgg19_bn_quan(class_num = 10, n_bits=8):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], n_bits, batch_norm=True), class_num, n_bits)


if __name__ == '__main__':
    model = vgg16_bn_quan(n_bits=4)

    print(model)

    for n, m in model.named_modules():
        if isinstance(m, quan_Linear) or isinstance(m, quan_Conv2d):
            print(m.N_bits)