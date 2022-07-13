import torch
import torch.nn as nn
import torch.nn.functional as F
from models.quantization import *

__all__ = ['ResNet', 'resnet18_quan', 'resnet34_quan', 'resnet50_quan', 'resnet101_quan', 'resnet152_quan']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, n_bits=8):
        super(BasicBlock, self).__init__()
        self.conv1 = quan_Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, n_bits=n_bits)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = quan_Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False, n_bits=n_bits)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                quan_Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, n_bits=n_bits),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, n_bits=8):
        super(Bottleneck, self).__init__()
        self.conv1 = quan_Conv2d(in_planes, planes, kernel_size=1, bias=False, n_bits=n_bits)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = quan_Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False, n_bits=n_bits)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = quan_Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False, n_bits=n_bits)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                quan_Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, n_bits=n_bits),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, n_bits=8):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.n_bits = n_bits

        self.conv1 = quan_Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False, n_bits=n_bits)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, n_bits=n_bits)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, n_bits=n_bits)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, n_bits=n_bits)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, n_bits=n_bits)
        self.linear = quan_Linear(512*block.expansion, num_classes, n_bits=n_bits)

    def _make_layer(self, block, planes, num_blocks, stride, n_bits):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n_bits=n_bits))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNet_mid(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, n_bits=8):
        super(ResNet_mid, self).__init__()
        self.in_planes = 64
        self.n_bits = n_bits

        self.conv1 = quan_Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False, n_bits=n_bits)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, n_bits=n_bits)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, n_bits=n_bits)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, n_bits=n_bits)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, n_bits=n_bits)
        self.linear = quan_Linear(512*block.expansion, num_classes, n_bits=n_bits)

    def _make_layer(self, block, planes, num_blocks, stride, n_bits):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n_bits=n_bits))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

def resnet18_quan_mid(num_classes=10, n_bits=8):
    return ResNet_mid(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, n_bits=n_bits)

def resnet18_quan(num_classes=10, n_bits=8):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, n_bits=n_bits)

def resnet34_quan(num_classes=10, n_bits=8):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, n_bits=n_bits)


def resnet50_quan(num_classes=10, n_bits=8):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, n_bits=n_bits)


def resnet101_quan(num_classes=10, n_bits=8):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, n_bits=n_bits)


def resnet152_quan(num_classes=10, n_bits=8):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, n_bits=n_bits)


def test():
    net = resnet18_quan()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

    for name, mod in net.named_modules():
        if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.Linear):
            print(mod.N_bits)

# test()



