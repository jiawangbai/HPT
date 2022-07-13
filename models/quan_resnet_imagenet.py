import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from models.quantization import *

from utils import *

__all__ = [
    'ResNet', 'resnet18_quan', 'resnet34_quan', 'resnet50', 'resnet101',
    'resnet152'
]

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, n_bits=8):
    """3x3 convolution with padding"""
    return quan_Conv2d(in_planes,
                       out_planes,
                       kernel_size=3,
                       stride=stride,
                       padding=1,
                       bias=False,
                       n_bits=n_bits)


def conv1x1(in_planes, out_planes, stride=1, n_bits=8):
    """1x1 convolution"""
    return quan_Conv2d(in_planes,
                       out_planes,
                       kernel_size=1,
                       stride=stride,
                       bias=False,
                       n_bits=n_bits)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, n_bits=8):
        super(BasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, n_bits=n_bits)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, n_bits=n_bits)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, n_bits=8):
        super(Bottleneck, self).__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes, n_bits)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride, n_bits)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion, n_bits)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 n_bits=8,
                 num_classes=1000,
                 zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.n_bits = n_bits

        self.conv1 = quan_Conv2d(3,
                                 64,
                                 kernel_size=7,
                                 stride=2,
                                 padding=3,
                                 bias=False,
                                 n_bits=self.n_bits)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = quan_Linear(512 * block.expansion, num_classes, n_bits=self.n_bits)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, self.n_bits),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, n_bits=self.n_bits))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, n_bits=self.n_bits))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class ResNet_mid(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 n_bits=8,
                 num_classes=1000,
                 zero_init_residual=False):
        super(ResNet_mid, self).__init__()
        self.inplanes = 64
        self.n_bits = n_bits

        self.conv1 = quan_Conv2d(3,
                                 64,
                                 kernel_size=7,
                                 stride=2,
                                 padding=3,
                                 bias=False,
                                 n_bits=self.n_bits)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = quan_Linear(512 * block.expansion, num_classes, n_bits=self.n_bits)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, self.n_bits),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, n_bits=self.n_bits))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, n_bits=self.n_bits))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


def resnet18_quan(pretrained=True, n_bits=8, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], n_bits, **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def resnet18_quan_mid(pretrained=True, n_bits=8, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_mid(BasicBlock, [2, 2, 2, 2], n_bits, **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet18'])
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

def resnet34_quan(pretrained=True, n_bits=8, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], n_bits, **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = model.state_dict()
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


if __name__ == '__main__':

    import os
    import torchvision.datasets as dset
    import torchvision.transforms as transforms

    os.environ["CUDA_VISIBLE_DEVICES"] = '3'

    net = resnet18_quan(True, 4).cuda()
    print(net)
    for m in net.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            print(m.N_bits)
    exit()
    # block for weight reset
    # for m in net.modules():
    #     if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
    #         m.__reset_stepsize__()
    #         m.__reset_weight__()
    #
    # print(net)


    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])  # here is actually the validation dataset

    test_dir = os.path.join('/data/dataset/imagenet/', 'val')
    test_data = dset.ImageFolder(test_dir, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=256,
                                              shuffle=True,
                                              num_workers=4,
                                              pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss().cuda()
    log = open(os.path.join('tmp_log.txt'), 'w')
    # validate(test_loader, net, criterion, log)
    #
    # exit()

    # update the step_size once the model is loaded. This is used for quantization.
    for m in net.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            # simple step size update based on the pretrained model or weight init
            m.__reset_stepsize__()
    step_param = [
        param for name, param in net.named_parameters() if 'step_size' in name
    ]
    optimizer_quan = torch.optim.SGD(step_param,
                                     lr=0.01,
                                     momentum=0.9,
                                     weight_decay=0,
                                     nesterov=True)

    for m in net.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            for i in range(
                    200
            ):  # runs 200 iterations to reduce quantization error
                optimizer_quan.zero_grad()
                weight_quan = quantize(m.weight, m.step_size,
                                       m.half_lvls) * m.step_size
                loss_quan = F.mse_loss(weight_quan,
                                       m.weight,
                                       reduction='mean')
                loss_quan.backward()
                optimizer_quan.step()

    for m in net.modules():
        if isinstance(m, quan_Conv2d):
            print(m.step_size.data.item(),
                  (m.step_size.detach() * m.half_lvls).item(),
                  m.weight.max().item())

    # block for weight reset
    for m in net.modules():
        if isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            m.__reset_weight__()

    validate(test_loader, net, criterion, log)

