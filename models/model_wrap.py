import torch.nn as nn
import torch
from bitstring import Bits
import torch.nn.functional as F


class Attacked_model(nn.Module):
    def __init__(self, model, dataset, arch):
        super(Attacked_model, self).__init__()

        self.model = model
        if isinstance(model, torch.nn.DataParallel):
            self.n_bits = model.module.n_bits
        else:
            self.n_bits = model.n_bits

        if dataset == "cifar10" or dataset == "svhn":
            if arch[:len("resnet18")] == "resnet18":
                self.w = model.linear.weight.data
                self.b = nn.Parameter(nn.Parameter(model.linear.bias.data), requires_grad=True)
                self.step_size = model.linear.step_size
            elif arch[:len("vgg16_bn")] == "vgg16_bn":
                self.w = model.classifier[6].weight.data
                self.b = nn.Parameter(model.classifier[6].bias.data, requires_grad=True)
                self.step_size = model.classifier[6].step_size
        elif dataset == "imagenet":
            if arch[:len("resnet18")] == "resnet18":
                self.w = model.fc.weight.data
                self.b = nn.Parameter(model.fc.bias.data, requires_grad=True)
                self.step_size = model.fc.step_size
            elif arch[:len("vgg16_bn")] == "vgg16_bn":
                self.w = model.classifier[6].weight.data
                self.b = nn.Parameter(model.classifier[6].bias.data, requires_grad=True)
                self.step_size = model.classifier[6].step_size

        self.w_twos = nn.Parameter(torch.zeros([self.w.shape[0], self.w.shape[1], self.n_bits]), requires_grad=True)

        base = [2**i for i in range(self.n_bits-1, -1, -1)]
        base[0] = -base[0]
        self.base = nn.Parameter(torch.tensor([[base]]).float())

        self.reset_w_twos()

    def forward(self, x):

        x = self.model(x)

        # covert w_twos to float
        w = self.w_twos * self.base
        w = torch.sum(w, dim=2) * self.step_size

        # calculate output
        x = F.linear(x, w, self.b)

        return x

    def reset_w_twos(self):
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                self.w_twos.data[i][j] = self.w_twos.data[i][j] + \
                    torch.tensor([int(b) for b in Bits(int=int(self.w[i][j]), length=self.n_bits).bin])




class Attacked_model2(nn.Module):
    def __init__(self, model, dataset, arch):
        super(Attacked_model2, self).__init__()

        self.model = model
        if isinstance(model, torch.nn.DataParallel):
            self.n_bits = model.module.n_bits
        else:
            self.n_bits = model.n_bits

        if dataset == "cifar10":
            if arch[:len("resnet20")] == "resnet20":
                self.conv = model.conv1.weight.data
                self.conv_step_size = model.conv1.step_size
                self.w = model.linear.weight.data
                self.b = nn.Parameter(nn.Parameter(model.linear.bias.data), requires_grad=True)
                self.w_step_size = model.linear.step_size
            elif arch[:len("vgg16_bn")] == "vgg16_bn":
                self.w = model.classifier[6].weight.data
                self.b = nn.Parameter(model.classifier[6].bias.data, requires_grad=True)
                self.step_size = model.classifier[6].step_size
        elif dataset == "imagenet":
            if arch[:len("resnet18")] == "resnet18":
                self.w = model.fc.weight.data
                self.b = nn.Parameter(model.fc.bias.data, requires_grad=True)
                self.step_size = model.fc.step_size
            elif arch[:len("vgg16_bn")] == "vgg16_bn":
                self.w = model.classifier[6].weight.data
                self.b = nn.Parameter(model.classifier[6].bias.data, requires_grad=True)
                self.step_size = model.classifier[6].step_size

        self.conv_twos = nn.Parameter(torch.zeros([self.conv.shape[0], self.conv.shape[1], self.conv.shape[2], self.conv.shape[3],
                                                  self.n_bits]), requires_grad=True)
        self.w_twos = nn.Parameter(torch.zeros([self.w.shape[0], self.w.shape[1], self.n_bits]), requires_grad=True)

        base = [2**i for i in range(self.n_bits-1, -1, -1)]
        base[0] = -base[0]
        self.base = nn.Parameter(torch.tensor([[base]]).float())

        self.reset_twos()

    def forward(self, x):
        conv = self.conv_twos * self.base
        conv = torch.sum(conv, dim=4) * self.conv_step_size

        x = F.conv2d(x, conv)
        x = self.model(x)

        # covert w_twos to float
        w = self.w_twos * self.base
        w = torch.sum(w, dim=2) * self.w_step_size

        # calculate output
        x = F.linear(x, w, self.b)

        return x

    def reset_twos(self):
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                self.w_twos.data[i][j] = self.w_twos.data[i][j] + \
                    torch.tensor([int(b) for b in Bits(int=int(self.w[i][j]), length=self.n_bits).bin])
        for i in range(self.conv.shape[0]):
            for j in range(self.conv.shape[1]):
                for k in range(self.conv.shape[2]):
                    for t in range(self.conv.shape[3]):
                        self.conv_twos.data[i][j][k][t] = self.conv_twos.data[i][j][k][t] + \
                            torch.tensor([int(b) for b in Bits(int=int(self.conv[i][j][k][t]), length=self.n_bits).bin])

