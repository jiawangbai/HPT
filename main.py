import warnings

warnings.filterwarnings("ignore")

import argparse
import os
import time

import torch
import torchvision
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import quan_resnet_cifar, quan_resnet_imagenet, quan_vgg_cifar
from utils import *
from models.quantization import *
from admm_opt import *
import numpy as np
import config
from models.model_wrap import Attacked_model

parser = argparse.ArgumentParser(description='Performing Hardly Perceptible Trojan Attack (HPT)')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18_quan_8')
parser.add_argument('--dataset', '--da', dest="dataset",
                    default="cifar10", choices=["cifar10", "svhn", "imagenet"], type=str)
parser.add_argument('--batch-size', dest='batch_size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--save-dir', dest='save_dir', type=str, default="save_tmp")
parser.add_argument('--target-class', dest='target_class', type=int, default=0)

parser.add_argument('--gamma', default=1000, type=float)
parser.add_argument('--b-bits', '-b_bits', default=10, type=int)
parser.add_argument('--n-clean', '-n_clean', default=128, type=int)

parser.add_argument('--epsilon', '-epsilon', default=0.04, type=float)
parser.add_argument('--kappa', '-kappa', default=0.01, type=float)

parser.add_argument('--init-iters', '-init_iters', default=500, type=int)
parser.add_argument('--init-lr-grid', '-init_lr_grid', default=0.01, type=float)
parser.add_argument('--init-lr-noise', '-init_lr_noise', default=0.01, type=float)

parser.add_argument('--ext-num-iters', '-ext_num_iters', default=3000, type=int)
parser.add_argument('--inn-num-iters', '-inn_num_iters', default=5, type=int)
parser.add_argument('--lr-weight', '-lr_weight', default=0.0001, type=float)
parser.add_argument('--lr-grid', '-lr_grid', default=0.00001, type=float)
parser.add_argument('--lr-noise', '-lr_noise', default=0.00001, type=float)
parser.add_argument('--initial-rho', '-initial_rho', default=0.0001, type=float)
parser.add_argument('--max-rho', '-max_rho', default=100, type=float)
parser.add_argument('--stop-threshold', '-stop_threshold', default=0.0001, type=float)

parser.add_argument('--gpu-id', dest='gpu_id', type=str, default='0')

args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.system("mkdir " + args.save_dir)
sys.stdout = Logger(os.path.join(args.save_dir, "log.txt"))

print(args)

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

if args.dataset == "cifar10":
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

    test_dir = config.cifar_root
    val_set = datasets.CIFAR10(root=test_dir, train=False, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=args.batch_size, shuffle=False, pin_memory=True)

elif args.dataset == "svhn":
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    test_dir = config.svhn_root

    val_set = datasets.SVHN(root=config.svhn_root, split='test', download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            ]))

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=args.batch_size, shuffle=False, pin_memory=True)

elif args.dataset == "imagenet":
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    test_dir = os.path.join(config.imagenet_root, 'validation')
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    val_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

class_num = 10 if args.dataset == "cifar10" or args.dataset == "svhn" else 1000

model_path = config.model_root + args.dataset + "/" + args.arch + "/"

n_bits = int(args.arch.split("_")[-1])
if args.dataset == "cifar10":
    if "resnet" in args.arch:
        model = torch.nn.DataParallel(quan_resnet_cifar.__dict__[args.arch[:-2] + "_mid"](class_num, n_bits))
    elif "vgg" in args.arch:
        model = quan_vgg_cifar.__dict__[args.arch[:-2] + "_mid"](class_num, n_bits)
        model.features = torch.nn.DataParallel(model.features)
elif args.dataset == "svhn":
    if "vgg" in args.arch:
        model = quan_vgg_cifar.__dict__[args.arch[:-2] + "_mid"](class_num, n_bits)
        model.features = torch.nn.DataParallel(model.features)
else:
    if "resnet" in args.arch:
        model = torch.nn.DataParallel(
            quan_resnet_imagenet.__dict__[args.arch[:-2] + "_mid"](False, n_bits=n_bits, num_classes=class_num))

checkpoint = torch.load(model_path + "model.th")
model.load_state_dict(checkpoint["state_dict"])

if isinstance(model, torch.nn.DataParallel):
    model = model.module

for m in model.modules():
    if isinstance(m, quan_Linear):
        m.__reset_stepsize__()
        m.__reset_weight__()
model.cuda()

load_model = Attacked_model(model, args.dataset, args.arch)
load_model.cuda()
load_model.eval()

if "resnet" in args.arch:
    model = torch.nn.DataParallel(model)
    load_model.model = torch.nn.DataParallel(load_model.model)

criterion = nn.CrossEntropyLoss().cuda()

np.random.seed(512)
clean_idx = np.random.choice(len(val_loader.dataset), args.n_clean, replace=False)


if args.dataset == "cifar10":
    normalize = Normalize(mean=[0.4914, 0.4822, 0.4465],
                          std=[0.2023, 0.1994, 0.2010])
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=(10, 10)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
    ])
    clean_dataset = ImageFolder_cifar10(val_loader.dataset.data[clean_idx],
                                      np.array(val_loader.dataset.targets)[clean_idx],
                                      transform=transform)

    clean_loader = torch.utils.data.DataLoader(
        dataset=clean_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True)

elif args.dataset == "svhn":
    normalize = Normalize(mean=[0.5, 0.5, 0.5],
                          std=[0.5, 0.5, 0.5])
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=(10, 10)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
    ])

    clean_dataset = ImageFolder_svhn(val_loader.dataset.data[clean_idx],
                                   np.array(val_loader.dataset.labels)[clean_idx],
                                   transform=transform)

    clean_loader = torch.utils.data.DataLoader(
        dataset=clean_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True)

elif args.dataset == "imagenet":

    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    paths = [val_loader.dataset.imgs[i][0] for i in clean_idx]
    targets = [val_loader.dataset.imgs[i][1] for i in clean_idx]

    clean_dataset = ImageFolder_imagenet(paths, targets, transform=transform)

    clean_loader = torch.utils.data.DataLoader(
        dataset=clean_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True)


def main():

    attacked_model, grid_total, delta_noise = \
        admm_opt(load_model, clean_loader, normalize, args)

    test_acc, _, _ = validate(val_loader, nn.Sequential(normalize, load_model), criterion)
    post_attack_test_acc, _, _ = validate(val_loader, nn.Sequential(normalize, attacked_model), criterion)
    attack_success_rate, _, _ = validate_tro(val_loader, grid_total, delta_noise, args.target_class,
                                               nn.Sequential(normalize, attacked_model), criterion)
    n_bit = torch.norm(attacked_model.w_twos.data.view(-1) - load_model.w_twos.data.view(-1), p=0).item()

    print("test_acc:{0:.4f} post_attack_test_acc:{1:.4f} attack_success_rate:{2:.4f} bit_flips:{3}".format(
          test_acc, post_attack_test_acc, attack_success_rate, n_bit))



if __name__ == '__main__':
    main()
