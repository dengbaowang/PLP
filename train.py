import argparse
import sys
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import random
import resnet as resnet
from ece_loss import ECELoss
from my_sgd import SGD

model_names = sorted(
    name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
    and callable(resnet.__dict__[name]))

parser = argparse.ArgumentParser(description='CIFAR10/100')
parser.add_argument('--arch',
                    metavar='ARCH',
                    default='resnet32',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')

parser.add_argument('-j',
                    '--workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 8)')

parser.add_argument('--epochs',
                    default=350,
                    type=int,
                    metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--batch-size',
                    default=128,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 512)')

parser.add_argument('--lr',
                    '--learning-rate',
                    default=0.1,
                    type=float,
                    metavar='LR',
                    help='initial learning rate')

parser.add_argument('--momentum',
                    default=0.9,
                    type=float,
                    metavar='M',
                    help='momentum')

parser.add_argument('--weight-decay',
                    '--wd',
                    default=1e-4,
                    type=float,
                    metavar='W',
                    help='weight decay (default: 5e-4)')

parser.add_argument('--dataset',
                    help='',
                    default='cifar10',
                    choices=['cifar10','cifar100','svhn'],
                    type=str)

parser.add_argument("--PLP",
                    action="store_true",
                    dest="PLP",
                    help="Whether to use PLP")

parser.add_argument('--gamma',
                    default=1.0,
                    type=float,
                    help='Coefficient of L1 Norm')

parser.add_argument('--seed',
                    default=101,
                    type=int,
                    help='seed for validation data split')
best_prec1 = 0
args = parser.parse_args()
print(args)

if args.dataset == 'cifar10':
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    num_classes = 10
    all_train_data1 = datasets.CIFAR10(
        root='../data',
        train=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]),
        download=True)
    all_train_data2 = datasets.CIFAR10(
        root='../data',
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        download=True)
    test_data = datasets.CIFAR10(
        root='../data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        download=True)
    
    indices = np.random.RandomState(args.seed).permutation(len(all_train_data1.targets))
    indices1 = indices[:45000] 
    indices2 = indices[45000:] 
    train_data = torch.utils.data.Subset(all_train_data1, indices1)
    val_data = torch.utils.data.Subset(all_train_data2, indices2)

elif args.dataset == 'cifar100':
    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                 std=[0.267, 0.256, 0.276])
    num_classes = 100
    all_train_data1 = datasets.CIFAR100(
        root='../data',
        train=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]),
        download=True)
    all_train_data2 = datasets.CIFAR100(
        root='../data',
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        download=True)
    test_data = datasets.CIFAR100(
        root='../data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        download=True)
    
    indices = np.random.RandomState(args.seed).permutation(len(all_train_data1.targets))
    indices1 = indices[:45000] 
    indices2 = indices[45000:] 
    train_data = torch.utils.data.Subset(all_train_data1, indices1)
    val_data = torch.utils.data.Subset(all_train_data2, indices2)

elif args.dataset == 'svhn':
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])

    num_classes = 10
    all_train_data = datasets.SVHN(
        root='../data/svhn',
        split='train',
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        download=True)
    test_data = datasets.SVHN(
        root='../data/svhn',
        split='test',
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        download=True)
    
    indices = np.random.RandomState(args.seed).permutation(len(all_train_data.labels))
    indices1 = indices[:68257] 
    indices2 = indices[68257:]
    train_data = torch.utils.data.Subset(all_train_data, indices1)
    val_data = torch.utils.data.Subset(all_train_data, indices2)

train_loader = torch.utils.data.DataLoader(train_data,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=True)

val_loader = torch.utils.data.DataLoader(val_data,
    batch_size=500,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True)

test_loader = torch.utils.data.DataLoader(test_data,
    batch_size=500,
    shuffle=False,
    num_workers=args.workers,
    pin_memory=True)


def main():
    model = resnet.__dict__[args.arch](num_classes=num_classes)
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = SGD(model.parameters(),
                                args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[150, 250], last_epoch= -1)

    ########################################################## PLP related code ##############################################################
    if args.arch == 'resnet18':
        layer_idx = [3, 9, 15, 24, 30, 39, 45, 54, 57, 60, 62]
        #layer_idx = [3, 15, 30, 45, 62] #block-wise 

    elif args.arch == 'resnet50':
        layer_idx = [161, 150, 141, 129, 120, 102, 93, 84, 72, 54, 33][::-1]
        #layer_idx = [161, 129, 72, 33, 3][::-1] #block-wise

    elif args.arch == 'vgg11':
        layer_idx = [34, 32, 20, 12, 8, 4][::-1]
    
    epoch_froze = (np.linspace(0,1,len(layer_idx)+1)**args.gamma*args.epochs).astype(int).tolist()[::-1]
    print('Starting freezing epoch list: ', epoch_froze)
    layer_name = []
    c = 0
    for name, param in model.named_parameters():
        layer_name.append(name)
        c += 1

    num_epoch = args.epochs
    for epoch in range(0, num_epoch):
        if args.PLP == True and epoch == epoch_froze[-1]:
            idx = layer_idx.pop()
            epoch_froze.pop()
        elif args.PLP == False:
            idx = layer_idx[-1] # just use a large number to make sure that no freezing operation

        print('Current lr {:.5e}, freezing parameters from {} to {}'.format(optimizer.param_groups[0]['lr'], layer_name[idx-1], layer_name[-1]))
        train(model, criterion, optimizer, epoch, idx)
        lr_scheduler.step()
        evaluate(model)
        if epoch % 10 == 0:
            evaluate_TS(model)
    ########################################################## PLP related code end ##############################################################

def train(model, criterion, optimizer, epoch, idx):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_var = input.cuda()
        target_var = target.cuda()
        
        output = model(input_var)
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step(idx)

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 20 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch,
                      i,
                      len(train_loader),
                      batch_time=batch_time,
                      data_time=data_time,
                      loss=losses,
                      top1=top1))

def evaluate(model):
    model.eval()
    correct = 0
    ece_criterion = ECELoss().cuda()
    with torch.no_grad():
        output_nosoftmax_list = []
        target_list = []
        for data_i, target_i in test_loader:
            data_i, target_i = data_i.cuda(), target_i.cuda()
            output_nosoftmax_i = model(data_i)
            output_nosoftmax_list.append(output_nosoftmax_i)
            target_list.append(target_i)
        
        output_nosoftmax = torch.cat(output_nosoftmax_list, 0)
        target = torch.cat(target_list, 0)
        output = torch.nn.functional.softmax(output_nosoftmax, dim=1)
        pred = output.argmax(
            dim=1,
            keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        ece = ece_criterion(output_nosoftmax, target)
    print('\nTest set: Accuracy: {:.2f}%    ECE (without post-hoc calibration): {:.4f}'.format(100. * correct /
                                                   len(test_loader.dataset), ece.item()))
def evaluate_TS(model):
    ece_criterion = ECELoss(n_bins=15).cuda()
    model.eval()
    with torch.no_grad():
        best_t = search_t(model)
        output_nosoftmax_list = []
        target_list = []
        for data_i, target_i in test_loader:
            data_i, target_i = data_i.cuda(), target_i.cuda()
            output_nosoftmax_i = model(data_i)
            output_nosoftmax_list.append(output_nosoftmax_i)
            target_list.append(target_i)
        
        output_nosoftmax = torch.cat(output_nosoftmax_list, 0)
        target = torch.cat(target_list, 0)
        ece_before = ece_criterion(output_nosoftmax, target)
        calibrated_output = output_nosoftmax / best_t
        ece_after = ece_criterion(calibrated_output, target)
        print('\nECE on Test Data After TS Calibration: ', round(ece_after.item(),4))

def search_t(model):
    model.eval()
    correct = 0
    ece_criterion = ECELoss().cuda()
    best_ece = 1000
    with torch.no_grad():
        output_nosoftmax_list = []
        target_list = []
        for data, target in val_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            else:
                data, target = data.cpu(), target.cpu()
            output_nosoftmax = model(data)
            output_nosoftmax_list.append(output_nosoftmax)
            target_list.append(target)
        
        output_nosoftmax = torch.cat(output_nosoftmax_list, 0)
        target = torch.cat(target_list, 0)

        output = torch.nn.functional.softmax(output_nosoftmax, dim=1)
        pred = output.argmax(
            dim=1,
            keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        for i in range(0, 500, 1):
            t_i = 0.01 * i
            ece = ece_criterion(output_nosoftmax / t_i, target)
            if ece < best_ece and ece != 0:
                best_temp = t_i
                best_ece = ece
    print('\nSearched Temperature on Validation Data: ', best_temp)
    return best_temp

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()