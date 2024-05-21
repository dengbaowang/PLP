import os
import builtins
import time
import shutil
import logging
import argparse
import torch
import warnings
import random
from torch import nn
from torch.backends import cudnn
import numpy as np
import torchvision
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from dataset_tiny import dataloader_generate
from resnet_tiny import resnet18, resnet50
from util_tiny import validate,AverageMeter,ProgressMeter,accuracy

from ece_loss import ECELoss
from my_sgd import SGD

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='To classify tinyimagenet/imagenet')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--dataset', type=str, choices=['tinyimagenet','imagenet'], default='tinyimagenet')
parser.add_argument('--model', type=str, choices=['resnet18','resnet34','resnet50', 'resnet101', 'resnet110', 'resnet152','densenet121','densenet169','densenet161'], default='resnet50')
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--wd', default=1e-4,type=float)
parser.add_argument('--bs',default=64,type=int)
parser.add_argument('--nw',default=64,type=int,help='number of workers for dataloader')
parser.add_argument('--data-dir', default='/apdcephfs/share_1364275/dengbaowang/data/tinyimagenet/tiny-imagenet-200/', type=str)
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',help='use pre-trained model')

parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--exp-dir', default='./saved_models/', type=str,help='experiment directory for saving checkpoints and logs')
parser.add_argument('--resume', default='', type=str,help='path to latest checkpoint (default: none)')

parser.add_argument('--world_size', default=1, type=int,help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,help='node rank for distributed training')
parser.add_argument('--dist_url', default='tcp://127.0.0.1:8080', type=str,help='url used to set up distributed training')
parser.add_argument('--dist_backend', default='nccl', type=str,help='distributed backend')
parser.add_argument('--seed',default=8888,type=int)
parser.add_argument('--gpu', default=0, type=int,help='GPU id to use.')
parser.add_argument('--multiprocessing_distributed', action='store_true',
                help='Use multi-processing distributed training to launch '
                     'N processes per node, which has N GPUs. This is the '
                     'fastest way to use PyTorch for either single node or '
                     'multi node data parallel training')

parser.add_argument("--PLP",
                    action="store_true",
                    dest="PLP",
                    help="Whether to use PLP")

parser.add_argument('--gamma',
                    default=1.0,
                    type=float,
                    help='Coefficient of PLP')

parser.add_argument('--seed_val',
                    default=101,
                    type=int,
                    help='Coefficient of Focal Loss')

args = parser.parse_args()
logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[logging.StreamHandler()])
logging.info(args.__dict__)


def classification(gpu,ngpus_per_node,args):
    cudnn.benchmark = True
    args.gpu = gpu

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
    if args.gpu is not None:
        print("==> Use GPU: {} for training".format(args.gpu))
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        #builtins.print = print_pass
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    #######################load model#######################
    print('==> Building model: {}'.format(args.model))
    if args.model == 'resnet18':
        model = resnet18(pretrained=args.pretrained)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 200)
    elif args.model == 'resnet50':
        model = resnet50(pretrained=args.pretrained)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 200)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.bs = int(args.bs / ngpus_per_node)
            args.nw = int((args.nw + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # comment out the following line for debugging
        #raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    print('==> Preparing dataloader: {}'.format(args.dataset))
    trainloader, train_loader_forpool, valloader, testloader, num_classes, channel,train_sampler = dataloader_generate(args)
    print('Number of train dataset: ', len(trainloader.dataset))
    print('Number of validation dataset: ', len(testloader.dataset))
    
    ########################################################## PLP related code ##############################################################
    if args.model == 'resnet18':
        layer_idx = [3, 9, 15, 24, 30, 39, 45, 54, 57, 60, 62]
        #layer_idx = [3, 15, 30, 45, 62] #block-wise 

    elif args.model == 'resnet50':
        layer_idx = [161, 150, 141, 129, 120, 102, 93, 84, 72, 54, 33][::-1]
        #layer_idx = [161, 129, 72, 33, 3][::-1] #block-wise

    epoch_froze = (np.linspace(0,1,len(layer_idx)+1)**args.gamma*args.epochs).astype(int).tolist()[::-1]
    print('Starting freezing epoch list: ', epoch_froze)
    layer_name = []
    c = 0
    for name, param in model.named_parameters():
        layer_name.append(name)
        c += 1

    # Train loop
    for epoch in range(args.start_epoch, args.epochs):
        if args.PLP == True and epoch == epoch_froze[-1]:
            idx = layer_idx.pop()
            epoch_froze.pop()
        elif args.PLP == False:
            idx = layer_idx[-1] # just use a large number to make sure that no freezing operation

        print('Current lr {:.5e}, freezing parameters from {} to {}'.format(optimizer.param_groups[0]['lr'], layer_name[idx-1], layer_name[-1]))
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args.lr)
        trainloss = train(trainloader, model,optimizer,epoch,criterion, idx)
        valacc = validate(testloader, model, criterion, epoch)
        evaluate_TS(model, valloader, testloader)
    ########################################################## PLP related code end ##############################################################


def train(train_loader, model, optimizer, epoch, criterion, idx):
    batch_time = AverageMeter('Time', ':.2f')
    data_time = AverageMeter('Data', ':.2f')
    losses = AverageMeter('loss', ':.2f')
    top1 = AverageMeter('Acc@1', ':.2f')
    top5 = AverageMeter('Acc@5', ':.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    end = time.time()

    model.train()
    for i, (image,label) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        label_bi = torch.zeros(image.size(0), 200).scatter_(1, label.view(-1,1).long(), 1)
        label_bi = label_bi.cuda()
        image,label=image.cuda(),label.cuda()
        # loss
        output = model(image)
        loss = torch.mean(criterion(output, label))

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, label, topk=(1, 5))
        losses.update(loss.item(), image.size(0))
        top1.update(acc1.item(), image.size(0))
        top5.update(acc5.item(), image.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step(idx)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            progress.display(i)

    return losses.avg

def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.5 ** (epoch // 30))
    #lr = init_lr * ((0.1 ** int(epoch >= 100)) * (0.1 ** int(epoch >= 150))) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def evaluate_TS(model, val_loader, test_loader):
    ece_criterion = ECELoss().cuda()
    model.eval()
    with torch.no_grad():
        best_t = search_t(model, val_loader)
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
        print('Searched Temperature: {}, ECE on Test Data After TS: {}'.format(round(best_t,4), round(ece_after.item(),4)))
    return best_t, ece_after.item()

def search_t(model, val_loader):
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
    #print('Searched Temperature on Validation Data: {}'.format(best_temp))
    return best_temp



if __name__=='__main__':
    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(classification, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        classification(args.gpu, ngpus_per_node, args)
