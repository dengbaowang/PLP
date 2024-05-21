import os
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from collections import defaultdict
from torch.utils.data import Sampler, Dataset
import numpy as np
class DatasetWrapper(Dataset):
    # Additinoal attributes
    # - indices
    # - classwise_indices
    # - num_classes
    # - get_class

    def __init__(self, dataset, indices=None):
        self.base_dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices

        # torchvision 0.2.0 compatibility
        if torchvision.__version__.startswith('0.2'):
            if isinstance(self.base_dataset, datasets.ImageFolder):
                self.base_dataset.targets = [s[1] for s in self.base_dataset.imgs]
            else:
                if self.base_dataset.train:
                    self.base_dataset.targets = self.base_dataset.train_labels
                else:
                    self.base_dataset.targets = self.base_dataset.test_labels

        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.targets[self.indices[i]]
            self.classwise_indices[y].append(i)
        self.num_classes = max(self.classwise_indices.keys())+1

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.targets[self.indices[i]]



def dataloader_generate(args):
    print('Data Preparation')
    num_workers=8

    if args.dataset == 'tinyimagenet':
        num_classes=200
        channel=3

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        train_val_dataset_dir = os.path.join('../data/tiny-imagenet-200', "train")
        test_dataset_dir = os.path.join('../data/tiny-imagenet-200', "val")

        trainset1 = datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train)
        trainset2 = datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_test)
        testset = datasets.ImageFolder(root=test_dataset_dir, transform=transform_test)
        indices = np.random.RandomState(args.seed_val).permutation(len(trainset1.targets))
        print("Totol training data number: ", len(trainset1.targets))
        indices1 = indices[:len(trainset1.targets)-10000] 
        indices2 = indices[len(trainset1.targets)-10000:] 
        trainset = torch.utils.data.Subset(trainset1, indices1)
        valset = torch.utils.data.Subset(trainset2, indices2)

    elif args.dataset == 'tinyimagenet-c':
        num_classes=200
        channel=3

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        test_dataset_dir = os.path.join('../../data/tiny-imagenet-200/' + corrupion_type, "val")

        testset = datasets.ImageFolder(root=test_dataset_dir, transform=transform_test)

    elif args.dataset=='imagenet':
        num_classes=1000
        channel=3

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),#224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),#256
            transforms.CenterCrop(224),#224
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        train_val_dataset_dir = os.path.join('/seu_share2/datasets/imagenet-c/TemPred_imagenet/imagenet', "train")
        test_dataset_dir = os.path.join('/seu_share2/datasets/imagenet-c/TemPred_imagenet/imagenet', "val")

        trainset1 = datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train)
        trainset2 = datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_test)
        testset = datasets.ImageFolder(root=test_dataset_dir, transform=transform_test)

        indices = np.random.RandomState(args.seed_val).permutation(len(trainset1.targets))
        print("Totol training data number: ", len(trainset1.targets))
        indices1 = indices[:len(trainset1.targets)-20000] 
        indices2 = indices[len(trainset1.targets)-20000:] 
        trainset = torch.utils.data.Subset(trainset1, indices1)
        valset = torch.utils.data.Subset(trainset2, indices2)

    else:
        assert "Unknown dataset"

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(valset)
    else:
        train_sampler = None
        val_sampler = None

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=(train_sampler is None),
                                              num_workers=args.nw, pin_memory=True, sampler=train_sampler,drop_last=True)
    trainloader_forpool = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=(train_sampler is None),
                                              num_workers=args.nw, pin_memory=True, sampler=train_sampler,drop_last=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.bs, shuffle=(val_sampler is None),
                                              num_workers=args.nw, pin_memory=True, sampler=val_sampler,drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=args.nw,
                                            pin_memory=True)

    return trainloader, trainloader_forpool, valloader, testloader,num_classes,channel,train_sampler
