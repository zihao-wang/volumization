# _*_ coding: utf-8 _*_

import os
import random
import numpy as np
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
import torchvision
from torchvision.transforms import transforms


def get_dataset(dataset, **kwargs):
    if dataset == "IMDB":
        return load_IMDB(**kwargs)
    elif dataset == "CIFAR10":
        return load_CIFAR10(**kwargs)
    elif dataset == "MNIST":
        return load_MNIST(**kwargs)
    else:
        raise IOError("unknown dataset")


def load_IMDB(rate=0.2, batch_size=32):
    TEXT = data.Field(sequential=True,
                      tokenize=lambda x: x.split(),
                      lower=True,
                      include_lengths=True,
                      batch_first=True,
                      fix_length=200)
    LABEL = data.LabelField()
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    train_size = len(train_data)
    sample = random.sample(range(train_size), int(rate * train_size))
    for i in sample:
        if train_data[i].label == 'pos':
            train_data[i].label = 'neg'
        else:
            train_data[i].label = 'pos'

    TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300))
    LABEL.build_vocab(train_data)

    train_data, valid_data = train_data.split()
    # Further splitting of training_data to create new training_data & validation_data
    train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                                   batch_size=batch_size,
                                                                   sort_key=lambda x: len(x.text),
                                                                   repeat=False,
                                                                   shuffle=True)

    return TEXT.vocab.vectors, train_iter, valid_iter, test_iter


def load_CIFAR10(rate, batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='.data/cifar10', train=True, download=True, transform=transform_train)
    L = len(trainset)
    target_array = np.asarray(trainset.targets)
    thr = (1 - rate) - rate / 10
    target_array = np.where(np.random.rand(L) <= thr, target_array, np.random.randint(0, 10, L))
    trainset.targets = target_array.tolist()
    idx = np.random.permutation(L)
    train_idx = idx[: int(L * 0.8)]
    val_idx = idx[int(L * 0.8):]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              sampler=SubsetRandomSampler(train_idx))
    valloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            sampler=SubsetRandomSampler(val_idx))

    testset = torchvision.datasets.CIFAR10(root='.data/cifar10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return None, trainloader, valloader, testloader


def load_MNIST(rate, batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.MNIST(root=".data/mnist", train=True, download=True, transform=transforms.ToTensor())
    L = len(trainset)
    target_array = np.asarray(trainset.targets)
    thr = (1-rate) - rate/10
    target_array = np.where(np.random.rand(L) <= thr, target_array, np.random.randint(0, 10, L))
    trainset.targets = target_array.tolist()
    idx = np.random.permutation(L)
    train_idx = idx[: int(L * 0.8)]
    val_idx = idx[int(L * 0.8):]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              sampler=SubsetRandomSampler(train_idx))
    valloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            sampler=SubsetRandomSampler(val_idx))

    testset = torchvision.datasets.MNIST(root=".data/mnist", train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return None, trainloader, valloader, testloader



class Logger:
    def __init__(self, task_name, dir_name='log', heading=None):
        os.makedirs(dir_name, exist_ok=True)
        self.logfile = os.path.join(dir_name, task_name + ".log")
        assert not os.path.exists(self.logfile)
        # todo: need hash to discriminate different tasks
        self.heading = heading
        if heading:
            self.L = len(heading)
            with open(self.logfile, mode='at') as f:
                f.write(", ".join(heading) + '\n')
        else:
            self.L = -1

    def append(self, *entry, verbose=True):
        """
        entry is a tuple
        """
        assert self.L < 0 or self.L == len(entry)
        if self.heading and verbose:
            print("|".join("{}: {:.4f}".format(k, v) for k, v in zip(self.heading, entry)))
        with open(self.logfile, mode='at') as f:
            f.write(", ".join(str(e) for e in entry) + "\n")


if __name__ == "__main__":
    load_CIFAR10(0.5)
