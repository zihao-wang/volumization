# _*_ coding: utf-8 _*_

import os
import random
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data.sampler import SubsetRandomSampler
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
from PIL import Image
import torchvision
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder


def get_dataset(dataset, **kwargs):
    if dataset == "IMDB":
        return load_IMDB(**kwargs)
    elif dataset == "CIFAR10":
        return load_CIFAR10(**kwargs)
    elif dataset == "CIFAR100":
        return load_CIFAR100(**kwargs)
    elif dataset == "MNIST":
        return load_MNIST(**kwargs)
    elif dataset == "TinyImageNet":
        return load_TinyImageNet(**kwargs)
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

    return TEXT.vocab.vectors, BatchWrapper(train_iter), BatchWrapper(valid_iter), BatchWrapper(test_iter)


class BatchWrapper:
    def __init__(self, dataset):
        self.dataset = dataset
    def __iter__(self):
        for batch in self.dataset:
            yield batch.text[0], batch.label

    def __len__(self):
        return len(self.dataset)


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
    thr = 1 - rate * 10 / 9
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


def load_CIFAR100(rate=0, batch_size=128):
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

    trainset = torchvision.datasets.CIFAR100(root='.data/cifar100', train=True, download=True, transform=transform_train)
    L = len(trainset)
    target_array = np.asarray(trainset.targets)
    thr = 1 - rate * 100 / 99
    target_array = np.where(np.random.rand(L) <= thr, target_array, np.random.randint(0, 100, L))
    trainset.targets = target_array.tolist()
    idx = np.random.permutation(L)
    train_idx = idx[: int(L * 0.8)]
    val_idx = idx[int(L * 0.8):]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              sampler=SubsetRandomSampler(train_idx))
    valloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            sampler=SubsetRandomSampler(val_idx))

    testset = torchvision.datasets.CIFAR100(root='.data/cifar100', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return None, trainloader, valloader, testloader


def load_MNIST(rate=0, batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.MNIST(root=".data/mnist", train=True, download=True, transform=transforms.ToTensor())
    L = len(trainset)
    target_array = np.asarray(trainset.targets)
    thr = 1 - rate * 10 / 9
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


def load_TinyImageNet(batch_size=128, size=240, **kwargs):
    transform_train = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = TinyImageNet(root='.data', train=True, download=False, transform=transform_train)
    L = len(trainset)
    idx = np.random.permutation(L)
    train_idx = idx[: int(L * 0.8)]
    val_idx = idx[int(L * 0.8):]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              sampler=SubsetRandomSampler(train_idx))
    valloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            sampler=SubsetRandomSampler(val_idx))

    testset = TinyImageNet(root='.data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return None, trainloader, valloader, testloader


class TinyImageNet(torchvision.datasets.VisionDataset):
    training_file = 'training.pt'
    val_file = 'val.pt'
    raw_file = "tiny-imagenet-200"
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    # root = ".data"

    def __init__(self, root, train=True, transform=None, target_transform=None, download=True):
        super(TinyImageNet, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        self.totensor = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.train = train
        self.classes = None
        if download:
            self.download()
        if self.classes is None:
            self.process()

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.val_file
        self.data, self.targets, self.path = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        # print(img.shape, self.path[index])
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def __check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.val_file)))

    def download(self):
        if self.__check_exists():
            print("preprocessed")
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        if os.path.exists(os.path.join(self.raw_folder, self.raw_file)):
            return
        else:
            from torchvision.datasets.utils import download_and_extract_archive
            filename = self.url.split('/')[-1]
            download_and_extract_archive(self.url, download_root=self.raw_folder, filename=filename)

    def process(self):
        if self.__check_exists():
            print("preprocessed")
            return
        print("Processing...")
        # train_set
        train_folder = os.path.join(self.raw_folder, self.raw_file, "train")
        self.classes = {cname: ic for ic, cname in enumerate(sorted(os.listdir(train_folder)))}
        train_data, train_target, train_path = [], [], []
        print("train set")
        for cname, ic in tqdm(self.classes.items()):
            for img_name in os.listdir(os.path.join(train_folder, cname, "images")):
                train_target.append(ic)
                with Image.open(os.path.join(train_folder, cname, "images", img_name)).convert('RGB') as tmp:
                    train_data.append(self.totensor(tmp))
                train_path.append(os.path.join(train_folder, cname, "images", img_name))
        train_set = (train_data, train_target, train_path)
        with open(os.path.join(self.processed_folder, self.training_file), "wb") as f:
            torch.save(train_set, f)
        # val_set
        val_data, val_target, val_path = [], [], []
        val_folder = os.path.join(self.raw_folder, self.raw_file, 'val')
        val_dict = {}
        with open(os.path.join(val_folder, "val_annotations.txt"), mode='rt') as f:
            for line in f.readlines():
                img_name, cname, *_ = line.strip().split()
                val_dict[img_name] = self.classes[cname]
        for img_name in tqdm(os.listdir(os.path.join(val_folder, "images"))):
            with Image.open(os.path.join(val_folder, "images", img_name)).convert('RGB') as tmp:
                val_data.append(self.totensor(tmp))
            val_target.append(val_dict[img_name])
            val_path.append(os.path.join(val_folder, "images", img_name))
        val_set = (val_data, val_target, val_path)
        with open(os.path.join(self.processed_folder, self.val_file), "wb") as f:
            torch.save(val_set, f)
        print("Done!")


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
                f.write(",".join(heading) + '\n')
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
    train_set = TinyImageNet(root='.data', train=True)
    print(len(train_set))
    val_set = TinyImageNet(root='.data', train=False)
    print(len(val_set))
