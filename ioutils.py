# _*_ coding: utf-8 _*_

import os
import sys
import time
import random
import torch
from torch.nn import functional as F
import numpy as np
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe


def get_dataset(dataset, **kwargs):
    if dataset == "IMDB":
        return load_IMDB(**kwargs)
    else:
        raise IOError("unknown dataset")


def load_IMDB(rate=0.2, batch_size=32):
    """
    tokenizer : Breaks sentences into a list of words. If sequential=False, no tokenization is applied
    Field : A class that stores information about the way of preprocessing
    fix_length : An important property of TorchText is that we can let the input to be variable length, and TorchText will
                 dynamically pad each sequence to the longest sequence in that "batch". But here we are using fi_length which
                 will pad each sequence to have a fix length of 200.
                 
    build_vocab : It will first make a vocabulary or dictionary mapping all the unique words present in the train_data to an
                  idx and then after it will use GloVe word embedding to map the index to the corresponding word embedding.
                  
    vocab.vectors : This returns a torch tensor of shape (vocab_size x embedding_dim) containing the pre-trained word embeddings.
    BucketIterator : Defines an iterator that batches examples of similar lengths together to minimize the amount of padding needed.
    """
    
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


class Logger:
    def __init__(self,  task_name, dir_name='log', heading=None):
        os.makedirs(dir_name, exist_ok=True)
        self.logfile = os.path.join(dir_name, task_name + ".log")
        assert not os.path.exists(self.logfile)
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
            f.write(", ".join(str(e) for e in entry)+"\n")


if __name__ == "__main__":
    load_IMDB()
