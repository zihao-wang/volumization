import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

from models import get_model
from ioutils import get_dataset, Logger

import math
import torch
import argparse
from vadam import Vadam


parser = argparse.ArgumentParser(description='Volumization Evaluation')

# task related
parser.add_argument('--dataset', type=str, default="IMDB")
parser.add_argument('--model', type=str, default="LSTM")
parser.add_argument('--task_id', type=str, default='default')
# optimizer related
parser.add_argument('--eps', type=float, default=1.9, help="epsilon")
parser.add_argument('--lr', type=float, default=2e-3, help="learning rate")
parser.add_argument('--v', type=float, default=0.1, help="limitation of volumization")
parser.add_argument('--batch_size', type=int, default=32, help="batch size")
parser.add_argument("--num_epochs", type=int, default=20, help="number of epochs")
# noise ratio
parser.add_argument('--noise_ratio', type=float, default=0.1, help="noise ratio")

params = parser.parse_args()
model_for_data = {"IMDB": ["LSTM"]}
assert params.model in model_for_data[params.dataset]

timestamp = time.strftime("%y%m%d-%H%M%S-", time.localtime())
log_dir_name = params.dataset
task_name = timestamp + params.model
train_logger = Logger(task_name=task_name+'train',
                      dir_name=log_dir_name,
                      heading='epoch train_loss train_acc val_loss val_acc test_loss test_acc'.split())


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    if torch.cuda.is_available():
        model.cuda()
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if text.size()[0] is not batch_size:
            continue
        optim.zero_grad()
        prediction = model(text)

        output = F.log_softmax(prediction, dim=1)
        loss = F.nll_loss(output, target)

        num_corrects = (torch.max(prediction[:, :2], 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        #clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        
        if steps % 100 == 0:
            print('Epoch:', epoch+1, 'Idx:', idx+1, 'Training Loss:', loss.item(), 'Training Accuracy:', acc.item())
        
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()
        
    return total_epoch_loss/len(train_iter), total_epoch_acc/len(train_iter)


def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch.text[0]
            if (text.size()[0] is not batch_size):
                continue
            target = batch.label
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text)
            loss = F.nll_loss(prediction, target)

            num_corrects = (torch.max(prediction[:, :2], 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)


def loss_theo(o, e):
    x = e * math.log(e) + (1 - e) * math.log((1 - e)/ (o - 1))
    print(-x)
    return x


if __name__ == "__main__":
    embedding, train_iter, valid_iter, test_iter = get_dataset(params.dataset,
                                                               rate=params.noise_ratio,
                                                               batch_size=params.batch_size)
    model_params = {}
    if params.dataset == "IMDB":
        model_params["output_size"] = 2
        model_params["hidden_size"] = 256

    if params.model == "LSTM":
        model_params["weights"] = {"weights": embedding}

    model = get_model(params.model, **model_params)

    optim = Vadam(model.parameters(), lr=params.lr, eps=1e-15, v=params.v)

    thr = loss_theo(params.eps, params.noise_ratio)
    for epoch in range(params.num_epochs):
        train_loss, train_acc = train_model(model, train_iter, epoch)
        val_loss, val_acc = eval_model(model, valid_iter)
        test_loss, test_acc = eval_model(model, test_iter)
        train_logger.append(epoch+1, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)

