import argparse
import os
import time
import json
import hashlib

from tqdm import tqdm
import torch
from torch import nn
import numpy as np

from models import get_model
from ioutils import get_dataset, Logger
from vadam import Vadam2
from vlaprop import VLaProp
from vsgd import VSGD

parser = argparse.ArgumentParser(description='Volumization Evaluation')

# task related
parser.add_argument('--dataset', type=str, default="MNIST")
parser.add_argument('--model', type=str, default="DNN")
parser.add_argument('--task_id', type=str, default='default')
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--save', type=bool, default=False)
# optimizer related
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--v', type=float, default=1, help="limitation of volumization")
parser.add_argument('--alpha', type=float, default=1.0, help="alpha")
parser.add_argument('--auto', type=float, default=True, help="Kaiming-V or not")
parser.add_argument('--weight_decay', type=float, default=0, help="default is None")
parser.add_argument('--batch_size', type=int, default=128, help="batch size")
parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
parser.add_argument("--lq", type=bool, default=False)
parser.add_argument("--optimizer", type=str, default='adam')
# noise ratio
parser.add_argument('--noise_ratio', type=float, default=0.0, help="noise ratio")

params = parser.parse_args()
model_for_data = {"MNIST": ["DNN"],
                  "IMDB": ["LSTMATT", "LSTM"],
                  "CIFAR10": ["ResNet18"]}
assert params.model in model_for_data[params.dataset]

if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(params.cuda))
else:
    device = 'cpu'

timestamp = time.strftime("%y%m%d-%H%M%S", time.localtime())
log_dir_name = os.path.join('log', params.dataset)
param_dict = vars(params)
config_hash = hashlib.sha224(json.dumps(param_dict).encode()).hexdigest()
task_name = params.task_id + '-' + timestamp + config_hash[:4] + "-" + params.model


train_logger = Logger(task_name=task_name,
                      dir_name=log_dir_name,
                      heading=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc'],
                      )
with open(os.path.join(log_dir_name, task_name + ".meta"), mode='wt') as f:
    json.dump(param_dict, f)


def lq_loss(logits, Y, q=0.7):
    index1 = torch.arange(len(Y))
    prob = torch.nn.functional.softmax(logits, 1)
    loss = 1 - prob[index1, Y.view(-1)] ** q
    # print(loss.mean())
    return loss.mean() / q


if params.lq:
    criterion = lq_loss
else:
    criterion = nn.CrossEntropyLoss()


def train_model(model, _iter):
    total_epoch_loss = 0
    total_epoch_acc = 0

    model.train()
    for X, Y in tqdm(_iter):
        X, Y = X.to(device), Y.to(device)

        optim.zero_grad()
        logits = model(X)
        preds = torch.max(logits, 1)[1].view(Y.size())
        loss = criterion(logits, Y)
        loss.backward()
        optim.step()

        num_corrects = (preds == Y).float().sum()
        acc = 100.0 * num_corrects / len(Y)

        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss / len(_iter), total_epoch_acc / len(_iter)


def eval_model(model, _iter):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for idx, (X, Y) in enumerate(_iter):
            X = X.to(device)
            Y = Y.to(device)

            logits = model(X)
            preds = torch.max(logits, 1)[1].view(Y.size())
            loss = criterion(logits, Y)

            num_corrects = (preds == Y).float().sum()
            acc = 100.0 * num_corrects / len(Y)

            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss / len(_iter), total_epoch_acc / len(_iter)


if __name__ == "__main__":
    embedding, train_iter, valid_iter, test_iter = get_dataset(params.dataset,
                                                               rate=params.noise_ratio,
                                                               batch_size=params.batch_size)
    model_params = {}
    if params.dataset == "IMDB":
        model_params["output_size"] = 2
        model_params["hidden_size"] = 256

    if params.model == "LSTM" or params.model=="LSTMATT":
        model_params["weights"] = embedding

    if params.dataset == "CIFAR100":
        model_params["num_classes"] = 100

    model = get_model(params.model, **model_params)
    model.to(device)

    if params.optimizer == "laprop":
        optim = VLaProp(model.parameters(), lr=params.lr, eps=1e-15,
                       v=params.v, alpha=params.alpha, auto_v=params.auto,
                       weight_decay=params.weight_decay)
    elif params.optimizer == "adam":
        optim = Vadam2(model.parameters(), lr=params.lr, eps=1e-15,
                        v=params.v, alpha=params.alpha, auto_v=params.auto,
                        weight_decay=params.weight_decay)
    elif params.optimizer == "sgd":
        optim = VSGD(model.parameters(), lr=params.lr,
                       v=params.v, alpha=params.alpha, auto_v=params.auto,
                       weight_decay=params.weight_decay)

    test_acc_list = []
    for epoch in range(params.num_epochs):
        train_loss, train_acc = train_model(model, train_iter)
        val_loss, val_acc = eval_model(model, valid_iter)
        test_loss, test_acc = eval_model(model, test_iter)
        test_acc_list.append(test_acc)
        train_logger.append(epoch + 1, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)
    print(np.max(test_acc_list), np.mean(test_acc_list[-10:]))
    if params.save:
        model.cpu()
        torch.save(model.state_dict(), "{}nr{}v{}alpha{}.pt".format(task_name, params.noise_ratio, params.v, params.alpha))

