import argparse
import os
import time
import json
import hashlib

from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from models import get_model
from models.efficientnet_pytorch import EfficientNet
from ioutils import get_dataset, Logger
from vadam import Vadam2

parser = argparse.ArgumentParser(description='Train EfficientNet on TinyImageNet')

# task related
parser.add_argument('--dataset', type=str, default="TinyImageNet")
parser.add_argument('--model', type=str, default="efficientnet-b0")
parser.add_argument('--task_id', type=str, default='default')
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--load_from', type=str, default="")
# optimizer related
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--v', type=float, default=-1, help="limitation of volumization")
parser.add_argument('--alpha', type=float, default=0.5, help="alpha")
parser.add_argument('--auto', type=float, default=True, help="Kaiming-V or not")
parser.add_argument('--weight_decay', type=float, default=0, help="default is None")
parser.add_argument('--batch_size', type=int, default=128, help="batch size")
parser.add_argument("--num_epochs", type=int, default=20, help="number of epochs")
# noise ratio
parser.add_argument('--noise_ratio', type=float, default=0.0, help="noise ratio")

params = parser.parse_args()

if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(params.cuda))
else:
    print("cuda not work")
    device = 'cpu'
print(device)

timestamp = time.strftime("%y%m%d-%H%M%S", time.localtime())
log_dir_name = os.path.join('log', params.dataset)
param_dict = vars(params)
config_hash = hashlib.sha224(json.dumps(param_dict).encode()).hexdigest()
task_name = params.task_id + '-' + timestamp + config_hash[:4] + "-" + params.model

train_logger = Logger(task_name=task_name,
                      dir_name=log_dir_name,
                      heading=['epoch', 'train_loss', 'train_acc', 'test_loss', 'test_acc1', 'test_acc5'],
                      )
with open(os.path.join(log_dir_name, task_name + ".meta"), mode='wt') as f:
    json.dump(param_dict, f)

criterion = nn.CrossEntropyLoss()


def train_model(model, _iter, size):
    total_epoch_loss = 0
    total_epoch_correct = 0
    total_count = 0
    num_small_batch = 128 / params.batch_size
    model.train()
    for i, (X, Y) in tqdm(enumerate(_iter)):
        X, Y = F.interpolate(X.to(device), size=size), Y.to(device)
        if (i+1) % num_small_batch == 0:
            optim.step()
            optim.zero_grad()
        logits = model(X)
        preds = torch.max(logits, 1)[1].view(Y.size())
        loss = criterion(logits, Y)
        loss.backward()

        num_corrects = (preds == Y).float().sum()
        total_count += len(Y)
        total_epoch_loss += loss.item()
        total_epoch_correct += num_corrects.item()

    return total_epoch_loss / len(_iter), total_epoch_correct / total_count


def eval_model(model, _iter, size):
    total_epoch_loss = 0
    total_epoch_acc1 = 0
    total_epoch_acc5 = 0
    total_counter = 0
    model.eval()
    with torch.no_grad():
        for idx, (X, Y) in enumerate(_iter):
            X, Y = F.interpolate(X.to(device), size=size), Y.to(device)

            logits = model(X)
            preds1 = torch.max(logits, 1)[1].reshape(-1, 1)
            preds5 = logits.topk(5, 1, True, True)[1]
            loss = criterion(logits, Y)

            Y = Y.reshape(-1, 1)
            top1_num_corrects = (preds1 == Y).float().sum()
            top5_num_corrects = (preds5 == Y.expand_as(preds5)).float().sum()

            total_epoch_loss += loss.item()
            total_epoch_acc1 += top1_num_corrects.item()
            total_epoch_acc5 += top5_num_corrects.item()
            total_counter += len(Y)

    return (total_epoch_loss / len(_iter),
            total_epoch_acc1 / total_counter * 100,
            total_epoch_acc5 / total_counter * 100)


if __name__ == "__main__":
    train_iter, test_iter = get_dataset("TinyImageNet", rate=params.noise_ratio, batch_size=params.batch_size)

    model = get_model(params.model)
    if params.load_from:
        model.load_state_dict(torch.load(params.load_from))
    model.to(device)
    optim = Vadam2(model.parameters(), lr=params.lr * params.batch_size / 128, eps=1e-15,
                   v=params.v, alpha=params.alpha, auto_v=params.auto,
                   weight_decay=params.weight_decay)
    imgsize = EfficientNet.get_image_size(params.model)
    top1acc, top5acc = [], []
    for epoch in range(params.num_epochs):
        train_loss, train_acc = train_model(model, train_iter, size=imgsize)
        test_loss, test_acc1, test_acc5 = eval_model(model, test_iter, size=imgsize)
        top1acc.append(test_acc1)
        top5acc.append(test_acc5)
        train_logger.append(epoch + 1, train_loss, train_acc, test_loss, test_acc1, test_acc5)

    model.cpu()
    os.makedirs("saved_model", exist_ok=True)
    print("finished {}, top1 best {:.4f}, mean {:.4f} | top5 best {:.4f}, mean {:.4f}".format(
        task_name, np.max(top1acc), np.mean(top1acc[10:]), np.max(top5acc), np.mean(top5acc[10:])))
    torch.save(model.state_dict(), os.path.join("saved_model", task_name + ".pt"))
