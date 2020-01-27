import argparse
import os
import time
import json
import hashlib

from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F

from models import get_model
from ioutils import get_dataset, Logger
from vadam import Vadam2

parser = argparse.ArgumentParser(description='Train EfficientNet on TinyImageNet')

# task related
parser.add_argument('--dataset', type=str, default="TinyImageNet")
parser.add_argument('--model', type=str, default="EfficientNet")
parser.add_argument('--task_id', type=str, default='default')
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--load_from', type='str', default="")
# optimizer related
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--v', type=float, default=-1, help="limitation of volumization")
parser.add_argument('--alpha', type=float, default=1.0, help="alpha")
parser.add_argument('--auto', type=float, default=True, help="Kaiming-V or not")
parser.add_argument('--weight_decay', type=float, default=0, help="default is None")
parser.add_argument('--batch_size', type=int, default=128, help="batch size")
parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
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


def train_model(model, _iter):
    total_epoch_loss = 0
    total_epoch_acc = 0

    model.train()
    for X, Y in tqdm(_iter):
        X, Y = F.interpolate(X.to(device), size=240), Y.to(device)

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
    total_epoch_acc1 = 0
    total_epoch_acc5 = 0
    total_counter = 0
    model.eval()
    with torch.no_grad():
        for idx, (X, Y) in enumerate(_iter):
            X, Y = F.interpolate(X.to(device), size=240), Y.to(device)

            logits = model(X)
            preds1 = torch.max(logits, 1)[1].view(Y.size())
            preds5 = logits.topk(5, 1, True, True).view(Y.size(), 5)
            loss = criterion(logits, Y)

            top1_num_corrects = (preds1 == Y).float().sum()
            top5_num_corrects = (preds5 == Y.view(Y.size(), 1)).float().sum()

            total_epoch_loss += loss.item()
            total_epoch_acc1 += top1_num_corrects.item()
            total_epoch_acc5 += top5_num_corrects.item()
            total_counter += Y.size()

    return total_epoch_loss / total_counter, total_epoch_acc1 / total_counter, total_epoch_acc5 / total_counter


if __name__ == "__main__":
    embedding, train_iter, test_iter = get_dataset("TinyImageNet",
                                                   rate=params.noise_ratio,
                                                   batch_size=params.batch_size)

    model = get_model("EfficientNet")
    if params.load_from:
        model.load_state_dict(torch.load(os.path.join("saved_model", params.load_from)))
    model.to(device)
    optim = Vadam2(model.parameters(), lr=params.lr, eps=1e-15,
                   v=params.v, alpha=params.alpha, auto_v=params.auto,
                   weight_decay=params.weight_decay)
    for epoch in range(params.num_epochs):
        train_loss, train_acc = train_model(model, train_iter)
        test_loss, test_acc1, test_acc5 = eval_model(model, test_iter)
        train_logger.append(epoch + 1, train_loss, train_acc, test_loss, test_acc1, test_acc5)

    model.cpu()
    os.makedirs("saved_model", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("saved_model", task_name + ".pt"))
