import argparse
import os
import time
import json
import hashlib
import random
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from models import get_model
from ioutils import get_dataset, Logger
from vadam import Vadam2, Vadamw

parser = argparse.ArgumentParser(description='Adversarial Attack Evaluation')

# task related
parser.add_argument('--dataset', type=str, default="MNIST")
parser.add_argument('--model', type=str, default="DNN")
parser.add_argument('--task_id', type=str, default='default')
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--load', type=bool, default=False)
parser.add_argument('--optimizer', type=str, default='vadamw')
parser.add_argument('--case_name', type=str, default='')
# optimizer related
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--v', type=float, default=1, help="limitation of volumization")
parser.add_argument('--alpha', type=float, default=1.0, help="alpha")
parser.add_argument('--auto', type=float, default=True, help="Kaiming-V or not")
parser.add_argument('--weight_decay', type=float, default=0, help="default is None")
parser.add_argument('--batch_size', type=int, default=128, help="batch size")
parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--attack_rounds", type=int, default=20, help="pgd attack rounds")
parser.add_argument("--strength_unit", type=float, default=0.03, help="attack unit")
parser.add_argument("--strength_step", type=int, default=5, help="attack steps")
parser.add_argument("--attack_type", type=str, default="fgsm")
parser.add_argument("--adversarial_training", type=bool, default=False)
# noise ratio
parser.add_argument('--noise_ratio', type=float, default=0.0, help="noise ratio")

params = parser.parse_args()
model_for_data = {"MNIST": ["DNN", "CNN"],
                  "IMDB": ["LSTMATT", "LSTM"],
                  "CIFAR10": ["ResNet18"],
                  "CIFAR100": ["ResNet18"]}
assert params.model in model_for_data[params.dataset]

if torch.cuda.is_available():
    device = torch.device('cuda:{}'.format(params.cuda))
else:
    device = 'cpu'

timestamp = time.strftime("%y%m%d-%H%M%S", time.localtime())
log_dir_name = os.path.join('log', params.dataset)
param_dict = vars(params)
config_hash = hashlib.sha224(json.dumps(param_dict).encode()).hexdigest()
task_name = params.task_id + '-' + timestamp + config_hash[:4] + "-" + params.model + "case_name{}".format(params.case_name)
model_name = params.dataset + params.model + \
                "wd{}".format(params.weight_decay) + \
                "v{}".format(params.v) + \
                "alpha{}".format(params.alpha) +\
                "task_id{}".format(params.task_id) + \
                "case_name{}".format(params.case_name)

if params.case_name:
    case_name = params.case_name
else:
    case_name = model_name

transform_train = transforms.Compose([
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

save_path = os.path.join('saved_model', model_name)

train_logger = Logger(task_name=task_name,
                      dir_name=log_dir_name,
                      heading=['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'test_loss', 'test_acc'],
                      )
with open(os.path.join(log_dir_name, task_name + ".meta"), mode='wt') as f:
    json.dump(param_dict, f)

criterion = nn.CrossEntropyLoss()


def train_model(model, _iter):
    total_epoch_loss = 0
    total_epoch_acc = 0

    model.train()
    for X, Y in tqdm(_iter):
        X, Y = X.to(device), Y.to(device)
        if params.adversarial_training:
            X = adversarial_samples(model, X, Y)
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


def fgsm_attack(model, images, labels, eps):
    loss = nn.CrossEntropyLoss()

    images.requires_grad = True
    outputs = model(images)

    model.zero_grad()
    cost = loss(outputs, labels).to(device)
    cost.backward()

    adv_images = images + eps * images.grad.sign()
    if params.dataset == "MNIST":
        adv_images = torch.clamp(adv_images, 0, 1).detach_()
    return adv_images


def pgd_attack(model, images, labels, eps=0.3, alpha=2 / 255, iters=params.attack_rounds):
    loss = nn.CrossEntropyLoss()

    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        if params.dataset == "MNIST":
            images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        else:
            images = (ori_images + eta).detach_()

    return images


def adversarial_samples(model, data, target, epsilon=None):
    if epsilon is None:
        epsilon = random.randint(0, params.strength_step+1) * params.strength_unit
    if params.attack_type == "fgsm":
        perturbed_data = fgsm_attack(model, data, target, eps=epsilon)
    else:
        perturbed_data = pgd_attack(model, data, target, eps=epsilon)
    return perturbed_data


def test(model, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    adv_examples = []
    count = 0
    # Loop over all examples in test set
    for data, target in tqdm(test_loader):
        count += len(target)
        data, target = data.to(device), target.to(device)
        perturbed_data = adversarial_samples(model, data, target, epsilon)
        init_output = model(data)
        init_pred = init_output.max(1, keepdim=True)[1]
        perturbed_output = model(perturbed_data)
        perturbed_pred = perturbed_output.max(1, keepdim=True)[1]
        for i in range(len(target)):
            if target[i] == init_pred[i] and init_pred[i] == perturbed_pred[i]:
                correct += 1

    # Calculate final accuracy for this epsilon
    final_acc = correct/count
    print("Epsilon: {:.4f}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, count, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


def eval_model(model, _iter):
    total_epoch_loss = 0
    total_epoch_correct = 0
    total_count = 0
    model.eval()
    with torch.no_grad():
        for idx, (X, Y) in enumerate(_iter):
            X = X.to(device)
            Y = Y.to(device)

            logits = model(X)
            preds = torch.max(logits, 1)[1].view(Y.size())
            loss = criterion(logits, Y)

            total_epoch_loss += loss.item()
            total_epoch_correct += (preds == Y).float().sum().item()
            total_count += len(Y)

    return total_epoch_loss / len(_iter), total_epoch_correct / total_count


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

    if os.path.exists(save_path) and params.load:
        print("model loaded")
        model.load_state_dict(torch.load(save_path))
    model.to(device)
    if params.optimizer == "vadam":
        optim = Vadam2(model.parameters(), lr=params.lr, eps=1e-15,
                       v=params.v, alpha=params.alpha, auto_v=params.auto,
                       weight_decay=params.weight_decay)
    elif params.optimizer == "vadamw":
        optim = Vadamw(model.parameters(), lr=params.lr, eps=1e-15,
                       v=params.v, alpha=params.alpha, auto_v=params.auto,
                       weight_decay=params.weight_decay)
    else:
        print("")
        exit()

    if not params.load:
        print("save_path : {} not exist".format(save_path))
        for epoch in range(params.num_epochs):
            train_loss, train_acc = train_model(model, train_iter)
            val_loss, val_acc = eval_model(model, valid_iter)
            test_loss, test_acc = eval_model(model, test_iter)
            train_logger.append(epoch + 1, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)

        os.makedirs("saved_model", exist_ok=True)
        torch.save(model.state_dict(), save_path)

    test_loss, test_acc = eval_model(model, test_iter)
    print(test_acc)
    aar = {'epsilon': [], 'test_acc': []}
    for i in range(params.strength_step+1):
        epsilon = params.strength_unit * i
        test_acc, _ = test(model, test_iter, epsilon)
        aar['epsilon'].append(epsilon)
        aar['test_acc'].append(test_acc)
    os.makedirs("results/new_adversarial_attack", exist_ok=True)
    print(case_name)
    with open(os.path.join('results/new_adversarial_attack', '{}.json'.format(case_name)), mode='wt') as f:
        json.dump(aar, f)

