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

parser = argparse.ArgumentParser(description='Adversarial Attack Evaluation')

# task related
parser.add_argument('--dataset', type=str, default="MNIST")
parser.add_argument('--model', type=str, default="DNN")
parser.add_argument('--task_id', type=str, default='default')
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--load', type=bool, default=False)
# optimizer related
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--v', type=float, default=1, help="limitation of volumization")
parser.add_argument('--alpha', type=float, default=1.0, help="alpha")
parser.add_argument('--auto', type=float, default=True, help="Kaiming-V or not")
parser.add_argument('--weight_decay', type=float, default=0, help="default is None")
parser.add_argument('--batch_size', type=int, default=128, help="batch size")
parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs")
parser.add_argument("--attack_rounds", type=int, default=10, help="pgd attack rounds")
parser.add_argument("--strength_unit", type=float, default=0.05, help="attack unit")
parser.add_argument("--strength_step", type=int, default=5, help="attack steps")
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
task_name = params.task_id + '-' + timestamp + config_hash[:4] + "-" + params.model
case_name = params.dataset + params.model + \
            "wd{}".format(params.weight_decay) + \
            "v{}".format(params.v) + \
            "alpha{}".format(params.alpha) +\
            "task_id{}".format(params.task_id)

save_path = os.path.join('saved_model', case_name)

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


def fgsm_attack(model, image, label, epsilon):
    output = model(image)
    # Calculate the loss
    loss = F.nll_loss(output, label)

    # Zero all existing gradients
    model.zero_grad()

    # Calculate gradients of model in backward pass
    loss.backward()

    # Collect datagrad
    data_grad = image.grad.data

    # Call FGSM Attack
    perturbed_data = fgsm_attack(image, epsilon, data_grad)

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def pgd_attack(model, images, labels, eps=0.3, alpha=2 / 255, iters=params.attack_rounds):
    images = images.to(device)
    labels = labels.to(device)
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


def test(model, test_loader, epsilon):
    # Accuracy counter
    correct = 0
    adv_examples = []
    count = 0
    # Loop over all examples in test set
    for data, target in tqdm(test_loader):
        count += len(target)
        data, target = data.to(device), target.to(device)
        perturbed_data = pgd_attack(model, data, target, eps=epsilon)
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

    if os.path.exists(save_path) and params.load_model:
        print("model loaded")
        model.load_state_dict(torch.load(save_path))
    model.to(device)
    optim = Vadam2(model.parameters(), lr=params.lr, eps=1e-15,
                   v=params.v, alpha=params.alpha, auto_v=params.auto,
                   weight_decay=params.weight_decay)
    if not os.path.exists(save_path):
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
    for i in range(params.strength_step):
        epsilon = params.strength_unit * (i+1)
        test_acc, _ = test(model, test_iter, epsilon)
        aar['epsilon'].append(epsilon)
        aar['test_acc'].append(test_acc)
    os.makedirs("adversarial_attack", exist_ok=True)
    print(case_name)
    with open(os.path.join('adversarial_attack', '{}.json'.format(case_name)), mode='wt') as f:
        json.dump(aar, f)

