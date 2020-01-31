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
# optimizer related
parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
parser.add_argument('--v', type=float, default=1, help="limitation of volumization")
parser.add_argument('--alpha', type=float, default=1.0, help="alpha")
parser.add_argument('--auto', type=float, default=True, help="Kaiming-V or not")
parser.add_argument('--weight_decay', type=float, default=0, help="default is None")
parser.add_argument('--batch_size', type=int, default=128, help="batch size")
parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs")
# noise ratio
parser.add_argument('--noise_ratio', type=float, default=0.0, help="noise ratio")

params = parser.parse_args()
model_for_data = {"MNIST": ["DNN"],
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
save_path = os.path.join('saved_model', params.model + params.dataset + params.task_id + ".pth")

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


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image


def test(model, test_loader, epsilon):

    # Accuracy counter
    correct = 0
    adv_examples = []
    count = 0
    # Loop over all examples in test set
    for datas, targets in tqdm(test_loader):
        count += len(targets)
        for i in range(len(targets)):
            # Send the data and label to the device
            data, target = datas[i:i+1].to(device), targets[i].to(device).reshape(1,)
            # Set requires_grad attribute of tensor. Important for Attack
            data.requires_grad = True

            # Forward pass the data through the model
            output = model(data)
            init_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            # If the initial prediction is wrong, dont bother attacking, just move on
            if init_pred.item() != target.item():
                continue

            # Calculate the loss
            loss = F.nll_loss(output, target)

            # Zero all existing gradients
            model.zero_grad()

            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = data.grad.data

            # Call FGSM Attack
            perturbed_data = fgsm_attack(data, epsilon, data_grad)

            # Re-classify the perturbed image
            output = model(perturbed_data)

            # Check for success
            final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            if final_pred.item() == target.item():
                correct += 1
                # Special case for saving 0 epsilon examples
                if (epsilon == 0) and (len(adv_examples) < 5):
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
            else:
                # Save some adv examples for visualization later
                if len(adv_examples) < 5:
                    adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                    adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )

    # Calculate final accuracy for this epsilon
    final_acc = correct/count
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, count, final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, adv_examples


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

    if os.path.exists(save_path):
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

    aar = {'epsilon': [], 'test_acc': []}
    for epsilon in [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        test_acc, _ = test(model, test_iter, epsilon)
        aar['epsilon'].append(epsilon)
        aar['test_acc'].append(test_acc)
    with open('AA{}.json'.format(params.dataset+params.model+params.task_id), mode='wt') as f:
        json.dump(aar, f)

