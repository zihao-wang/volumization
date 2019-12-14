import os
import time
import load_data
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
from models.LSTM import LSTMClassifier

import math
import torch
import argparse
from myadam import Adam


parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--eps', type=float, default=1.9)
parser.add_argument('--r', type=float, default=0.1)
parser.add_argument('--v', type=float, default=0.1)
args = parser.parse_args()
eps_a = args.eps

learning_rate = 2e-3
batch_size = 32
output_size = 2
hidden_size = 256
embedding_length = 300



def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    if torch.cuda.is_available():
        model.cuda()
    #optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-3, eps=1e-15)
    
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch.text[0]
        target = batch.label
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not batch_size):# One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction = model(text)
        #print(prediction.shape)
        
        #if epoch < 5: 
        #    eps = 10
        #else:
        #    eps = eps_a
        #output = F.softmax(prediction, dim=1)
        #output = output + (output[:, 2] / eps).unsqueeze(1)
        #output.log_()
        output = F.log_softmax(prediction, dim=1)
        loss = F.nll_loss(output, target)


        num_corrects = (torch.max(prediction[:, :2], 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        loss.backward()
        #clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        
        if steps % 100 == 0:
            print ('Epoch:', epoch+1, 'Idx:', idx+1, 'Training Loss:', loss.item(), 'Training Accuracy:', acc.item())
        
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

            #eps = 9.9
            #output = F.softmax(prediction, dim=1)
            #output = output + (prediction[:, 2] / eps).unsqueeze(1)
            #output.l()
            loss = F.nll_loss(prediction, target)

            num_corrects = (torch.max(prediction[:, :2], 1)[1].view(target.size()).data == target.data).sum()
            acc = 100.0 * num_corrects/len(batch)
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    return total_epoch_loss/len(val_iter), total_epoch_acc/len(val_iter)


if __name__ == "__main__":

    TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset(rate=args.r,
                                                                                                  batch_size=batch_size)
    model = LSTMClassifier(batch_size, output_size+1, hidden_size, vocab_size, embedding_length, word_embeddings)
    optim = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, eps=1e-15, v=args.v)
    #optim = LaProp(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    loss_fn = F.cross_entropy

    def loss_theo(o, e):
       # return max(math.log(1/o), e * math.log(e + (e/o)) + (1 - e)
       #            * math.log ((1 - e)/o))

        x = e * math.log(e) + (1 - e) * math.log((1 - e)/ (o - 1))
        print(-x)
        return x

    print(loss_theo(eps_a, args.r))
    for epoch in range(20):
        train_loss, train_acc = train_model(model, train_iter, epoch)
        val_loss, val_acc = eval_model(model, valid_iter)


        print(loss_theo(eps_a, args.r))
        print('Epoch:', epoch+1, 'Train Loss:', train_loss, 'Train Acc:',
              train_acc, 'Val. Loss:', val_loss, 'Val. Acc:', val_acc)

    test_loss, test_acc = eval_model(model, test_iter)
    print('Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%')




assert False

''' Let us now predict the sentiment on a single sentence just for the testing purpose. '''
test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."
test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."

test_sen1 = TEXT.preprocess(test_sen1)
test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]

test_sen2 = TEXT.preprocess(test_sen2)
test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]

test_sen = np.asarray(test_sen1)
test_sen = torch.LongTensor(test_sen)
test_tensor = Variable(test_sen, volatile=True)
test_tensor = test_tensor.cuda()
model.eval()
output = model(test_tensor, 1)
out = F.softmax(output, 1)
if (torch.argmax(out[0]) == 1):
    print ("Sentiment: Positive")
else:
    print ("Sentiment: Negative")
