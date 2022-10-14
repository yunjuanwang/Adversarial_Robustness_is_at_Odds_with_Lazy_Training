'''
Projected adversarial training with different network width and weight deviation
'''

import numpy as np
import torch, torchvision
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from typing import *
from PIL import Image
import random
import copy
import pickle
import argparse
import sys
from torch.autograd import Variable, Function
from sklearn.model_selection import StratifiedKFold, train_test_split, ShuffleSplit

from art.estimators.classification import KerasClassifier, PyTorchClassifier
from art.defences.trainer import AdversarialTrainerMadryPGD,AdversarialTrainer
from art.data_generators import PyTorchDataGenerator
from art.utils import load_dataset, check_and_transform_label_format

import datetime, time
date_time = datetime.datetime.utcnow().isoformat().replace(":", "")
torch.backends.cudnn.benchmark=True

parser = argparse.ArgumentParser(description='Adv-Atk-NTK')
parser.add_argument('--T', default=1000, type=int, help='epoch')
parser.add_argument('--width', default=1000, type=int, help='width of network')
parser.add_argument('--lr', default=0.1, type=float, help='step size')
parser.add_argument('--bs', default=32, type=int, help='batch size')
parser.add_argument('--num1', default=0, type=int, help='mnist number1')
parser.add_argument('--num2', default=1, type=int, help='mnist number2')
parser.add_argument('--size', default=28, type=int, help='size of image')
parser.add_argument('--gap', default=0.1, type=float, help='weight deviation')
parser.add_argument('--atk_iters', default=100, type=int, help='attack iters')
#parser.add_argument('--alpha', default=0.05, type=float, help='alpha')
parser.add_argument('--eps', default=0.1, type=float, help='epsilon')
parser.add_argument('--seed', default=0, type=int, help='seed')
args = parser.parse_args()

seed = args.seed
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
(x_train, y_train_onehot), (x_test, y_test_onehot), min_, max_ = load_dataset(str("mnist"))
x_train = np.float32(np.reshape(x_train, (-1, 784)))
x_test = np.float32(np.reshape(x_test, (-1, 784)))
y_train_onehot = np.float32(y_train_onehot)
y_test_onehot = np.float32(y_test_onehot)
y_train = np.argmax(y_train_onehot, axis=1)
y_test = np.argmax(y_test_onehot, axis=1)

class net(nn.Module):
    def __init__(self, d=784, width=10000):
        super(net, self).__init__()
        self.fc1 = nn.Linear(d, width, bias = True)
        self.fc2 = nn.Linear(width, 1, bias = True)
        # set the top layer weight to be -1 or +1.
        self.fc1.weight.data.normal_(0.0, 1)
        self.fc1.bias.data.normal_(0.0, 1)
        self.fc2.weight.data = 1/math.sqrt(width)*torch.tensor(np.sign(np.random.normal(size = (1, width))).astype("float32"))
        self.fc2.bias.data = 1/math.sqrt(width)*torch.tensor(np.sign(np.random.normal(size = (1, 1))).astype("float32"))
    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # x = torch.sigmoid(self.fc2(x))
        return x
        
        
class Logistic_Loss(torch.nn.Module):
    def __init__(self):
        super(Logistic_Loss, self).__init__()

    def forward(self, inputs, target):
        L = torch.log(1 + torch.exp(-target*inputs.t()))
        return torch.mean(L)
        
        
def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None]
def pgd_l2(model, X, y, criterion, epsilon, alpha, num_iter):
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = criterion(model(X + delta), y)
        loss.backward()
        delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
        delta.data = torch.min(torch.max(delta.detach(), -X), 1-X) # clip X+delta to [0,1]
        delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
        delta.grad.zero_()
        
    return delta.detach()



num1 = args.num1
num2 = args.num2

X = np.concatenate((x_train[y_train==num1],x_train[y_train==num2]))
y = np.concatenate((np.array([-1]*len(x_train[y_train==num1])),np.array([1]*len(x_train[y_train==num2]))))

n = len(y)
idx = np.arange(n)
np.random.shuffle(idx)
X_train = X[idx]
Y_train = y[idx]
X_train_tensor = torch.from_numpy(X_train).float().to(device)
Y_train_tensor = torch.from_numpy(Y_train).long().to(device)

X_test = np.concatenate((x_test[y_test==num1], x_test[y_test==num2]))
Y_test = np.concatenate((np.array([-1]*len(x_test[y_test==num1])),np.array([1]*len(x_test[y_test==num2]))))

X_test_tensor = Variable(torch.from_numpy(X_test).float()).to(device)
Y_test_tensor = torch.from_numpy(Y_test).long().to(device)


width = args.width
lr = args.lr
T = args.T
bs = args.bs
size = args.size
gap = args.gap

eps = args.eps
alpha = args.eps*2.5/100
attack_iters = args.atk_iters

#name = 'mnist_ntk'+str(num1)+'vs'+str(num2)+'_size'+str(size)+'_T'+str(T)+'_bs'+str(bs)+'_lr'+str(lr)+'_eps'+str(eps)+'_alpha'+str(alpha)+'_atkiters'+str(attack_iters)+'_width'+str(width)+'C0_'+str(C0)+'_'+str(args.seed)
name = 'mnist_ntk'+str(num1)+'vs'+str(num2)+'_size'+str(size)+'_T'+str(T)+'_bs'+str(bs)+'_lr'+str(lr)+'_eps'+str(eps)+'_atkiters'+str(attack_iters)+'_width'+str(width)+'_gap'+str(gap)+'_'+str(args.seed)
log_filename = 'log_atk_gap/'+name+'.txt'
log = open(log_filename, 'w')
sys.stdout = log


Transform = transforms.Resize((size, size))
X_train_tensor = Transform(X_train_tensor.reshape(len(X_train_tensor),1,28,28)).reshape(-1, size*size)
X_test_tensor = Transform(X_test_tensor.reshape(len(X_test_tensor),1,28,28)).reshape(-1, size*size)
# normalize the data
X_train_tensor /= torch.norm(X_train_tensor, dim=1)[:,None]
X_test_tensor /= torch.norm(X_test_tensor, dim=1)[:,None]




model = net(d=size*size, width=width).to(device)
model1 =  copy.deepcopy(model)

for param in model.fc2.parameters():
    param.requires_grad = False

optimizer = optim.SGD(model.parameters(), lr = lr)
#optimizer = optim.SGD(model.parameters(), lr = lr, weight_decay=5e-4, momentum=0.9)
criterion = Logistic_Loss()

trainloss = []
testacc = []
diffweight = []
robusttestacc = []
robusttrainacc = []

c = 0
for t in range(T):
    l=[]
    robustacc = []
    delta = pgd_l2(model, X_train_tensor, Y_train_tensor, criterion, eps, alpha, attack_iters)
    pret_X_train_tensor = X_train_tensor + delta
    for i in range((n - 1) // bs + 1):
        begin = i * bs
        end = begin + bs
        X, y = pret_X_train_tensor[begin:end], Y_train_tensor[begin:end]
        
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        l.append(loss.item())
        robustacc.append(np.sum(((pred.sign()).squeeze(-1)==y).detach().cpu().numpy()) / len(y))
    
    accuracy = np.sum(((model(X_test_tensor).sign()).squeeze(-1)==Y_test_tensor).detach().cpu().numpy()) / len(Y_test_tensor)
    
    trained_weight = torch.cat((model.fc1.weight, model.fc1.bias[:,None]),dim=1)
    random_weight = torch.cat((model1.fc1.weight, model1.fc1.bias[:,None]),dim=1)
    weight_diff = np.linalg.norm((trained_weight - random_weight).detach().cpu().numpy(),axis=1)
    
    delta = pgd_l2(model, X_test_tensor, Y_test_tensor, criterion, eps, alpha, attack_iters)
    robust_accuracy = np.sum(((model(X_test_tensor+delta).sign()).squeeze(-1)==Y_test_tensor).detach().cpu().numpy()) / len(Y_test_tensor)
    
    trainloss.append(np.mean(l))
    testacc.append(accuracy)
    diffweight.append(weight_diff)
    robusttrainacc.append(np.mean(robustacc))
    robusttestacc.append(robust_accuracy)
    
    print('t:', t, 'loss:', trainloss[-1], 'robust train acc:', robusttrainacc[-1], 'clean test acc:', accuracy, 'robust test acc:', robust_accuracy ,'weight diff:', weight_diff.max())
    if weight_diff.max() > gap:
        s = np.where(weight_diff > gap)[0]
        trained_weight[s,:] = random_weight[s,:] + (trained_weight - random_weight)[s,:] / torch.norm((trained_weight - random_weight)[s,:],dim=1)[:,None] * gap
        model.fc1.weight.data = trained_weight[:,:size*size]
        model.fc1.bias.data = trained_weight[:,size*size]
    
    if t >=2:
        if robusttrainacc[-1]-robusttrainacc[-2]>1e-4:
            c = 0
        else:
            c += 1
        if c > 20:
            break

print("robust test acc:", robusttestacc[-1])

torch.save({
            'epoch':t,
            'model_state_dict':model.state_dict(),
            'trainloss':trainloss,
            'testacc':testacc,
            'diffweight':diffweight,
            'robusttestacc':robusttestacc,
           },'checkpoint/'+name+'_'+str(date_time)+'.pth')
    
f = open('log_atk_gap/'+name+'.pkl', 'wb')
pickle.dump((robusttestacc[-1],date_time),f)
f.close()


