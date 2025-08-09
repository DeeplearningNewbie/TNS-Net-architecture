# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import numpy as np
import sympy as sp
from sympy import *
import scipy
import scipy.io as sio
from scipy.io import savemat
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import math
from math import e
import matplotlib.pyplot as plt
import random

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# sigma_data = sio.loadmat('Data_sigmaC.mat')
p_data = sio.loadmat(r'Data_240702_RBvector.mat')
t_max = sio.loadmat(r'Data_240702_Tmax.mat')
# sigma_tensor = torch.tensor(sigma_data['Sigma'])
p0_tensor = torch.tensor(p_data['P_RB'])
t0_tensor = torch.tensor(t_max['T_max'])
p_tensor = p0_tensor[0:1000]
t_tensor = t0_tensor[0:1000]
Num = p_tensor.size()[0]
# index = torch.randperm(Num)
train_ratio = 0.7
Index = round(Num * train_ratio)
index = torch.randperm(Num)
train_data = TensorDataset(t_tensor[:Index], p_tensor[:Index])
test_data = TensorDataset(t_tensor[Index:], p_tensor[Index:])
Num_batch = Index
train_loader = DataLoader(dataset=train_data, batch_size=Num_batch, drop_last=True, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=Num - Num_batch, drop_last=True, shuffle=True)



# train_data = TensorDataset(t_tensor, p_tensor)
epsilon = torch.tensor([0.01], dtype=torch.float32)
alpha = torch.tensor([0.05], dtype=torch.float32)
factor = torch.tensor([0.9], dtype=torch.float32)
len_alpha = torch.tensor([0.1], dtype=torch.float32)
len_beta = torch.tensor([0.5], dtype=torch.float32)
Lamda = torch.linspace(1e3, 5e3, 50, dtype=torch.float32) # arriving rate
UE = 50  # UE number
mec = torch.tensor([10 ** 4], dtype=torch.float32)  # MEC computing capacity
eps = torch.tensor([10 ** (-8)], dtype=torch.float32)
# 网络层数
M = 10  # long scale
K = 10 # short scale
N = torch.tensor([100], dtype=torch.float32)  # RB number
# Num_batch = Num # batch-size
tau = torch.tensor([0.1 * 10 ** (-3)], dtype=torch.float32)  # time slot
threshold = torch.tensor([0.3], dtype=torch.float32)

# tau = np.array([0.5 * 10 ** (-3)])
con = torch.tensor([10 ** 6], dtype=torch.float32)  # constant
# mu = torch.Tensor([4.7583 * 10 ** 6, 4.7394 * 10 ** 6, 4.7013 * 10 ** 6])   # one RB rate

# g(sigma)函数
pr, Mu, V, lam, t = symbols('pr Mu V lam t', positive=True, real=True)
C = N * Mu * (pr) * con
# Q = (1 - (1 - (pr)) ** N) / tau
Q = N * (pr) / tau
bili = C * Q / (C + Q)
s1 = ((C + Q - lam) - ((C - Q) ** 2 + lam ** 2 + 2 * (C + Q) * lam) ** (1/2)) / 2
s2 = ((C + Q - lam) + ((C - Q) ** 2 + lam ** 2 + 2 * (C + Q) * lam) ** (1/2)) / 2
Fun = 1 + s1 * s2 / (s2 - s1) * ((1 / (V - s1) - 1 / (V - s2)) * e ** (-V * t) + V * e ** (-s2 * t) / (s2 * (V - s2)) - V * e ** (-s1 * t) / (s1 * (V - s1)))
diff_Fun = diff(Fun, pr)
diff2_Fun = diff(diff_Fun, pr)
# print(float(bili.evalf(subs={pr:0.5, Mu:1})))
# eq = sp.lambdify([pr, Mu], bili, 'math')
f = lambdify([pr, Mu, V, lam, t], Fun)
diff_f = lambdify([pr, Mu, V, lam, t], diff_Fun)
diff2_f = lambdify([pr, Mu, V, lam, t], diff2_Fun)
f1 = lambdify([pr, Mu, lam], s1)
f2 = lambdify([pr, Mu, lam], s2)
Bili = lambdify([pr, Mu], bili)

def function_g(sigma, p, mu, Num_batch):
    # print('function', p)
    c = N * p * mu * con
    q = (1 - (1 - p) ** N) / tau    #N * (p + eps) / tau
    numer = torch.pow(Lamda, 2) * torch.pow(q + c, 2)
    # lambda_c = c * q / (c + q)
    # print(lambda_c[i])
    item1 = c * mec * (1 - sigma) / UE + Lamda * (q + c)
    item2 = q * mec * (1 - sigma) / UE + Lamda * (q + c)
    # print('cq', c, q)
    # w = lambda_c / (torch.sum(lambda_c, axis=1).reshape(Num_batch,1))
    w = Lamda / torch.sum(Lamda)
    ###  sum求和应该是w*c*q
    Sum = torch.sum(w * numer / (item1 * item2), axis=1)
    Grad_sum = torch.sum(w * numer * c * mec / (UE * torch.pow(item1, 2) * item2) + w * numer * q * mec / (UE * torch.pow(item2, 2) * item1), axis=1)
    # print((Sum - sigma)/(Grad_sum - 1))
    return (Sum.reshape(Num_batch,1) - sigma) / (Grad_sum.reshape(Num_batch,1) - 1)
# 自定义神经网络模块
class my_Net(nn.Module):
    def __init__(self):
        super(my_Net, self).__init__()
        self.beta = nn.Parameter(torch.rand(K), requires_grad=True)
        # self.beta = nn.ParameterList([nn.Parameter(torch.rand(1, 1), requires_grad=True) for i in range(K)])
        self.alpha = nn.Parameter(torch.rand(M), requires_grad=True)
        # self.alpha = nn.ParameterList([nn.Parameter(torch.rand(1, 1), requires_grad= True) for i in range(M)])
        self.gamma = nn.Parameter(torch.rand(1), requires_grad=True)
        # self.gamma = nn.ParameterList([nn.Parameter(torch.rand(1, 1), requires_grad=True) for i in range(M)])
        self.mu = nn.Parameter(torch.rand(UE), requires_grad=True)
        # self.muu = nn.ParameterList([nn.Parameter(torch.rand(1, 1), requires_grad= True) for i in range(UE)])
        # self.new = nn.Parameter(torch.rand(M), requires_grad=True)
        # self.new = nn.ParameterList([nn.Parameter(torch.rand(1, 1), requires_grad=True) for i in range(M)])
        self.sigmoid = nn.Sigmoid()
        self.relu6 = nn.ReLU6()
        self.tanh = nn.Tanh()
        self.factor = nn.Parameter(torch.rand(1), requires_grad=True)

        # nn.init.uniform_(self.beta, 0.2, 0.4)
        # nn.init.uniform_(self.alpha, 0.001, 0.003)
        # nn.init.uniform_(self.gamma, 1.2, 1.5)
        # nn.init.uniform_(self.mu, 0.4, 0.6)

        nn.init.constant_(self.beta, 0.5)
        nn.init.constant_(self.gamma, 1.2)
        nn.init.constant_(self.alpha, 0.0005)
        nn.init.constant_(self.mu, 0.05)  # 0.1
    def forward(self, x, Num_batch):
        # p = torch.rand((Num_batch, UE), dtype=torch.float32) * 0.3
        scale = 0.005 / (x.max()-x.min())
        x = 0.005 + scale * (x - x.min())
        p = torch.ones((Num_batch, UE), dtype=torch.float32) * 0.2
        sigma = torch.ones((Num_batch, 1), dtype=torch.float32) * 0.5
        # Lamda.requires_grad = False
        False_mu = self.mu.detach()
        # Eq_p0 = sp.solve([bili.evalf(subs={Mu: False_mu[0]}) - Lamda[0], 0 < pr < 1], pr)
        p0 = Lamda * (False_mu * con * tau + 1) / (N * False_mu * con)
        for m in range(M):
            # sigma = torch.rand((Num_batch, 1), dtype=torch.float32) * 0.5
            step = self.factor # step size
            for k in range(K):
                # print(self.beta[k] * function_g(sigma, p, self.mu))
                sigma = sigma - self.beta[k] * function_g(sigma, p, self.mu, Num_batch)
                sigma = self.relu6(sigma*6.0)/6.0
                # print('sigma', sigma)
            # print(p, self.mu, mec * (1-sigma), Lamda, x)
            # self.mu = self.mu.detach().numpy()
            # trans = Bili(p, False_mu)
            p_old = p.detach()
            # print('p', p)
            # value_s1 = f1(p_old, False_mu, Lamda)
            # value_s2 = f2(p_old, False_mu, Lamda)
            # torch.set_printoptions(precision=16)
            z1 = f(p_old, False_mu, mec * (1-sigma), Lamda, x)
            label = z1 < 1 - epsilon
            # print((label==True).any().item())
            z2 = diff_f(p_old, False_mu, mec * (1-sigma), Lamda, x)
            # print('diff_z',z2)
            z3 = diff2_f(p_old, False_mu, mec * (1-sigma), Lamda, x)

            # p = p - alpha * delta_G / delta2_G
            delta_G = self.gamma ** (m) + z2 / (1 - z1 - epsilon)
            p_new = p - self.alpha[m] * delta_G
            # p_new = p - self.alpha[m] * delta_G
            # print(p_new)
            p_new = threshold - F.relu(threshold - p_new)
            p_new = F.relu(p_new - p0) + p0

            p = p_new


        # print('p', p)
        return p

def train_model():
    net = my_Net()
    paras = list(net.parameters())

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    train_loss_dict = []
    test_loss_dict = []
    for epoch in range(300):
        train_loss = 0.0
        for index, data in enumerate(train_loader, 0):
            loss = 0.0
            inputs, label = data
            inputs = inputs.to(torch.float32)
            # print('input', inputs)
            label = label.to(torch.float32)
            # print('input', inputs)
            # print('label', label)
            outputs = net(inputs, Num_batch)
            # print('output', outputs)

            # with torch.autograd.detect_anomaly():
            for w in net.parameters():
                for index_w in w:
                    loss = loss + F.relu(-1 * index_w) * 1
            loss = loss + criterion(outputs, label)
            train_loss = train_loss + loss
        # if train_loss < 0.01:
        train_loss_dict.append(train_loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # train_loss_dict.append(train_loss.item())
        # print('epoch:', epoch, 'train loss:', epoch_loss.item())
        # loss = np.array(loss_dict)
        # savemat('loss_TRBA.mat', {'loss_TRBA': loss})

        torch.save(net.state_dict(), 'model_TwoScale.pkl')
        test_model = my_Net()
        test_model.load_state_dict(torch.load('model_TwoScale.pkl'))
        test_loss = 0.0
        for index, data in enumerate(test_loader):
            test_inputs, test_labels = data
            test_inputs = test_inputs.to(torch.float32)
            # new_label = label.reshape(Num-Num_batch, 1, UE)
            test_labels = test_labels.to(torch.float32)
            test_outputs = test_model.forward(test_inputs, Num - Num_batch)
            for w in test_model.parameters():
                for index_w in w:
                    test_loss = test_loss + F.relu(-1 * index_w) * 1
            test_loss = test_loss + criterion(test_outputs, test_labels)
        # if test_loss < 0.01:
        # print('input', test_inputs)
        # print('label', test_labels)
        # print('output', test_outputs)
        test_loss_dict.append(test_loss.item())
        print('epoch:', epoch, 'train loss:', train_loss.item(), 'test loss', test_loss.item())
    for num, para in enumerate(paras):
        print('number:', num)
        print(para)
        print('________')
    return test_loss_dict
# mian function
if __name__ == '__main__':

    # train_data = TensorDataset(t_tensor[:Index], p_tensor[:Index])
    # test_data = TensorDataset(t_tensor[Index:], p_tensor[Index:])
    losses = []
    for i in range(1):
        print('time', i)
        losses.append(train_model())
    mean_loss = np.mean(losses, axis=0)

    loss = np.array(mean_loss)
    savemat('test_loss_TwoScale.mat', {'test_loss_TwoScale': loss})
    print('test finished')

    # plt.plot(train_loss_dict, label='train loss for every epoch')
    plt.plot(mean_loss, label='test loss for every epoch')
    plt.legend()
    plt.show()
    print('train finished')
    # loss = np.array(train_loss_dict)
    # # savemat('train_loss_TwoScale.mat', {'train_loss_TwoScale': loss})
    # print('train finished')
    # loss = np.array(test_loss_dict)
    # # savemat('test_loss_TwoScale.mat', {'test_loss_TwoScale': loss})
    # print('test finished')
    #
    # plt.plot(train_loss_dict, label='train loss for every batch')
    # plt.plot(test_loss_dict, label='test loss for every batch')
    # plt.legend()
    # plt.show()
    # 保存模型参数（字典形式）

    # 定义模型并载入模型参数
    # test_net = my_Net()
    # test_net.load_state_dict(torch.load('model.pkl'))
    # loss收敛图


