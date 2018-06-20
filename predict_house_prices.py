# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import time

train = pd.read_csv('./dataset/train.csv')
test = pd.read_csv('./dataset/test.csv')

print('一共有 {} 个训练集样本'.format(train.shape[0]))
print('一共有 {} 个测试集样本'.format(test.shape[0]))

all_features = pd.concat((train.loc[:, 'MSSubClass':'SaleCondition'],
                          test.loc[:, 'MSSubClass':'SaleCondition']))

numeric_feats = all_features.dtypes[all_features.dtypes != "object"].index # 取出所有的数值特征
all_features[numeric_feats] = all_features[numeric_feats].apply(lambda x: (x - x.mean()) 
                                                                / (x.std()))
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features = all_features.fillna(all_features.mean())

num_train = train.shape[0]

train_features = all_features[:num_train].values.astype(np.float32)
test_features = all_features[num_train:].values.astype(np.float32)

train_labels = train.SalePrice.values[:, None].astype(np.float32)
test_labels = test.SalePrice.values[:, None].astype(np.float32)


from torch import nn

def get_model():
    # todo: 使用 nn.Sequential 来构造多层神经网络，注意第一层的输入
    model = nn.Sequential(
        nn.Linear(331, 100), 
        nn.Tanh(),
        nn.Linear(100, 30), 
        nn.Tanh(),
        nn.Linear(30, 10), 
        nn.Tanh(),
        nn.Linear(10, 1)
    )
    return model

# 可以调整的超参
batch_size = 32
epochs = 400
use_gpu = True
lr = 1
weight_decay = 10

if use_gpu:
    criterion = nn.MSELoss().cuda() # todo: 使用 mse 作为 loss 函数
else:
    criterion = nn.MSELoss() # todo: 使用 mse 作为 loss 函数

from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from utils import get_rmse_log

# todo: 将所有的 feature 和 label 都转成 torch 的 Tensor
train_features = torch.from_numpy(train_features).float()
train_labels = torch.from_numpy(train_labels).float()
test_features = torch.from_numpy(test_features).float()
test_labels = torch.from_numpy(test_labels).float()

# 构建一个数据的迭代器
def get_data(x, y, batch_size, shuffle):
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size, shuffle=shuffle)
    # return DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=2)

def train_model(model, x_train, y_train, x_valid, y_valid, epochs, lr, weight_decay):
    metric_log = dict()
    metric_log['train_loss'] = list()
    if x_valid is not None:
        metric_log['valid_loss'] = list()
    
    train_data = get_data(x_train, y_train, batch_size, True)
    if x_valid is not None:
        valid_data = get_data(x_valid, y_valid, batch_size, False)
    else:
        valid_data = None
    
    # todo: 构建优化器，推荐使用 Adam，也可以尝试一下别的优化器
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    # optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=1, alpha=0.9)
    # optimizer = torch.optim.Adadelta(model.parameters(), rho=0.99)
    optimizer = torch.optim.Adam(model.parameters(), lr=1)

    if use_gpu:
        model = model.cuda()
    
    start = time.time()
    for e in range(epochs):
        # 训练模型
        running_loss = 0
        model.train()
        for x, y in train_data:
            # x, y = data
            if use_gpu:
                x = x.cuda()
                y = y.cuda()
            x = Variable(x)
            y = Variable(y)
            
            # todo: 前向传播
            out = model(x)
            
            # todo: 计算 loss
            loss = criterion(out, y)
            
            # todo: 反向传播，更新参数
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss
        
        metric_log['train_loss'].append(get_rmse_log(model, x_train, y_train, use_gpu))
        
        # 测试模型
        if x_valid is not None:
            metric_log['valid_loss'].append(get_rmse_log(model, x_valid, y_valid, use_gpu))
            print_str = 'epoch: {}, train loss: {:.3f}, valid loss: {:.3f}'\
            .format(e+1, metric_log['train_loss'][-1], metric_log['valid_loss'][-1])
        else:
            print_str = 'epoch: {}, train loss: {:.3f}'.format(e+1, metric_log['train_loss'][-1])
        if (e + 1) % 10 == 0:
            print(print_str)
            # print()

    print("Time used: {} seconds.".format(time.time() - start))
    # =======不要修改这里的内容========
    # 可视化
    figsize = (10, 5)
    plt.figure(figsize=figsize)
    plt.plot(metric_log['train_loss'], color='red', label='train')
    if valid_data is not None:
        plt.plot(metric_log['valid_loss'], color='blue', label='valid')
    plt.legend(loc='best')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()
    
model = get_model()
train_model(model, train_features, train_labels, test_features, test_labels, epochs, lr, weight_decay)
