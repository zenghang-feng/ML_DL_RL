# 1. 简介

本文采用 Pytorch 按照算法原理逐步实现 RNN 网络，对IMDB数据进行分类。


# 2. 程序实现

```
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchtext.datasets as datasets
import torchtext.data as data
from torchtext.vocab import GloVe

#################################################################
# 加载2数据集，注意torchtext版本，这里用的是 0.5.0
#################################################################
time_steps = 20
TEXT = data.Field(lower=True, batch_first=True, fix_length=time_steps)
LABEL = data.Field(sequential=False)
data_train, data_test = datasets.IMDB.splits(TEXT, LABEL)

max_size = 10000
vec_size = 300
# 以下建立词表的过程中会默认添加两个token： unk、pad，所以减去2 ==========
TEXT.build_vocab(data_train, vectors=GloVe(name='6B', dim=vec_size), max_size=max_size-2, min_freq=10)
LABEL.build_vocab(data_train)

batch_size = 128
train_iter, test_iter = data.BucketIterator.splits((data_train, data_test), batch_size=batch_size, shuffle=True)


#################################################################
# 初始化RNN计算单元的各个参数
#################################################################
input_size = max_size           # 等于独热编码大小，也是词表大小
hidde_size = 300                # 隐藏层向量维度自定义设置
output_size = 2                 # 输出层维度是分类维度，这里是二分类
# 隐藏层参数 ======================================================
w_ih = torch.rand((input_size, hidde_size))
w_hh = torch.rand((hidde_size, hidde_size))
b_h = torch.zeros(hidde_size)
# 输出层参数 ======================================================
w_ho = torch.rand((hidde_size, output_size))
b_o = torch.zeros((output_size))
# 配置梯度 ========================================================
parameters = [w_ih, w_hh, b_h, w_ho, b_o]
for p in parameters:
    p.requires_grad_(True)
# 初始化隐藏层的状态矩阵 ============================================
hide_state = torch.rand((batch_size, hidde_size))

#################################################################
# 定义网络训练的超参数，训练网络
#################################################################
# 定义迭代训练的次数，学习率 ========================================
epochs = 300
lr = 0.005

# 开始训练网络 ====================================================
print('---------------------开始训练网络-------------------------')
for epoch in range(epochs):
    loss_all = 0
    # train_sample = next(iter(train_iter))
    for train_sample in train_iter:
        x = train_sample.text                           # 是一个 batch_size x fix_length 的矩阵(128 x 20)，fix_length是句子长度，相当于时间步
        y = train_sample.label
        x_t = x.t()                                     # 转置为 时间步 x batch_size

        x_t_onehot = F.one_hot(x_t, num_classes=max_size).type(torch.float)
        if len(x_t_onehot[0]) != batch_size:            # 最后一个不是batch_size大小的样本不参与训练
            continue
        vec_out = []                                    # 存储每个时间步的输出，后续只取最后一个时间步的输出
        # 前向传播 --------------------------------------------
        for i in range(time_steps):
            x_t_onehot_ti = x_t_onehot[i]

            hide_state = torch.tanh(torch.matmul(x_t_onehot_ti, w_ih) + torch.matmul(hide_state.data, w_hh) + b_h)
            out = torch.matmul(hide_state, w_ho) + b_o
            vec_out.append(out)

        y_pre = vec_out[-1]
        y_pre_soft = F.softmax(y_pre, dim=1)            # 每个样本输出的是 1x2 的向量
        y_onehot = F.one_hot(y-1, 2)         # 同样转换成 1x1 的向量

        # 计算交叉熵损失 ---------------------------------------
        loss = (0 - torch.sum(torch.log(y_pre_soft) * y_onehot)) / batch_size
        loss_all += loss

        loss.backward()

        # 梯度更新 -------------------------------------------- RNN比较容易出现梯度消失或者爆炸的问题，后续可以进一步优化
        w_ih.data = w_ih.data - lr * w_ih.grad
        w_hh.data = w_hh.data - lr * w_hh.grad
        b_h.data = b_h.data - lr * b_h.grad
        w_ho.data = w_ho.data - lr * w_ho.grad
        b_o.data = b_o.data - lr * b_o.grad

        # 梯度清零 --------------------------------------------
        w_ih.grad.data.zero_()
        w_hh.grad.data.zero_()
        b_h.grad.data.zero_()
        w_ho.grad.data.zero_()
        b_o.grad.data.zero_()

    print("Epoch {}, loss: {}".format(epoch, loss_all.data / (len(train_iter) / batch_size)))



```
