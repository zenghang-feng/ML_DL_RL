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
time_steps = 50
TEXT = data.Field(lower=True, batch_first=True, fix_length=time_steps)
LABEL = data.Field(sequential=False)
data_train, data_test = datasets.IMDB.splits(TEXT, LABEL)

max_size = 10000
vec_size = 300
# 以下建立词表的过程中会默认添加两个token： unk、pad，所以减去2 ==========
TEXT.build_vocab(data_train, vectors=GloVe(name='6B', dim=vec_size), max_size=max_size-2, min_freq=10)
LABEL.build_vocab(data_train)

batch_size = 32
train_iter, test_iter = data.BucketIterator.splits((data_train, data_test), batch_size=batch_size, shuffle=True)

#################################################################
# 后续将文本输入给RNN之前，先转换成词向量
#################################################################
embed = nn.Embedding(max_size, vec_size)
embed.weight.data = TEXT.vocab.vectors


#################################################################
# 初始化RNN计算单元的各个参数
#################################################################
input_size = vec_size           # 等于独热编码大小，也是词表大小
hidde_size = 300                # 隐藏层向量维度自定义设置
output_size = 2                 # 输出层维度是分类维度，这里是二分类
# RNN隐藏层参数 ===================================================
w_ih = torch.randn((input_size, hidde_size))
w_hh = torch.randn((hidde_size, hidde_size))
b_h = torch.zeros(hidde_size)
# RNN输出层参数 ===================================================
w_ho = torch.randn((hidde_size, output_size))
b_o = torch.zeros(output_size)
# 线性层参数 ======================================================
w_l = torch.randn((time_steps*output_size, output_size))
b_l = torch.randn(output_size)
# 配置梯度 ========================================================
parameters = [w_ih, w_hh, b_h, w_ho, b_o, w_l, b_l]
for p in parameters:
    p.requires_grad_(True)
# 定义损失函数 ====================================================
loss_fun = nn.CrossEntropyLoss()

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

    for train_sample in train_iter:
        x = train_sample.text                           # 是一个 batch_size x fix_length 的矩阵(32 x 20)，fix_length是句子长度，相当于时间步
        y = train_sample.label
        x_t = x.t()                                     # 转置为 时间步 x batch_size

        x_t_vec = embed(x_t)                            # 维度是 时间步 x batch_size x vec_size(20x32x300)

        # 初始化隐藏层的状态矩阵 --------------------------------
        batch_size_real = len(x_t_vec[0])                           # 最后一个不足batch_size的小批次单独处理
        hide_state = torch.randn((batch_size_real, hidde_size))
        vec_out = []                                                # 存储每个时间步的输出，后续只取最后一个时间步的输出
        # 前向传播 --------------------------------------------
        for i in range(time_steps):
            x_t_vec_ti = x_t_vec[i]

            hide_state = torch.tanh(torch.matmul(x_t_vec_ti, w_ih) + torch.matmul(hide_state.data, w_hh) + b_h)
            out = torch.matmul(hide_state, w_ho) + b_o
            vec_out.append(out)

        rnn_out = torch.cat(vec_out, dim=0)
        rnn_out = rnn_out.reshape((time_steps, batch_size_real, output_size))
        rnn_out_p = rnn_out.permute(1, 0, 2)                        # 交换时间步、batch_size的维度顺序
        rnn_out_p = rnn_out_p.reshape((rnn_out_p.shape[0], -1))     # 时间步叠加成一个维度

        y_pre = torch.matmul(rnn_out_p, w_l) + b_l
        # y_pre_soft = F.softmax((F.tanh(y_pre)), dim=1)     # 每个样本输出的是 1x2 的向量；归一化后再进行softmax
        # y_onehot = F.one_hot(y-1, 2)                     # 同样转换成 1x1 的向量

        # 计算交叉熵损失 ---------------------------------------
        # loss = (0 - torch.sum(torch.log(y_pre_soft) * y_onehot)) / batch_size_real
        loss = loss_fun(y_pre, y - 1).mean()
        loss_all += loss

        loss.backward()

        # 梯度更新 -------------------------------------------- 可以通过参数列表迭代更新，简化写法
        w_ih.data = w_ih.data - lr * w_ih.grad
        w_hh.data = w_hh.data - lr * w_hh.grad
        b_h.data = b_h.data - lr * b_h.grad
        w_ho.data = w_ho.data - lr * w_ho.grad
        b_o.data = b_o.data - lr * b_o.grad
        w_l.data = w_l.data - lr * w_l.grad
        b_l.data = b_l.data - lr * b_l.grad

        # 梯度清零 --------------------------------------------
        w_ih.grad.data.zero_()
        w_hh.grad.data.zero_()
        b_h.grad.data.zero_()
        w_ho.grad.data.zero_()
        b_o.grad.data.zero_()
        w_l.grad.data.zero_()
        b_l.grad.data.zero_()

    print("epoch", epoch, "total loss is", loss_all)


