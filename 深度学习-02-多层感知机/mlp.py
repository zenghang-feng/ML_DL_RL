import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.nn import functional as F
from sklearn import datasets

############################################################
# 读取数据，处理数据
############################################################
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

# 随机打乱数据集 ==============================================
np.random.seed(116) # 使用相同的seed，使输入特征/标签一一对应
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)

# 划分训练集和测试集 ===========================================
x_train = torch.tensor(x_data[:-30], dtype=torch.float)
y_train = torch.tensor(y_data[:-30], dtype=torch.int64)
x_test = torch.tensor(x_data[-30:], dtype=torch.float)
y_test = torch.tensor(y_data[-30:], dtype=torch.int64)

# 将训练数据的特征和标签组合 =====================================
data_train = Data.TensorDataset(x_train, y_train)
data_test = Data.TensorDataset(x_test, y_test)

# 随机读取小批量 上一篇手动实现了读取小批量，这里直接引用torch=========
batch_size = 30
train_iter = Data.DataLoader(data_train, batch_size, shuffle=True)
test_iter = Data.DataLoader(data_test, batch_size)

# 参数初始化 ==================================================
w1 = torch.tensor(np.random.normal(0, 0.01, (4, 3)), dtype=torch.float32)
b1 = torch.zeros(3, dtype=torch.float32)
w2 = torch.tensor(np.random.normal(0, 0.01, (3, 3)), dtype=torch.float32)
b2 = torch.zeros(3, dtype=torch.float32)
params = [w1, b1, w2, b2]
for param in params:
    param.requires_grad_(requires_grad=True)

##############################################################
# 构建模型，训练模型
##############################################################
# 定义超参数 ===================================================
loss_all = 0
lr = 0.01
epoch = 1000
total_correct = 0

# 模型训练 =======================================================
for epoch in range(epoch):
    loss_all = 0                                                # 每一个epoch总loss

    for x_train, y in train_iter:
        # 计算第1层
        y_1 = torch.matmul(x_train, w1) + b1
        y_1 = F.relu(y_1)

        # 计算第2层
        y_2 = torch.matmul(y_1, w2) + b2
        y_2 = F.softmax(y_2, dim=1)                             # 每个样本输出的是 1x3 的向量

        # 通过独热编码将标签转化为与预测值同样的格式
        y = torch.nn.functional.one_hot(y, 3)        # 同样转换成 1x3 的向量

        # 计算交叉熵损失
        loss = (0 - torch.sum(torch.log(y_2) * y)) / batch_size
        loss_all += loss

        # 这里是对loss求导
        loss.backward()  # 小批量的损失对模型参数求梯度

        # 更新各个参数
        w1.data = w1.data - lr * w1.grad
        b1.data = b1.data - lr * b1.grad
        w2.data = w2.data - lr * w2.grad
        b2.data = b2.data - lr * b2.grad

        # 梯度清零
        w1.grad.data.zero_()
        b1.grad.data.zero_()
        w2.grad.data.zero_()
        b2.grad.data.zero_()

    print("Epoch {}, loss: {}".format(epoch, loss_all.data / 4))

################################################################
# 模型预测
################################################################
for x_test, y_test in test_iter:
    y_p1 = torch.matmul(x_test, w1) + b1
    y_p1 = F.relu(y_p1)
    y_p2 = torch.matmul(y_p1, w2) + b2
    y_pre = F.softmax(y_p2, dim=1)

    pred = torch.argmax(y_pre, dim=1)
    correct = torch.as_tensor(torch.eq(pred, y_test), dtype=torch.int64)
    correct = torch.sum(correct)
    total_correct += correct

acc = total_correct / y_test.shape[0]
print('test_acc', acc)
