import torch
import random

############################################################
# 生成数据集：y = 2 * x1 + 5 * x2 + 9 * x3 + 10
############################################################
# 生成自变量（特征）x_i，因变量y（标签）==========================
feature_nums = 3
sample_nums = 1000
w_real = torch.tensor(data=[[2], [5], [9]], dtype=torch.float)              # 定义的是一个3维列向量
b_real = 10
data_x = torch.normal(mean=0, std=10, size=(sample_nums, feature_nums))     # 生成自变量
y_lab = torch.matmul(data_x, w_real) + b_real                               # 生成因变量

# 构建批量读取数据的生成器函数 =================================
def data_iter(data_x, y_lab, batch_size):
    """
    :param data_x: 特征矩阵
    :param y_lab: 标签向量
    :param batch_size: 批量大小
    :return:
    """
    sample_nums = data_x.shape[0]
    idx = list(range(sample_nums))
    random.shuffle(idx)                                                     # 打乱索引序列的编号顺序
    for i in range(0, sample_nums, batch_size):
        idx_batch = torch.tensor(idx[i : min(i+batch_size, sample_nums)])   # 生成大小为batch_size的索引编号（行号）序列
        yield data_x[idx_batch], y_lab[idx_batch]                           # 按照上述序列中行号提取出一个批量的数据


############################################################
# 构建线性网络，并进行训练
############################################################
# 参数初始化 =================================================
w = torch.normal(mean=0, std=0.01, size=(feature_nums, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 定义学习率，循环次数，批量大小 =================================
lr = 0.001
epochs = 500
batch_size = 128

# 构建线性网络模型：前向计算，反向传播求梯度优化参数 =================
for epoch in range(epochs):
    for x_iter, y_iter in data_iter(data_x=data_x, y_lab=y_lab, batch_size=batch_size):
        y_pre = torch.matmul(x_iter, w) + b
        y_dif = (y_pre - y_iter) ** 2 / 2
        loss_mean = y_dif.mean()
        loss_sum = y_dif.sum()
        loss_sum.backward()

        params = [w, b]
        for p in params:
            p.data = p.data - lr * p.grad / batch_size
            p.grad.data.zero_()

    print("epoch", epoch+1, "average loss is ", loss_mean.data)

print("------------------------训练完成-----------------------")
print(w, b)