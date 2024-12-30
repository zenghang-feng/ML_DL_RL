# 1 基本描述：

本文采用Pytorch实现AutoEncoder算法，对IRIS数据进行降维。

# 2 程序实现

```
import torch
import numpy as np
import pandas as pd
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

######################################################################
# 加载数据，转换成torch批量迭代加载的模式
######################################################################
x_data = load_iris().data
y_data = load_iris().target
# 随机打乱数据集 ==============================================
np.random.seed(116)  # 使用相同的seed，使输入特征/标签一一对应
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
# 用全部样本作为训练数据 ========================================
x_train = torch.tensor(x_data, dtype=torch.float)
y_train = torch.tensor(y_data, dtype=torch.int64)
# 将训练数据的特征和标签组合 =====================================
data_train = Data.TensorDataset(x_train, x_train)
# 随机读取小批量 上一篇手动实现了读取小批量，这里直接引用torch=========
batch_size = 30
train_iter = Data.DataLoader(data_train, batch_size, shuffle=True)


######################################################################
# 构建AutoEncoder网络模型，定义损失函数，定义优化器
######################################################################
# 定义网络结构 ================================================
class AutoEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, embed_size):
        super(AutoEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size_e = hidden_size
        self.embed_size = embed_size
        self.hidden_size_d = hidden_size
        self.output_size = input_size
        # 定义编码器 ---------------------------------------------
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, self.hidden_size_e),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.hidden_size_e, self.embed_size)
        )
        # 定义解码器 ---------------------------------------------
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.embed_size, self.hidden_size_d),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.hidden_size_d, self.output_size)
        )

    def forward(self, x):
        x_embed = self.encoder(x)
        y_pre = self.decoder(x_embed)

        return x_embed, y_pre


# 实例化网络模型 ================================================
in_size = x_train.shape[1]
hid_size = 3
emb_size = 2
auto_encoder = AutoEncoder(input_size=in_size, hidden_size=hid_size, embed_size=emb_size)

# 定义损失函数 ==================================================
loss_fun = torch.nn.MSELoss()

# 定义优化器 ====================================================
lr = 0.0001
opt = torch.optim.ASGD(params=auto_encoder.parameters(), lr=lr)


######################################################################
# 训练网络模型
######################################################################
epochs = 500
print('---------------------- 开始训练模型----------------------')
for e in range(epochs):
    loss_all = 0
    for x_, y_ in data_train:
        x_e, y_p = auto_encoder(x_)
        loss_tmp = loss_fun(y_p, y_)
        loss_all = loss_all + loss_tmp
        opt.zero_grad()
        loss_tmp.backward()
        opt.step()

    print("epoch", e, "/", epochs, "total loss is", loss_all)

# 保存模型 =====================================================
torch.save(auto_encoder.state_dict(), "auto_encoder.pth")
print('---------------------- 训练模型结束----------------------')


######################################################################
# 将降维后的数据和标签进行关联,并可视化
######################################################################
# 利用训练好的模型进行推理预测 ============================================
x_encode, y_pre_all = auto_encoder(torch.tensor(x_data, dtype=torch.float))
# 对特征取值区间进行放大 =================================================
x_encode = x_encode * 10
# 数据处理，然后绘制二维散点图 ============================================
df_iris_dr = pd.merge(left=pd.DataFrame(x_encode.detach().numpy()), right=pd.DataFrame(y_data),
                      how="inner", left_index=True, right_index=True)
df_iris_dr.columns = ["f1", "f2", "target"]

fig, ax = plt.subplots()
df_iris_dr[df_iris_dr["target"] == 0].plot.scatter(x="f1", y="f2", color="g", label=0, ax=ax)
df_iris_dr[df_iris_dr["target"] == 1].plot.scatter(x="f1", y="f2", color="r", label=1, ax=ax)
df_iris_dr[df_iris_dr["target"] == 2].plot.scatter(x="f1", y="f2", color="b", label=2, ax=ax)
plt.show()

```

降到2个维度之后，绘制不同类别数据的散点图如下：
![i](https://github.com/zenghang-feng/ML_DL_RL/blob/main/机器学习-08-降维算法-AutoEncoder/Softmax_ASGD.png)
