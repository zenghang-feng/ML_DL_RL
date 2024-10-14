import numpy as np
import matplotlib.pyplot as plt


#################################################################
# 生成二维正态分布的数据，正常数据
#################################################################
sample_nums = 100
x1 = np.random.normal(loc=10, scale=2, size=sample_nums)  # 第一个维度的特征
x2 = np.random.normal(loc=8, scale=1, size=sample_nums)  # 第二个维度的特征


#################################################################
# 定义正态概率密度函数
#################################################################
def norm_distribute(x, mean, std):
    """
    :param x: 随机变量取值，是一个标量数值
    :return: 概率密度函数对应的取值，是一个标量数值
    """
    return (1 / (np.power(2 * np.pi, 0.5) * std)) * np.power(np.e, -(np.power(x - mean, 2)) / (2 * np.power(std, 2)))


#################################################################
# 定义概率模型和异常检测算法
#################################################################
# 分别计算两个维度数据的均值，方差，定义正态分布概率密度 ==================
mean_x1 = np.mean(x1)
std_x1 = np.std(x1)
mean_x2 = np.mean(x2)
std_x2 = np.std(x2)


# 根据概率独立性假设，定义概率模型 ====================================
def p_x(x_1, x_2):
    """
    :param x_1: 第一个维度特征的数值
    :param x_2: 第二个维度特征的数值
    :return: （x_1,x_2）出现的概率
    """
    return norm_distribute(x_1, mean_x1, std_x1) * norm_distribute(x_2, mean_x2, std_x2)


# 通过正样本中最小的p_x定义 ε =======================================
e = 1
for i in range(sample_nums):
    px = p_x(x1[i], x2[i])
    if px < e:
        e = px


# 定义异常检测算法 =================================================
def anomaly_detection(e, pos=(0, 0)):
    """
    :param e: 判断是否异常的最小概率值
    :param pos: 样本点的坐标
    :return: 是否是异常的标签和对应的概率，0代表正常，1代表异常
    """
    px = p_x(pos[0], pos[1])
    if px >= e:
        return 0, px
    else:
        return 1, px


# 给定一个新样本，计算数据产生的概率px，进行异常检测 ======================
x_new, y_new = 13, 15
res, px = anomaly_detection(e, pos=(x_new, y_new))
print(res, px)

#################################################################
# 数据可视化样本点
#################################################################
plt.scatter(x=x1, y=x2)
plt.scatter(x=[x_new], y=[y_new], c='red', s=100)

plt.show()
