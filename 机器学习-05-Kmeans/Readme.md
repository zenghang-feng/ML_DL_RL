
基于Python基础数据结构和Numpy计算向量距离的函数，编写程序实现Kmeans算法

# 1.算法原理


算法流程如下图所示：  

![i](https://github.com/zenghang-feng/ML_DL/blob/main/机器学习-05-Kmeans/pic0.png)

# 2. 程序实现

```
import numpy as np
import matplotlib.pyplot as plt


#################################################################
# 生成二维向量的集合，返回值是一个二维列表
#################################################################
def gen_data(x_range=(0, 1), y_range=(0, 1), sample_nums=10):
    res = []
    x_pos = np.random.uniform(low=x_range[0], high=x_range[1], size=sample_nums)
    y_pos = np.random.uniform(low=y_range[0], high=y_range[1], size=sample_nums)
    for i in range(sample_nums):
        res.append([x_pos[i], y_pos[i]])

    return res


#################################################################
# 生成3个簇的数据，编号并存放在一个字典中
#################################################################
cluster_1 = gen_data(x_range=(1, 3), y_range=(9, 11), sample_nums=30)
cluster_2 = gen_data(x_range=(2, 6), y_range=(5, 7), sample_nums=50)
cluster_3 = gen_data(x_range=(12, 15), y_range=(7, 9), sample_nums=60)
clusters = cluster_1 + cluster_2 + cluster_3
dit_data = {}
no = 1
for v in clusters:
    dit_data[no] = v
    no += 1


#################################################################
# 构建Kmeans 算法
#################################################################
# 指定聚类中心的个数，这是一个超参数，可以自己设置 =======================
cluster_no = 3
# 根据聚类中心个数，初始化中心点 ======================================
x_min = dit_data[1][0]
x_max = dit_data[1][0]
y_min = dit_data[1][1]
y_max = dit_data[1][1]
for k in dit_data:
    x_tmp = dit_data[k][0]
    y_tmp = dit_data[k][1]
    if x_tmp < x_min:
        x_min = x_tmp
    if x_tmp > x_max:
        x_max = x_tmp
    if y_tmp < y_min:
        y_min = y_tmp
    if y_tmp > y_max:
        y_max = y_tmp

dit_cc = {}                                     # 存储聚类中心点的坐标数据
for i in range(cluster_no):
    dit_cc[i+1] = [np.random.uniform(low=x_min, high=x_max), np.random.uniform(low=y_min, high=y_max)]

dit_cc_copy = dit_cc                            # 初始聚类中心备份，和最终聚类中心对比

# 初始化其他参数 ==============================================
loop_nums = 300                                 # 总计迭代次数用于判断是否终止迭代
loss_list = []                                  # 每次迭代前后损失函数下降的数值
dit_res = {}                                    # 存储没给点最终归于哪一个聚类中心

# 算法迭代 ===================================================
for l in range(loop_nums):
    # 给每个点分配距离最近的聚类中心
    loss_tmp = 0
    for k in dit_data:
        dis_min = np.linalg.norm(np.array([x_max, y_max])-np.array([x_min,y_min]))       # 当前点到聚类中心点距离最小值，初始化为数据集对角线点的距离
        point_k = dit_data[k]
        for c in dit_cc:
            point_c = dit_cc[c]
            dis_kc = np.linalg.norm(np.array(point_k) - np.array(point_c))
            if dis_kc < dis_min:
                dit_res[k] = c
                dis_min = dis_kc
        loss_tmp = loss_tmp + dis_min

    loss_avg = loss_tmp / len(dit_data)
    loss_list.append(loss_avg)

    # 更新聚类中心的位置坐标
    for c in dit_cc:
        x_tmp = 0
        y_tmp = 0
        num_count = 0
        for k in dit_res:
            if dit_res[k] == c:
                x_tmp = x_tmp + dit_data[k][0]
                y_tmp = y_tmp + dit_data[k][1]
                num_count += 1                          # 理论上不会是0
        dit_cc[c] = [x_tmp/num_count, y_tmp/num_count]

print('----------------------算法迭代完成--------------------------')

#################################################################
# 数据可视化聚类结果
#################################################################
l_data_x = []
l_data_y = []
for k in dit_data:
    l_data_x.append(dit_data[k][0])
    l_data_y.append(dit_data[k][1])

l_cluster_center_x_init = []
l_cluster_center_y_init = []
l_cluster_center_x = []
l_cluster_center_y = []
for c in dit_cc:
    l_cluster_center_x_init.append(dit_cc_copy[c][0])
    l_cluster_center_y_init.append(dit_cc_copy[c][0])
    l_cluster_center_x.append(dit_cc[c][0])
    l_cluster_center_y.append(dit_cc[c][1])

plt.scatter(x=l_data_x, y=l_data_y)
plt.scatter(x=l_cluster_center_x_init, y=l_cluster_center_y_init, c='black', s=80)
plt.scatter(x=l_cluster_center_x, y=l_cluster_center_y, c='red', s=100)

plt.show()
```

打印初始化聚类中心点、最终聚类效果的散点图：

![i](https://github.com/zenghang-feng/ML_DL/blob/main/机器学习-05-Kmeans/pic1.png)

其中黑色的点代表初始聚类中心，红色的点代表最终聚类中心，蓝色的点代表样本点；

