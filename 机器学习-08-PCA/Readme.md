# 1 基本描述：

本文采用numpy矩阵运算实现PCA算法，并和sklearn中的PCA算法结果进行对比。

# 2 程序实现

```
import numpy as np
from sklearn.decomposition import PCA

############################################################
# PCA算法计算过程
############################################################
# 1 构建输入矩阵 =============================================
n = 100                             # 样本数量
c = 5                               # 输入矩阵原始的特征维数
dim_red = 2                         # 降维后的特征维数
matrices = np.random.randint(low=-100, high=100, size=(n, c))

# 2 每一列特征减去对应的均值，对数据进行中心化 ====================
matrices_col_mean = np.average(a=matrices, axis=0)
matrices_center = matrices - matrices_col_mean

# 3 计算中心化后的矩阵的协方差矩阵 ==============================
matrices_cov = 1 / (n-1) * np.dot(matrices_center.T, matrices_center)

# 4 计算协方差矩阵的特征值和特征向量 ============================
eigen_value, eigen_vector = np.linalg.eig(matrices_cov)

# 5 取协方差矩阵的特征值前dim_red个最大的特征向量，构建投影矩阵 =====
idx_top = eigen_value.argsort()[-dim_red:][::-1]
eigen_vector_top = eigen_vector[:, idx_top]
matrices_v = eigen_vector_top / np.linalg.norm(x=eigen_vector_top, ord=None, axis=0)

# 6 将中心化之后的矩阵乘以投影矩阵得到降维后的数据 =================
matrices_res = np.dot(matrices_center, matrices_v)


############################################################
# 对比sklearn中的pca
############################################################
pca = PCA(n_components=2)
matrices_res_sk = pca.fit_transform(matrices)

print(matrices_res[:5, :])
print("------------------------分隔------------------------")
print(matrices_res_sk[:5, :])
# 关于向量正负号差异，可以参考如下博客 ---------------------------
# https://www.cnblogs.com/lpzblog/p/9519756.html
```
