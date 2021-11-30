# %% [markdown]
# # PCA 降维逐步讲解
# 
# 首先加载 *Iris* 数据集。

# %%
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

print(X)

# %% [markdown]
# 接下来，需要对数据进行标准化（去均值）。
# 
# 首先求出每列数据的均值。

# %%
import numpy as np

mean = np.mean(X, axis=0)

print(mean)

# %% [markdown]
# 之后从每行数据（每次观测中）减去均值：

# %%
X_scaled = X - mean


print(X_scaled)

# %% [markdown]
# 求协方差矩阵以及特征值、特征向量：

# %%
cov = np.cov(X_scaled)

eigen_val, eigen_vec = np.linalg.eig(cov)

# %% [markdown]
# 特征值：

# %%
print(eigen_val)

# %% [markdown]
# 特征向量：

# %%
print(eigen_vec)

# %% [markdown]
# 将特征向量按特征值排序：

# %%
idx = eigen_val.argsort()[::-1]
eigen_val = eigen_val[idx]
eigen_vec = eigen_vec[:,idx]

# %% [markdown]
# 接下来选择前 \(n\) 个成分：

# %%
components_num = 2
principal_components = eigen_vec[:, 0:components_num]

print(principal_components)

# %% [markdown]
# 之后，可以将降维后的数据可视化。我们将 4 维的数据降至了 2 维，恰好可以绘制在 2 维平面上。

# %%
import pandas as pd

principal_df = pd.DataFrame(
    data=principal_components,
    columns=['PC1', 'PC2']
)

tag_df = pd.DataFrame(
    data=y,
    columns=['target']
)

final_df = pd.concat([principal_df, tag_df], axis=1)

print(final_df)


# %%
import matplotlib.pyplot as plt

plt.figure()
plt.title('2 component PCA')
plt.scatter(
    final_df['PC1'], 
    final_df['PC2'], 
)
plt.xlabel('Principal Component 1', fontsize = 15)
plt.ylabel('Principal Component 2', fontsize = 15)
plt.show()


