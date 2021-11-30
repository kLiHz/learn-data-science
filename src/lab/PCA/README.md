# 实验一：PCA

## 一、实验目的

掌握 PCA 算法的原理

## 二、实验要求

通过本实验应达到如下要求：

- 理解数据降维过程
- 熟练使用 Python 或其他工具实现 PCA 算法

## 三、实验器材

- 计算机一台
- Python 或其他编程工具

## 四、实验内容

主成分分析（PCA）是一种提取数据集最重要特征的统计程序。PCA 的一个关键点是降维。降维是减少给定数据集的维数的过程。PCA 使我们能够找到数据变化最大的方向。

简而言之，使用 PCA 可以让我们找到数据集中的一些最重要的组成部分，即在描述某件事时一些最重要的变量。

<!--
Principal Components Analysis(PCA) is a statistical procedure that extracts the most important feature of a dataset. A key point of PCA is the Dimensionality Reduction. Dimensionality Reduction is the process of reducing the number of the dimensions of the given dataset. PCA allows us to find the direction along which our data varies the most. Briefly speaking, using PCA allows us to find some of the most important components in a dataset, i.e., some of the most important variables when describing a certain thing.
-->

### 1. PCA 降维过程

想象一下，我们用 N 个值描述一个事物，每个值称为一个“成分 (Component)”。这些值将形成一个 N 维向量。将 PCA 应用于 N 维数据集会产生 N 个 N 维特征向量 (Eigenvector)、N 个特征值 (Eigenvalue) 和一个 N 维中心点。这些特征向量即是数据集的“主成分 (Principal Component)”。每个特征向量的大小都编码在相应的特征值中，指示数据沿主成分变化的程度。

<!--
Imagine that we describe a thing with N values, with each one called a component. These values will form a N-dimensional vector. Applying PCA to a N-dimensional dataset yields N N-dimensional eigenvectors, N eigenvalues and a N-dimensional center point. Those eigenvectors are the principal components of the dataset. The size of each eigenvector is encoded in the corresponding eigenvalue, indicating how much the data vary along the principal component.
-->

我们的目标是要将一个给定的 *p* 维数据集 **X** 变换到一个具有较小维度的 *L* 维数据集 **Y**，相当于求 **X** 矩阵的 K-L 变换结果 **Y**。

<p>\[
\mathbf{Y} = \text{KLT} \{ \mathbf{Y} \}
\]</p>

#### 组织数据

假设我们的数据集中又若干冠词，每个观测都具有 *p* 个变量，而我们希望缩减数据集，使得每个观测可以仅使用 *L* 个变量表示 (*L* < *p*)。更进一步来假设，数据以 *n* 个数据向量 \\(x\_1 \\cdots x\_n\\) 来表示，每个 \\(x\_i\\) 代表一个具有 *p* 变量的观测。

- 将 \\(x\_1 \\cdots x\_n\\) 写作行向量（row vector），每个向量都有 *p* 列；
- 将这些行向量放入一个 \\(n \\times p\\) 的矩阵 **X** 中。

#### 计算经验均值

- 对每个维度 \\(j = 1, ..., p\\) 求经验均值（empirical mean）；
- 将这些均值放入一个 \\(p \\times 1 \\) 维的经验均值向量 **u** 中。

<p>\[
\mathbf{u}[j] = \frac 1 n \sum_{i=1}^{n}{\mathbf{X}[i,j]}
\]</p>

#### 计算与均值的偏差

从数据中减去均值是求解主成分基过程中的重要一步，它可以最小化估计数据时的均方差。因此，我们需要通过以下步骤将数据中心化：

- 从数据矩阵 **X** 的每一行中减去经验均值向量 **u**；
- 将得到的结果存储在 \\(n \\times p\\) 维的矩阵 **B** 中

用公式来表示就是：

<p>\[
\mathbf{B} = \mathbf{X} - \mathbf{h} \mathbf{B}^\intercal
\]</p>

其中，**h** 为一 \\(n \\times 1\\) 维的向量：

<p>\[
\mathbf{h}[i] = 1, i = 1, ..., n
\]</p>

#### 求协方差矩阵

通过矩阵 **B** 和其自身的外积运算，求出一个 \\(p \times p\\) 维的经验协方差矩阵 **C**。

<p>\[
\mathbf{C} = \frac{1}{n-1} \mathbf{B}^\intercal \mathbf{B}
\]</p>

其中，对矩阵 **B** 应该使用共轭转置。但当 **B** 中仅含有实数时，共轭转置等同于一般的转置。

#### 求协方差矩阵的特征向量和特征值

首先计算将矩阵 **C** 对角化的特征向量矩阵 **V**：

\\[ \\mathbf{V}^{-1}\\mathbf{C}\\mathbf{V} = \\mathbf{D} \\]

其中 **D** 是矩阵 **C** 特征值的对角矩阵。

**D** 是一个 \\(p \\times p\\) 的对角矩阵，形式如下：

<p>\[
\mathbf{D}[k,l] = 
\begin{cases}
  \lambda_k, & k = l \\
  0, & k \neq l \\
\end{cases}
\]</p>

其中，\\( \\lambda\_j \\) 表示协方差矩阵 **C** 的第 *j* 个特征值。

矩阵 **V** 也是 \\(p \\times p\\) 的形状，包含 *p* 个长度为 *p* 的列向量，即协方差矩阵 **C** 的 *p* 个特征向量。

特征值和特征向量一一对应，第 *j* 个特征值对应着第 *j* 个特征向量。


### 2. 实践

#### 鸢尾花数据集

[安德森鸢尾花卉数据集](https://zh.wikipedia.org/zh-cn/安德森鸢尾花卉数据集) ([*Iris* flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set)) 最初是埃德加·安德森从加拿大加斯帕半岛上的鸢尾属花朵中提取的形态学变异数据，后由罗纳德·费雪作为判别分析的一个例子，运用到统计学中。

数据集中有 150 个样本，包含鸢尾属下的三个亚属。每个样本有 4 个特征参数，分别为花萼长度，花萼宽度，花瓣长度，花瓣宽度。

Python 的 scikit-learn 库中的也附带了该数据集，在 [scikit-learn 网站的上有一篇关于使用该数据集的文档](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)。也可以使用 [archive.ics.uci.edu 上 CSV 格式的 *Iris* 数据集](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)。

可以使用 `pandas` 读取 CSV 格式的文件为 Pandas `DataFrame` 格式。

```python
import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
names = features + ['species']

df = pd.read_csv(url, names=names)
```

PCA 受比例影响，因此我们应该在应用 PCA 之前，对数据集中的数据进行缩放（标准化 / 去均值）。

```python
from sklearn import preprocessing

# Separating out the features
x = df.loc[:, features].values

# Standardizing the features
scaler = preprocessing.StandardScaler().fit(x)

x_scaled = scaler.transform(x)
```

之后，便可以对数据应用 PCA。下面直接使用 `sklearn` 中的 PCA 工具将数据降至 2 维。

```python
from sklearn import decomposition

pca = decomposition.PCA(n_components=2)
pca.fit(x)
x_decomposed = pca.tansform(x)

principalDf = pd.DataFrame(
    data = x_decomposed,
    columns = ['Principal Component 1', 'Principal Component 2']
)

finalDf = pd.concat([principalDf, df[['species']]], axis = 1)
```

[demo.py](./demo.py)

## 五、实验心得



## 参考资料

- [A STEP-BY-STEP EXPLANATION OF PRINCIPAL COMPONENT ANALYSIS - builtin.com](https://builtin.com/data-science/step-step-explanation-principal-component-analysis)
- [OpenCV: Introduction to Principal Component Analysis (PCA)](https://docs.opencv.org/3.4/d1/dee/tutorial_introduction_to_pca.html)
- [OpenCV: cv::PCA Class Reference](https://docs.opencv.org/master/d3/d8d/classcv_1_1PCA.html)
- [Dimension reduction with PCA in OpenCV](https://stackoverflow.com/questions/44186239/dimension-reduction-with-pca-in-opencv)
- <https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60>
- <https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html>
- <https://scikit-learn.org/stable/modules/preprocessing.html>


