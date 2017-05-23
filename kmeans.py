# coding: utf8
import tensorflow as tf
from random import choice, shuffle
from numpy import array
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets.samples_generator import make_blobs

# from sklearn.datasets.samples_generator import make_circles

K = 4  # 类别数目
MAX_ITERS = 1000  # 最大迭代次数
N = 200  # 样本点数目

centers = [[-2, -2], [-2, 1.5], [1.5, -2], [2, 1.5]]  # 簇中心

# 生成人工数据集
# data, features = make_circles(n_samples=200, shuffle=True, noise=0.1, factor=0.4)
data, features = make_blobs(n_samples=N, centers=centers, n_features=2, cluster_std=0.8, shuffle=False, random_state=42)
plt.scatter(data[:, 0], data[:, 1])
plt.show()
