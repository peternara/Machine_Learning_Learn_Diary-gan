# coding: utf8
import tensorflow as tf
from random import choice, shuffle
from numpy import array
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets.samples_generator import make_blobs
from sklearn.datasets.samples_generator import make_circles


def ScatterPlot(X, Y, assignments=None, centers=None):
    if assignments is None:
        assignments = [0] * len(X)
    fig = plt.figure(figsize=(14, 8))
    cmap = ListedColormap(['red', 'green', 'blue', 'magenta'])
    plt.scatter(X, Y, c=assignments, cmap=cmap)
    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], c=range(len(centers)),
                    marker='+', s=400, cmap=cmap)
    plt.xlabel('Height (in)')
    plt.ylabel('Weight (lbs)')
    plt.show()


centers = [[-2, -2], [-2, 1.5], [1.5, -2], [2, 1.5]]  # 簇中心

# 生成人工数据集
# data, features = make_circles(n_samples=200, shuffle=True, noise=0.1, factor=0.4)
data, features = make_blobs(n_samples=200, centers=centers, n_features=2, cluster_std=0.8, shuffle=False,
                            random_state=42)

# 显示原图像
plt.scatter(data[:, 0], data[:, 1])
# plt.show()

kmeans = tf.contrib.learn.KMeansClustering(num_clusters=4, relative_tolerance=0.0001)

sess = tf.Session()


def input_fn():
    return tf.constant(data, tf.float32, np.shape(data)), None


kmeans.fit(input_fn=input_fn)
clusters = kmeans.clusters()
assignments = list(kmeans.predict_cluster_idx(input_fn=input_fn))
ScatterPlot(data[:, 0], data[:, 1], assignments, clusters)
