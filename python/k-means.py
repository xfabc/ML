import numpy as np
import matplotlib.pyplot as plt

# 定义KMeans类
class KMeans:
    def __init__(self, k=3, max_iters=100):
        """
        初始化KMeans类
        :param k: 聚类的数量
        :param max_iters: 最大迭代次数
        """
        self.k = k
        self.max_iters = max_iters
        self.centroids = None  # 存储最终的聚类中心

    def fit(self, X):
        """
        训练KMeans模型
        :param X: 输入数据 (n_samples, n_features)
        """
        # 随机初始化聚类中心
        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # 分配每个样本到最近的聚类中心
            labels = self._assign_clusters(X)

            # 更新聚类中心
            new_centroids = self._update_centroids(X, labels)

            # 检查是否收敛
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

    def predict(self, X):
        """
        根据训练好的模型预测每个样本所属的聚类
        :param X: 输入数据 (n_samples, n_features)
        :return: 每个样本所属的聚类标签
        """
        return self._assign_clusters(X)

    def _assign_clusters(self, X):
        """
        分配每个样本到最近的聚类中心
        :param X: 输入数据
        :return: 每个样本所属的聚类标签
        """
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)

    def _update_centroids(self, X, labels):
        """
        更新聚类中心为每个簇的均值
        :param X: 输入数据
        :param labels: 当前的聚类标签
        :return: 新的聚类中心
        """
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])
        return new_centroids


# 主程序
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(loc=[0, 0], scale=1, size=(100, 2)),
        np.random.normal(loc=[5, 5], scale=1, size=(100, 2)),
        np.random.normal(loc=[10, 10], scale=1, size=(100, 2))
    ])

    # 可视化原始数据
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], s=50, cmap='viridis')
    plt.title("Original Data")

    # 创建并训练KMeans模型
    kmeans = KMeans(k=3)
    kmeans.fit(X)

    # 获取聚类结果
    labels = kmeans.predict(X)

    # 可视化聚类结果
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', s=200, alpha=0.75, label='Centroids')
    plt.title("K-Means Clustering")
    plt.legend()
    plt.show()