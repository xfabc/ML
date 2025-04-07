import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs


def rbf_kernel(X1, X2, gamma=0.1):
    pairwise_dists = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + \
                     np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * pairwise_dists)


class KernelSVM:
    def __init__(self, kernel='linear', C=1.0, gamma=0.1, learning_rate=0.001, epochs=1000):
        '''

        :param kernel:
        :param C:
        :param gamma:
        :param learning_rate:
        :param epochs:
        '''
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.lr = learning_rate
        self.epochs = epochs
        self.alpha = None
        self.support_vectors = None
        self.support_labels = None

    def _compute_kernel(self, X1, X2):
        if self.kernel == 'rbf':
            return rbf_kernel(X1, X2, self.gamma)
        else:
            return np.dot(X1, X2.T)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)

        for _ in range(self.epochs):
            P = self._compute_kernel(X, X)
            grad = y - np.dot(P, self.alpha * y)
            condition = (self.alpha > 0) & (self.alpha < self.C)
            self.alpha += self.lr * grad * y
            self.alpha = np.clip(self.alpha, 0, self.C)

        sv_indices = (self.alpha > 1e-5)
        self.support_vectors = X[sv_indices]
        self.support_labels = y[sv_indices]
        self.alpha_sv = self.alpha[sv_indices]

    def predict(self, X):
        K = self._compute_kernel(X, self.support_vectors)
        return np.sign(np.dot(K, self.alpha_sv * self.support_labels)).astype(int)


# 生成非线性数据
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)
y = np.where(y == 0, -1, 1)

# 训练RBF核SVM
svm_rbf = KernelSVM(kernel='rbf', gamma=0.99, C=1.0, learning_rate=0.001, epochs=1000)
svm_rbf.fit(X, y)


# 可视化函数（与之前相同）
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                         np.linspace(y_min, y_max, 50))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title("RBF Kernel SVM Decision Boundary")
    plt.show()


plot_decision_boundary(svm_rbf, X, y)