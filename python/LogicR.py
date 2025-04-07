import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:

    def __init__(self, learn_rate = 0.01, n_iterations = 1000):

        self.learn_rate = learn_rate
        self.n_iterations = n_iterations
        self.wigths = None
        self.bias = 0.0
    def fit(self, X, y):

        # 初始化权重和偏置
        n_samples, n_features = X.shape
        self.wigths = np.zeros(n_features)

        # 梯度下降
        for _ in range(self.n_iterations):
            Linear_model = np.dot(X, self.wigths) + self.bias
            y_pred = self.sigmoid(Linear_model)

            dw = (1/n_samples)*np.dot(X.T, (y_pred-y))
            db = (1/n_samples)*np.sum(y_pred-y)

            self.wigths = self.wigths * self.learn_rate * dw
            self.bias = self.bias *self.learn_rate *db

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

    def predict(self,X):
        Linear_model = np.dot(X,self.wigths)+self.bias
        y_pred = self.sigmoid(Linear_model)
        class_pred = [0 if y<=0.5 else 1 for y in y_pred]
        # return class_pred
        return np.array(class_pred)
def plot_decision_boundary(X, y, model):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.show()

# 生成模拟数据（二分类）
np.random.seed(42)
X = np.random.randn(20, 2)  # 20个样本，2个特征
y = (X.dot(np.array([2, -3])) + 5 + np.random.randn(20)) > 0  # 线性边界 + 噪声
y = y.astype(int)

# 训练模型
model = LogisticRegression(learn_rate=0.1, n_iterations=1000)
model.fit(X, y)

# 可视化
plot_decision_boundary(X, y, model)


