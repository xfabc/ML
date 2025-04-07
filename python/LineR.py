import numpy as np
import matplotlib.pyplot as plt

# 生成数据
np.random.seed(42)
X = np.linspace(0, 10, 100)
noise = np.random.normal(0, 2, 100)
y = 2 * X + 1 + noise
X = X.reshape(-1, 1)
X_b = np.c_[np.ones((100, 1)), X]


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        loss = 0
        for _ in range(self.n_iterations):
            y_pred = X.dot(self.weights)
            gradient = (1 / n_samples) * X.T.dot(y_pred - y)
            self.weights -= self.learning_rate * gradient
            # 计算损失函数
            loss += np.mean((y_pred - y) ** 2)
            if _ % 100 == 0:
                loss = loss/100.0
                print("Iteration:", _, "Loss:", loss)
                loss = 0


    def predict(self, X):
        return X.dot(self.weights)


# 训练模型
model = LinearRegression()
model.fit(X_b, y)
print("Learned weights:", model.weights)

# 可视化
plt.scatter(X[:, 0], y, label='Data')
plt.plot(X[:, 0], model.predict(X_b), color='red', label='Prediction')
plt.legend()
plt.show()