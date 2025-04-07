import numpy as np
import matplotlib.pyplot as plt

class LinearSVM:

    def __init__(self, learning_rate=0.01, n_iterations=1000, C = 1.0):
        '''

        :param learning_rate: 学习率
        :param n_iterations: 迭代次数
        :param C: 惩罚系数
        '''
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.C = C
        self.wights = None
        self.bias = None
    def _hinge_loss(self, y_true, y_pred):

        return np.maximum(0, 1 - y_true * y_pred)

    def _computer_gradient(self, X, y, y_pred):

        condition = 1- y * y_pred > 0
        dw = np.zeros(X.shape[1])
        db = 0
        for i in range(X.shape[1]):
            if condition[i]:
                dw += self.C * (y[i] * X[i])
        dw += 2 * self.wights
        db = np.sum(-self.C * y[condition])
        return dw, db
    def fit(self, X, y):

        self.wights = np.zeros(X.shape[1])

        for _ in range(self.n_iterations):
            #前向传播
            Linear_out = np.dot(X, self.wights) + self.bias
            #计算损失
            loss = self._hinge_loss(y, Linear_out)
            #计算梯度
            dw, db = self._computer_gradient(X, y, Linear_out)
            self.wights -= self.lr * dw
            self.bias -= self.lr * db
    def predict(self, X):

        Linear_out = np.dot(X, self.wights) + self.bias
        return np.sign(Linear_out)

if __name__ == '__main__':
    # 生成线性可分的二维数据
    from sklearn.datasets import make_blobs  # 仅用于生成测试数据
    X, y = make_blobs(n_samples=100, centers=2, random_state=42)
    y = np.where(y == 0, -1, 1)  # 将标签转换为-1和1

    # 可视化数据
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
    plt.title("Linearly Separable Data")
    plt.show()

    # 初始化并训练模型
    svm = LinearSVM(learning_rate=0.001, C=1.0, epochs=1000)
    svm.fit(X, y)

    # 预测
    y_pred = svm.predict(X)

    # 计算准确率
    accuracy = np.mean(y == y_pred)
    print(f"Accuracy: {accuracy:.2f}")


    # 可视化决策边界
    def plot_decision_boundary(model, X, y):
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                             np.linspace(y_min, y_max, 50))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
        plt.title("SVM Decision Boundary")
        plt.show()


    plot_decision_boundary(svm, X, y)



