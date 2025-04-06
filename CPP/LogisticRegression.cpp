#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

// Sigmoid函数
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// 计算预测值
double predict(const vector<double>& weights, const vector<double>& features) {
    double z = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
        z += weights[i] * features[i];
    }
    return sigmoid(z);
}

// 训练逻辑回归模型
void trainLogisticRegression(vector<vector<double>>& data, vector<int>& labels, vector<double>& weights, double learningRate, int epochs) {
    size_t numSamples = data.size();
    size_t numFeatures = weights.size();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double totalLoss = 0.0;

        for (size_t i = 0; i < numSamples; ++i) {
            // 计算预测值
            double prediction = predict(weights, data[i]);

            // 计算损失（交叉熵）
            totalLoss += -labels[i] * log(prediction) - (1 - labels[i]) * log(1 - prediction);

            // 更新权重
            for (size_t j = 0; j < numFeatures; ++j) {
                weights[j] -= learningRate * (prediction - labels[i]) * data[i][j];
            }
        }

        // 打印每轮的平均损失
        cout << "Epoch " << epoch + 1 << ", Average Loss: " << totalLoss / numSamples << endl;
    }
}

int main() {
    // 初始化随机数种子
    srand(static_cast<unsigned>(time(0)));

    // 示例数据集（特征和标签）
    vector<vector<double>> data = {
        {1.0, 2.0},   // 样本1
        {2.0, 3.0},   // 样本2
        {3.0, 3.0},   // 样本3
        {4.0, 5.0},   // 样本4
        {1.0, 1.0}    // 样本5
    };
    vector<int> labels = {0, 0, 1, 1, 0}; // 对应样本的标签

    // 添加偏置项（常数项）
    for (auto& sample : data) {
        sample.insert(sample.begin(), 1.0); // 在每个样本前插入1.0作为偏置
    }

    // 初始化权重（包括偏置项）
    size_t numFeatures = data[0].size();
    vector<double> weights(numFeatures, 0.0);

    // 设置超参数
    double learningRate = 0.1;
    int epochs = 1000;

    // 训练模型
    trainLogisticRegression(data, labels, weights, learningRate, epochs);

    // 输出最终的权重
    cout << "Trained Weights: ";
    for (double weight : weights) {
        cout << weight << " ";
    }
    cout << endl;

    // 测试模型
    vector<double> testSample = {1.0, 2.5, 3.0}; // 测试样本（包括偏置项）
    double prediction = predict(weights, testSample);
    cout << "Test Sample Prediction: " << prediction << endl;

    return 0;
}