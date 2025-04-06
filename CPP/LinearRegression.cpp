#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

// 计算预测值
double predict(const vector<double>& weights, const vector<double>& features) {
    double prediction = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
        prediction += weights[i] * features[i];
    }
    return prediction;
}

// 计算均方误差（MSE）
double computeMSE(const vector<vector<double>>& data, const vector<double>& labels, const vector<double>& weights) {
    double mse = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        double prediction = predict(weights, data[i]);
        mse += pow(prediction - labels[i], 2);
    }
    return mse / data.size();
}

// 训练线性回归模型
void trainLinearRegression(vector<vector<double>>& data, vector<double>& labels, vector<double>& weights, double learningRate, int epochs) {
    size_t numSamples = data.size();
    size_t numFeatures = weights.size();

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // 存储每个样本的梯度
        vector<double> gradients(numFeatures, 0.0);

        for (size_t i = 0; i < numSamples; ++i) {
            double prediction = predict(weights, data[i]);
            double error = prediction - labels[i];

            // 更新梯度
            for (size_t j = 0; j < numFeatures; ++j) {
                gradients[j] += error * data[i][j];
            }
        }

        // 更新权重
        for (size_t j = 0; j < numFeatures; ++j) {
            weights[j] -= learningRate * (gradients[j] / numSamples);
        }

        // 打印每轮的损失
        double mse = computeMSE(data, labels, weights);
        cout << "Epoch " << epoch + 1 << ", MSE: " << mse << endl;
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
    vector<double> labels = {2.0, 3.0, 4.0, 5.0, 1.0}; // 对应样本的目标值

    // 添加偏置项（常数项）
    for (auto& sample : data) {
        sample.insert(sample.begin(), 1.0); // 在每个样本前插入1.0作为偏置
    }

    // 初始化权重（包括偏置项）
    size_t numFeatures = data[0].size();
    vector<double> weights(numFeatures, 0.0);

    // 设置超参数
    double learningRate = 0.01;
    int epochs = 1000;

    // 训练模型
    trainLinearRegression(data, labels, weights, learningRate, epochs);

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