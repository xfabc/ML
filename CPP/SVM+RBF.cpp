#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

// 定义超参数
struct SVMParams {
    double learningRate = 0.01; // 学习率
    int epochs = 1000;          // 迭代次数
    double lambda = 0.01;       // 正则化参数
    double gamma = 0.5;         // RBF 核的参数
};

// 计算 RBF 核
double rbfKernel(const vector<double>& x1, const vector<double>& x2, double gamma) {
    double distance = 0.0;
    for (size_t i = 0; i < x1.size(); ++i) {
        distance += pow(x1[i] - x2[i], 2);
    }
    return exp(-gamma * distance);
}

// 计算预测值（使用 RBF 核）
double predict(const vector<vector<double>>& data, const vector<int>& labels, const vector<double>& alphas, const vector<double>& sample, double b, double gamma) {
    double prediction = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        if (alphas[i] > 0) { // 只考虑支持向量
            prediction += alphas[i] * labels[i] * rbfKernel(data[i], sample, gamma);
        }
    }
    prediction += b; // 加上偏置项
    return prediction;
}

// 训练带 RBF 核的 SVM 模型
void trainSVMWithRBF(vector<vector<double>>& data, vector<int>& labels, vector<double>& alphas, double& b, const SVMParams& params) {
    size_t numSamples = data.size();

    // 初始化拉格朗日乘子和偏置
    alphas.assign(numSamples, 0.0);
    b = 0.0;

    for (int epoch = 0; epoch < params.epochs; ++epoch) {
        double totalLoss = 0.0;

        for (size_t i = 0; i < numSamples; ++i) {
            // 计算预测值
            double prediction = predict(data, labels, alphas, data[i], b, params.gamma);

            // 计算误差
            double error = prediction - labels[i];
            totalLoss += abs(error);

            // 更新拉格朗日乘子
            alphas[i] -= params.learningRate * error;

            // 确保拉格朗日乘子非负
            alphas[i] = max(0.0, alphas[i]);

            // 更新偏置项
            b -= params.learningRate * error;
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
        {2.0, 3.0},   // 样本1
        {3.0, 3.0},   // 样本2
        {4.0, 5.0},   // 样本3
        {1.0, 1.0}    // 样本4
    };
    vector<int> labels = {1, 1, -1, -1}; // 对应样本的标签（+1 或 -1）

    // 设置超参数
    SVMParams params;

    // 初始化拉格朗日乘子和偏置
    vector<double> alphas;
    double b;

    // 训练模型
    trainSVMWithRBF(data, labels, alphas, b, params);

    // 输出最终的拉格朗日乘子和偏置
    cout << "Trained Alphas: ";
    for (double alpha : alphas) {
        cout << alpha << " ";
    }
    cout << endl;
    cout << "Trained Bias (b): " << b << endl;

    // 测试模型
    vector<double> testSample = {2.5, 3.5}; // 测试样本
    double prediction = predict(data, labels, alphas, testSample, b, params.gamma);
    cout << "Test Sample Prediction: " << (prediction >= 0 ? 1 : -1) << endl;

    return 0;
}