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
};

// 计算预测值
int predict(const vector<double>& weights, const vector<double>& features) {
    double score = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
        score += weights[i] * features[i];
    }
    return (score >= 0) ? 1 : -1; // 输出 +1 或 -1
}

// 训练线性 SVM 模型
void trainLinearSVM(vector<vector<double>>& data, vector<int>& labels, vector<double>& weights, const SVMParams& params) {
    size_t numSamples = data.size();
    size_t numFeatures = weights.size();

    for (int epoch = 0; epoch < params.epochs; ++epoch) {
        double totalLoss = 0.0;

        for (size_t i = 0; i < numSamples; ++i) {
            double prediction = 0.0;
            for (size_t j = 0; j < numFeatures; ++j) {
                prediction += weights[j] * data[i][j];
            }

            // 计算损失（Hinge Loss）
            double hingeLoss = max(0.0, 1 - labels[i] * prediction);
            totalLoss += hingeLoss;

            // 更新权重
            if (hingeLoss > 0) {
                for (size_t j = 0; j < numFeatures; ++j) {
                    weights[j] -= params.learningRate * (params.lambda * weights[j] - labels[i] * data[i][j]);
                }
            } else {
                for (size_t j = 0; j < numFeatures; ++j) {
                    weights[j] -= params.learningRate * (params.lambda * weights[j]);
                }
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
        {2.0, 3.0},   // 样本1
        {3.0, 3.0},   // 样本2
        {4.0, 5.0},   // 样本3
        {1.0, 1.0}    // 样本4
    };
    vector<int> labels = {1, 1, -1, -1}; // 对应样本的标签（+1 或 -1）

    // 添加偏置项（常数项）
    for (auto& sample : data) {
        sample.insert(sample.begin(), 1.0); // 在每个样本前插入1.0作为偏置
    }

    // 初始化权重（包括偏置项）
    size_t numFeatures = data[0].size();
    vector<double> weights(numFeatures, 0.0);

    // 设置超参数
    SVMParams params;

    // 训练模型
    trainLinearSVM(data, labels, weights, params);

    // 输出最终的权重
    cout << "Trained Weights: ";
    for (double weight : weights) {
        cout << weight << " ";
    }
    cout << endl;

    // 测试模型
    vector<double> testSample = {1.0, 2.5, 3.0}; // 测试样本（包括偏置项）
    int prediction = predict(weights, testSample);
    cout << "Test Sample Prediction: " << prediction << endl;

    return 0;
}