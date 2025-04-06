#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

// 定义超参数
struct SVMParams {
    double C = 1.0;       // 正则化参数
    double tol = 1e-3;    // 容忍误差
    double gamma = 0.5;   // RBF 核的参数
    int maxPasses = 1000;   // 最大迭代次数
};

// 计算 RBF 核
double rbfKernel(const vector<double>& x1, const vector<double>& x2, double gamma) {
    double distance = 0.0;
    for (size_t i = 0; i < x1.size(); ++i) {
        distance += pow(x1[i] - x2[i], 2);
    }
    return exp(-gamma * distance);
}

// 计算误差
double calculateError(const vector<vector<double>>& data, const vector<int>& labels, const vector<double>& alphas, double b, int i, double gamma) {
    double prediction = 0.0;
    for (size_t j = 0; j < data.size(); ++j) {
        if (alphas[j] > 0) { // 只考虑支持向量
            prediction += alphas[j] * labels[j] * rbfKernel(data[j], data[i], gamma);
        }
    }
    prediction += b; // 加上偏置项
    return prediction - labels[i];
}

// 选择第二个拉格朗日乘子
int selectSecondAlpha(int firstIndex, const vector<double>& errors, size_t numSamples) {
    int secondIndex = -1;
    double maxDeltaE = 0.0;
    for (int i = 0; i < numSamples; ++i) {
        if (i == firstIndex) continue;
        double deltaE = abs(errors[firstIndex] - errors[i]);
        if (deltaE > maxDeltaE) {
            maxDeltaE = deltaE;
            secondIndex = i;
        }
    }
    return secondIndex;
}

// 优化两个拉格朗日乘子
bool optimizeTwoAlphas(vector<vector<double>>& data, vector<int>& labels, vector<double>& alphas, double& b, double C, double gamma, int i1, int i2, vector<double>& errors) {
    if (i1 == i2) return false;

    double y1 = labels[i1], y2 = labels[i2];
    double alpha1 = alphas[i1], alpha2 = alphas[i2];

    double E1 = errors[i1], E2 = errors[i2];
    double k11 = rbfKernel(data[i1], data[i1], gamma);
    double k12 = rbfKernel(data[i1], data[i2], gamma);
    double k22 = rbfKernel(data[i2], data[i2], gamma);

    double eta = 2 * k12 - k11 - k22;
    if (eta >= 0) return false;

    double L, H;
    if (y1 != y2) {
        L = max(0.0, alpha2 - alpha1);
        H = min(C, C + alpha2 - alpha1);
    } else {
        L = max(0.0, alpha1 + alpha2 - C);
        H = min(C, alpha1 + alpha2);
    }
    if (L == H) return false;

    double newAlpha2 = alpha2 - y2 * (E1 - E2) / eta;
    newAlpha2 = max(L, min(H, newAlpha2));

    if (abs(newAlpha2 - alpha2) < 1e-5) return false;

    double newAlpha1 = alpha1 + y1 * y2 * (alpha2 - newAlpha2);

    // 更新偏置项 b
    double b1 = E1 + y1 * (newAlpha1 - alpha1) * k11 + y2 * (newAlpha2 - alpha2) * k12 + b;
    double b2 = E2 + y1 * (newAlpha1 - alpha1) * k12 + y2 * (newAlpha2 - alpha2) * k22 + b;
    b = (b1 + b2) / 2.0;

    // 更新拉格朗日乘子
    alphas[i1] = newAlpha1;
    alphas[i2] = newAlpha2;

    // 更新误差缓存
    errors[i1] = calculateError(data, labels, alphas, b, i1, gamma);
    errors[i2] = calculateError(data, labels, alphas, b, i2, gamma);

    return true;
}

// 使用 SMO 算法训练 SVM 模型
void trainSVMWithSMO(vector<vector<double>>& data, vector<int>& labels, vector<double>& alphas, double& b, const SVMParams& params) {
    size_t numSamples = data.size();

    // 初始化拉格朗日乘子、偏置和误差缓存
    alphas.assign(numSamples, 0.0);
    b = 0.0;
    vector<double> errors(numSamples, 0.0);
    for (int i = 0; i < numSamples; ++i) {
        errors[i] = calculateError(data, labels, alphas, b, i, params.gamma);
    }

    int passes = 0;
    while (passes < params.maxPasses) {
        int numChangedAlphas = 0;

        for (int i = 0; i < numSamples; ++i) {
            double Ei = errors[i];
            if ((labels[i] * Ei < -params.tol && alphas[i] < params.C) || (labels[i] * Ei > params.tol && alphas[i] > 0)) {
                // 选择第二个拉格朗日乘子
                int j = selectSecondAlpha(i, errors, numSamples);
                if (j == -1) continue;

                // 优化两个拉格朗日乘子
                if (optimizeTwoAlphas(data, labels, alphas, b, params.C, params.gamma, i, j, errors)) {
                    numChangedAlphas++;
                }
            }
        }

        if (numChangedAlphas == 0) {
            passes++;
        } else {
            passes = 0;
        }
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
    trainSVMWithSMO(data, labels, alphas, b, params);

    // 输出最终的拉格朗日乘子和偏置
    cout << "Trained Alphas: ";
    for (double alpha : alphas) {
        cout << alpha << " ";
    }
    cout << endl;
    cout << "Trained Bias (b): " << b << endl;

    // 测试模型
    vector<double> testSample = {2.5, 3.5}; // 测试样本
    double prediction = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        if (alphas[i] > 0) {
            prediction += alphas[i] * labels[i] * rbfKernel(data[i], testSample, params.gamma);
        }
    }
    prediction += b;
    cout << "Test Sample Prediction: " << (prediction >= 0 ? 1 : -1) << endl;

    return 0;
}