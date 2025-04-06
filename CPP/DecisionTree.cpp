#include <iostream>
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>

using namespace std;

// 定义样本结构
struct Sample {
    vector<int> features; // 特征值
    int label;            // 标签
};

// 计算熵
double calculateEntropy(const vector<Sample>& samples) {
    map<int, int> labelCounts;
    for (const auto& sample : samples) {
        labelCounts[sample.label]++;
    }

    double entropy = 0.0;
    int totalSamples = samples.size();
    for (const auto& pair : labelCounts) {
        double probability = static_cast<double>(pair.second) / totalSamples;
        entropy -= probability * log2(probability);
    }
    return entropy;
}

// 根据某个特征划分数据集
vector<vector<Sample>> splitDataByFeature(const vector<Sample>& samples, int featureIndex) {
    map<int, vector<Sample>> splitMap;
    for (const auto& sample : samples) {
        splitMap[sample.features[featureIndex]].push_back(sample);
    }

    vector<vector<Sample>> splits;
    for (const auto& pair : splitMap) {
        splits.push_back(pair.second);
    }
    return splits;
}

// 找到最佳分裂特征
int findBestFeatureToSplit(const vector<Sample>& samples, int numFeatures) {
    double baseEntropy = calculateEntropy(samples);
    double bestInfoGain = 0.0;
    int bestFeature = -1;

    for (int feature = 0; feature < numFeatures; ++feature) {
        vector<vector<Sample>> splits = splitDataByFeature(samples, feature);

        double newEntropy = 0.0;
        for (const auto& split : splits) {
            double weight = static_cast<double>(split.size()) / samples.size();
            newEntropy += weight * calculateEntropy(split);
        }

        double infoGain = baseEntropy - newEntropy;
        if (infoGain > bestInfoGain) {
            bestInfoGain = infoGain;
            bestFeature = feature;
        }
    }
    return bestFeature;
}

// 构建决策树节点
struct TreeNode {
    int featureIndex = -1;               // 分裂的特征索引
    map<int, TreeNode*> children;        // 子节点
    int leafLabel = -1;                  // 如果是叶节点，存储标签
};

// 递归构建决策树
TreeNode* buildDecisionTree(vector<Sample>& samples, int numFeatures) {
    // 如果所有样本属于同一类，则返回叶节点
    bool allSameLabel = true;
    int firstLabel = samples[0].label;
    for (const auto& sample : samples) {
        if (sample.label != firstLabel) {
            allSameLabel = false;
            break;
        }
    }
    if (allSameLabel) {
        TreeNode* leaf = new TreeNode();
        leaf->leafLabel = firstLabel;
        return leaf;
    }

    // 找到最佳分裂特征
    int bestFeature = findBestFeatureToSplit(samples, numFeatures);
    if (bestFeature == -1) {
        // 如果无法分裂，返回多数类作为叶节点
        map<int, int> labelCounts;
        for (const auto& sample : samples) {
            labelCounts[sample.label]++;
        }
        int majorityLabel = -1;
        int maxCount = 0;
        for (const auto& pair : labelCounts) {
            if (pair.second > maxCount) {
                majorityLabel = pair.first;
                maxCount = pair.second;
            }
        }
        TreeNode* leaf = new TreeNode();
        leaf->leafLabel = majorityLabel;
        return leaf;
    }

    // 创建分裂节点
    TreeNode* node = new TreeNode();
    node->featureIndex = bestFeature;

    // 按最佳特征分裂数据
    vector<vector<Sample>> splits = splitDataByFeature(samples, bestFeature);
    for (auto& split : splits) {
        int value = split[0].features[bestFeature];
        node->children[value] = buildDecisionTree(split, numFeatures);
    }

    return node;
}

// 预测函数
int predict(const TreeNode* root, const vector<int>& features) {
    const TreeNode* node = root;
    while (node->featureIndex != -1) {
        int featureValue = features[node->featureIndex];
        if (node->children.find(featureValue) == node->children.end()) {
            // 如果未找到匹配的子节点，返回多数类
            break;
        }
        node = node->children.at(featureValue);
    }
    return node->leafLabel;
}

int main() {
    // 示例数据集
    vector<Sample> data = {
        {{0, 0}, 0}, // 样本1
        {{0, 1}, 0}, // 样本2
        {{1, 0}, 1}, // 样本3
        {{1, 1}, 1}  // 样本4
    };

    int numFeatures = 2; // 特征数量

    // 构建决策树
    TreeNode* root = buildDecisionTree(data, numFeatures);

    // 测试模型
    vector<int> testSample = {1, 0}; // 测试样本
    int prediction = predict(root, testSample);
    cout << "Test Sample Prediction: " << prediction << endl;

    return 0;
}