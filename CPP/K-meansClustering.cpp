#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <random>

using namespace std;

// 样本点结构体
struct Point {
    vector<double> coordinates; // 坐标
    int clusterId = -1;         // 所属簇ID，默认为-1表示未分配
};

// 计算两个点之间的欧氏距离
double calculateDistance(const Point& p1, const Point& p2) {
    double distance = 0.0;
    for (size_t i = 0; i < p1.coordinates.size(); ++i) {
        distance += pow(p1.coordinates[i] - p2.coordinates[i], 2);
    }
    return sqrt(distance);
}

// 初始化簇中心
void initializeCentroids(vector<Point>& centroids, const vector<Point>& samples, int k) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, samples.size() - 1);

    // 随机选择k个不同的初始点作为簇中心
    vector<bool> selected(samples.size(), false);
    for (int i = 0; i < k; ++i) {
        while (true) {
            int index = dis(gen);
            if (!selected[index]) {
                centroids.push_back(samples[index]);
                selected[index] = true;
                break;
            }
        }
    }
}

// 分配样本到最近的簇中心
void assignClusters(vector<Point>& samples, const vector<Point>& centroids) {
    for (auto& sample : samples) {
        double minDistance = numeric_limits<double>::max();
        int closestCentroidIndex = -1;

        for (size_t i = 0; i < centroids.size(); ++i) {
            double distance = calculateDistance(sample, centroids[i]);
            if (distance < minDistance) {
                minDistance = distance;
                closestCentroidIndex = i;
            }
        }

        sample.clusterId = closestCentroidIndex;
    }
}

// 更新簇中心位置
void updateCentroids(vector<Point>& centroids, const vector<Point>& samples) {
    size_t numFeatures = centroids[0].coordinates.size();
    vector<int> counts(centroids.size(), 0);

    // 初始化簇中心坐标累加值
    for (auto& centroid : centroids) {
        fill(centroid.coordinates.begin(), centroid.coordinates.end(), 0.0);
    }

    // 累加属于每个簇的所有点的坐标
    for (const auto& sample : samples) {
        int clusterId = sample.clusterId;
        for (size_t i = 0; i < numFeatures; ++i) {
            centroids[clusterId].coordinates[i] += sample.coordinates[i];
        }
        counts[clusterId]++;
    }

    // 平均化得到新的簇中心坐标
    for (size_t i = 0; i < centroids.size(); ++i) {
        if (counts[i] > 0) {
            for (size_t j = 0; j < numFeatures; ++j) {
                centroids[i].coordinates[j] /= counts[i];
            }
        }
    }
}

// K-means 聚类主函数
void kMeansClustering(vector<Point>& samples, int k, int maxIterations) {
    vector<Point> centroids;
    initializeCentroids(centroids, samples, k);

    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        assignClusters(samples, centroids);
        updateCentroids(centroids, samples);

        // 输出当前迭代的信息
        cout << "Iteration " << iteration + 1 << ": ";
        for (size_t i = 0; i < centroids.size(); ++i) {
            cout << "Centroid " << i << ": [";
            for (double coord : centroids[i].coordinates) {
                cout << coord << " ";
            }
            cout << "] ";
        }
        cout << endl;
    }
}

int main() {
    // 示例数据集
    vector<Point> data = {
        {{1.0, 2.0}},   // 样本1
        {{2.0, 1.0}},   // 样本2
        {{4.0, 5.0}},   // 样本3
        {{5.0, 6.0}},   // 样本4
        {{8.0, 9.0}},   // 样本5
        {{9.0, 8.0}}    // 样本6
    };

    int k = 2; // 要划分成的簇数量
    int maxIterations = 100; // 最大迭代次数

    // 进行K-means聚类
    kMeansClustering(data, k, maxIterations);

    // 输出最终的聚类结果
    cout << "Final Clusters:" << endl;
    for (const auto& sample : data) {
        cout << "Sample: [";
        for (double coord : sample.coordinates) {
            cout << coord << " ";
        }
        cout << "], Cluster: " << sample.clusterId << endl;
    }

    return 0;
}