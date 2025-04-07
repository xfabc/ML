import math
from collections import Counter

# 示例数据集
data = [
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes'],
    ['Sunny', 'Mild', 'Normal', 'Strong', 'Yes'],
    ['Overcast', 'Mild', 'High', 'Strong', 'Yes'],
    ['Overcast', 'Hot', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Strong', 'No']
]

features = ['Outlook', 'Temperature', 'Humidity', 'Wind']

# 计算信息熵
def entropy(data):
    labels = [row[-1] for row in data]
    total = len(labels)
    label_counts = Counter(labels)
    entropy = 0
    for count in label_counts.values():
        prob = count / total
        entropy -= prob * math.log2(prob)
    return entropy

# 计算信息增益
def information_gain(data, feature_index):
    total_entropy = entropy(data)
    total_len = len(data)
    feature_values = set(row[feature_index] for row in data)
    weighted_entropy = 0
    for value in feature_values:
        subset = [row for row in data if row[feature_index] == value]
        subset_entropy = entropy(subset)
        weighted_entropy += (len(subset) / total_len) * subset_entropy
    return total_entropy - weighted_entropy

# 选择最佳特征
def choose_best_feature(data):
    num_features = len(data[0]) - 1
    best_feature = -1
    max_gain = -1
    for i in range(num_features):
        gain = information_gain(data, i)
        if gain > max_gain:
            max_gain = gain
            best_feature = i
    return best_feature

# 多数投票
def majority_vote(labels):
    counter = Counter(labels)
    return counter.most_common(1)[0][0]

# 构建决策树
def build_tree(data, features):
    labels = [row[-1] for row in data]
    if len(set(labels)) == 1:
        return labels[0]
    if len(data[0]) == 1:
        return majority_vote(labels)
    best_feature_index = choose_best_feature(data)
    best_feature = features[best_feature_index]
    tree = {best_feature: {}}
    feature_values = set(row[best_feature_index] for row in data)
    for value in feature_values:
        subset = [row[:best_feature_index] + row[best_feature_index+1:] for row in data if row[best_feature_index] == value]
        subtree = build_tree(subset, features[:best_feature_index] + features[best_feature_index+1:])
        tree[best_feature][value] = subtree
    return tree

# 预测
def predict(tree, features, sample):
    if not isinstance(tree, dict):
        return tree
    root_feature = list(tree.keys())[0]
    feature_index = features.index(root_feature)
    value = sample[feature_index]
    if value not in tree[root_feature]:
        return None
    subtree = tree[root_feature][value]
    return predict(subtree, features, sample)

# 构建决策树
tree = build_tree(data, features)
print("决策树:", tree)

# 测试预测
sample = ['Sunny', 'Cool', 'High', 'Strong']
prediction = predict(tree, features, sample)
print("预测结果:", prediction)