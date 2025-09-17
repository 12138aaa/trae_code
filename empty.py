import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression

# # 设置中文显示
# plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
# plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# # 加载加州房价数据集（替代波士顿房价数据集）
# california = fetch_california_housing()
# X = pd.DataFrame(california.data, columns=california.feature_names)
# y = pd.Series(california.target, name='MedHouseVal')  # 目标变量：房屋中位数价值（单位：10万美元）
# data = pd.concat([X, y], axis=1)

# # 数据基本信息
# print("数据集形状:", data.shape)
# print("\n特征名称:", california.feature_names)
# print("\n数据前5行:")
# print(data.head())

# # 2. 相关性分析
# # 计算特征与目标变量的相关性
# corr_pearson = data.corrwith(y, method='pearson')  # 皮尔逊相关
# corr_spearman = data.corrwith(y, method='spearman')  # 斯皮尔曼相关

# # 排序相关性结果
# corr_pearson_sorted = corr_pearson.sort_values(ascending=False)
# corr_spearman_sorted = corr_spearman.sort_values(ascending=False)

# # 可视化特征与目标变量的相关性
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# corr_pearson_sorted.plot(kind='bar', color='steelblue')
# plt.title('特征与房价的皮尔逊相关性')
# plt.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='相关性阈值0.3')
# plt.axhline(y=-0.3, color='red', linestyle='--', alpha=0.7)
# plt.ylabel('相关系数')
# plt.legend()

# plt.subplot(1, 2, 2)
# corr_spearman_sorted.plot(kind='bar', color='salmon')
# plt.title('特征与房价的斯皮尔曼相关性')
# plt.axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='相关性阈值0.3')
# plt.axhline(y=-0.3, color='red', linestyle='--', alpha=0.7)
# plt.ylabel('相关系数')
# plt.legend()

# plt.tight_layout()
# plt.show()

# # 特征间的相关性热力图
# plt.figure(figsize=(12, 10))
# correlation_matrix = data.corr(method='spearman')
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.title('特征间的相关性热力图')
# plt.tight_layout()
# plt.show()

# 3. 特征重要性评估（基于模型）
# 划分训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # 3.1 随机森林特征重要性
# rf = RandomForestRegressor(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)

# # 获取默认特征重要性（基于不纯度）
# rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
# rf_importance_sorted = rf_importance.sort_values(ascending=False)

# # 3.2 排列重要性（更稳健）
# perm_result = permutation_importance(
#     rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
# )
# perm_importance = pd.Series(perm_result.importances_mean, index=X.columns)
# perm_importance_sorted = perm_importance.sort_values(ascending=False)

# # 可视化随机森林特征重要性
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# rf_importance_sorted.plot(kind='bar', color='forestgreen')
# plt.title('随机森林默认特征重要性（不纯度）')
# plt.ylabel('重要性得分')

# plt.subplot(1, 2, 2)
# perm_importance_sorted.plot(kind='bar', color='darkorange')
# plt.title('随机森林排列特征重要性')
# plt.ylabel('性能下降幅度')

# plt.tight_layout()
# plt.show()


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # 假设数据集
# data = {
#     '身高': [170, 165, 180, 175, 160],
#     '体重': [70, 60, 85, 75, 55],
#     '年龄': [25, 22, 30, 28, 20]
# }
# df = pd.DataFrame(data)

# # 计算皮尔逊相关性矩阵
# corr_matrix = df.corr(method='pearson')

# # 绘制热图
# sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
# plt.title('相关性矩阵热图')
# plt.show()

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.datasets import load_iris
import pandas as pd

# 加载数据集（以 iris 数据集为例）
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)  # 特征
y = iris.target  # 目标变量（分类）

# 确保特征值为非负（卡方检验要求）
X = X.apply(lambda x: x - x.min() if x.min() < 0 else x)  # 转换为非负值

# # 使用 SelectKBest 和卡方检验选择前 2 个特征
# selector = SelectKBest(score_func=chi2, k=2)
# X_new = selector.fit_transform(X, y)

# # 输出选择的特征
# selected_features = X.columns[selector.get_support()].tolist()
# print("选择的特征：", selected_features)
# print("卡方统计量：", selector.scores_)

for column in X.columns:
    print(column)