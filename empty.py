#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, chi2


def load_data():
    """加载鸢尾花数据集并返回特征和目标变量"""
    print("正在加载数据集...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='species')
    target_names = iris.target_names
    
    # 将数值目标转换为类别名称
    y_names = pd.Series([target_names[i] for i in y], name='species_name')
    
    return X, y, y_names, target_names


def explore_data(X, y_names):
    """进行探索性数据分析"""
    print("\n数据探索:")
    print(f"数据集形状: {X.shape}")
    
    # 合并特征和目标变量
    data = pd.concat([X, y_names], axis=1)
    
    print("\n数据前5行:")
    print(data.head())
    
    print("\n数据统计摘要:")
    print(X.describe())
    
    return data


def visualize_data(X, y, target_names):
    """可视化数据分布和关系"""
    print("\n生成数据可视化...")
    
    # 设置图表风格
    sns.set(style="whitegrid")
    
    # 为不同类别设置颜色
    colors = ['blue', 'orange', 'green']
    
    # 创建散点图矩阵
    plt.figure(figsize=(12, 10))
    scatter_matrix = pd.plotting.scatter_matrix(X, c=y, figsize=(12, 10),
                                               marker='o', hist_kwds={'bins': 20},
                                               s=60, alpha=0.8, cmap='viridis')
    
    # 添加图例
    plt.suptitle('鸢尾花数据集特征散点图矩阵', fontsize=16)
    
    # 创建箱线图
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(X.columns):
        plt.subplot(2, 2, i+1)
        for j, species in enumerate(np.unique(y)):
            plt.boxplot(X[feature][y == species], positions=[j+1], 
                       widths=0.6, patch_artist=True,
                       boxprops=dict(facecolor=colors[j], color='black'),
                       medianprops=dict(color='black'))
        plt.title(f'{feature} 分布')
        plt.xticks([1, 2, 3], target_names)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle('各特征在不同类别中的分布', fontsize=16)
    
    # 相关性热图
    plt.figure(figsize=(10, 8))
    correlation_matrix = X.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('特征相关性热图', fontsize=14)
    
    # 保存图表而不是显示
    plt.savefig('data_visualization.png')
    print("可视化图表已保存为 'data_visualization.png'")


def feature_selection(X, y):
    """使用卡方检验进行特征选择"""
    print("\n进行特征选择...")
    
    # 确保特征值为非负（卡方检验要求）
    X_nonneg = X.apply(lambda x: x - x.min() if x.min() < 0 else x)
    
    # 使用SelectKBest和卡方检验选择特征
    selector = SelectKBest(score_func=chi2, k=2)
    X_new = selector.fit_transform(X_nonneg, y)
    
    # 获取选择的特征
    selected_features = X.columns[selector.get_support()].tolist()
    feature_scores = pd.Series(selector.scores_, index=X.columns)
    
    print(f"选择的特征: {selected_features}")
    print("\n特征重要性分数:")
    for feature, score in feature_scores.sort_values(ascending=False).items():
        print(f"  {feature}: {score:.4f}")
    
    return X_new, selected_features, feature_scores


def train_model(X, y, selected_features):
    """使用选择的特征训练随机森林模型"""
    print("\n使用选择的特征训练模型...")
    
    # 只使用选择的特征
    X_selected = X[selected_features]
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 训练随机森林分类器
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 在测试集上评估模型
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n模型准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    return model, accuracy


def main():
    """主函数"""
    print("=== 鸢尾花数据集分析与特征选择 ===\n")
    
    # 加载数据
    X, y, y_names, target_names = load_data()
    
    # 探索数据
    data = explore_data(X, y_names)
    
    # 可视化数据
    visualize_data(X, y, target_names)
    
    # 特征选择
    X_new, selected_features, feature_scores = feature_selection(X, y)
    
    # 训练模型
    model, accuracy = train_model(X, y, selected_features)
    
    print("\n分析完成!")
    
    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    feature_scores.sort_values().plot(kind='barh', color='teal')
    plt.xlabel('卡方统计量')
    plt.ylabel('特征')
    plt.title('特征重要性排序')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("特征重要性图表已保存为 'feature_importance.png'")


if __name__ == "__main__":
    main()