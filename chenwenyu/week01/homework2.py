#week01/homework2.py
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 2025
Function: Use SVM and XGBoost to classify exist text data
Author: Chen Wenyu
"""
import chardet
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# 1. 数据加载和预处理
with open('./Week01/dataset.csv', 'rb') as f:
    result = chardet.detect(f.read(10000))
print("检测到的编码:", result['encoding'])

df = pd.read_csv('./Week01/dataset.csv', 
                 header=None, 
                 names=['raw_data'],
                 encoding=result['encoding'])

df[['text', 'label']] = df['raw_data'].str.split('\t', expand=True)
df = df.drop(columns=['raw_data']).dropna()

# 2. 中文分词
jieba.add_word("双鸭山")  # 添加自定义词典
jieba.add_word("墓王之王")
texts = [' '.join(jieba.cut(str(text))) for text in df['text']]
labels = df['label'].values

# 3. TF-IDF向量化（最简）
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(texts)

# 4. 划分数据集（添加label encoder）
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, labels, df.index, test_size=0.2, random_state=42
)

# 标签编码（XGBoost需要数值型标签）
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# 5. 训练SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("\nSVM分类报告:")
print(classification_report(y_test, y_pred_svm))

# 6.训练XGBoost模型
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
   random_state=42,
    tree_method='hist'  # 兼容性更好
)

xgb.fit(X_train, y_train_encoded)
y_pred_xgb = le.inverse_transform(xgb.predict(X_test))

print("\nXGBoost分类报告:")
print(classification_report(y_test, y_pred_xgb))

# 7. 对比模型准确率
print(f"SVM测试集准确率: {svm.score(X_test, y_test):.4f}")
print(f"XGBoost测试集准确率: {xgb.score(X_test, y_test_encoded):.4f}")

# 8. 错误分析（示例）
wrong_cases_svm = df.loc[idx_test[y_test != y_pred_svm]][['text', 'label']]
print("\nSVM错误样本示例（10条）:")
print(wrong_cases_svm.sample(10))
wrong_cases_xg = df.loc[idx_test[y_test != y_pred_xgb]][['text', 'label']]
print("\nXGBoost错误样本示例（10条）:")
print(wrong_cases_xg.sample(10))