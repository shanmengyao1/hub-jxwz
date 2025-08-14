import  pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# 导入数据集
dataset = pd.read_csv("dataset.csv", sep = "\t", header=None)
# 数据处理
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
# 进行特征提取
vectorizer = CountVectorizer()
input_feature = vectorizer.fit_transform(input_sentence.values)
# 训练模型
model1 = KNeighborsClassifier() # KNN模型
model1.fit(input_feature, dataset[1].values)
model2 = LogisticRegression()   # 逻辑回归模型
model2.fit(input_feature, dataset[1].values)
# 测试文本处理
test_query = input("待预测的文本：")
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vectorizer.transform([test_sentence])
# 模型推理
prediction1 = model1.predict(test_feature)
prediction2 = model2.predict(test_feature)

print("KNN模型的预测结果：", prediction1)
print("逻辑回归模型的预测结果：", prediction2)
