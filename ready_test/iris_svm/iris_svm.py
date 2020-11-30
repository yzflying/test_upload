import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
from sklearn import svm  #导入svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score   #导入分类准确率
# from sklearn.exceptions import ChangedBehaviorWarning
from sklearn.preprocessing import LabelEncoder


"""此处需要注意，csv文件与xls文件不能通过改文件后缀名来转换，而是要打开文件，另存为希望格式的文件"""
data = pd.read_csv('iris.csv')
x, y = data[['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']], data['Species']
"""对标签y进行one-hot编码"""
encoder = LabelEncoder().fit(y)
y = encoder.transform(y)   #对标签y进行数字化编码,150 X 1


# """划分训练集、验证集"""
# x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size=0.7)  # random_state=0或不指定，每次获取的随机样本不一致
# """svm.SVC分类建模，训练"""
# clf = svm.SVC(C=0.5530, kernel='linear', gamma=13.1739)
# """
# C为容差正则项的惩罚系数，默认为1；C越大，即对分错样本的惩罚程度越大，因此在训练样本中准确率越高，但是泛化能力降低
# kernel为核函数类型，主要有线性核linear、高斯核rbf；一般属性较少时可以用高斯核进行多维扩展
# gamma为核函数rbf系数，默认为1/feature_num；越大则训练集拟合越好，但可能过拟合
# """
# clf.fit(x_train, y_train)


# """clf.score模型正确率；clf.predict模型预测值；clf.decision_function模型计算值与各分类标签的距离"""
# print(clf.score(x_train, y_train))
# print('训练集准确率：', accuracy_score(y_train, clf.predict(x_train)))
# print(clf.score(x_test, y_test))
# print('测试集准确率：', accuracy_score(y_test, clf.predict(x_test)))
# print('decision_function:', clf.decision_function(x_train))   #120*3的数组，每行代表模型预测值与各类别的距离
# print('predict:', clf.predict(x_train))  #120*1的数组，代表模型预测的类别


# """作图：训练数据集'Sepal.Length', 'Sepal.Width'属性的散点图；x_train.values将DataFrame对象X_df转成ndarray数组"""
# plt.scatter(x_train.values[:, 0], x_train.values[:, 1], c=y_train)     # 训练样本散点图，按真实标签分类着色
# plt.show()
# plt.scatter(x_train.values[:, 0], x_train.values[:, 1], c=clf.predict(x_train)) # 训练样本散点图，按预测标签分类着色
# plt.show()


"""hyperopt模块调参"""
from hyperopt import hp
from hyperopt import fmin, tpe, Trials, STATUS_OK


"""定义最小化函数f"""
def f(params):
    # x_ = x[:]
    clf = svm.SVC(**params)
    acc = cross_val_score(clf, x, y, cv=10).mean()     # 交叉验证得分
    return {'loss': -acc, 'status': STATUS_OK}


"""定义各参数取值范围或取值类型"""
space4svm = {
    'C': hp.uniform('C', 0, 20),
    'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'poly', 'rbf']),
    'gamma': hp.uniform('gamma', 0, 20),
}

"""Trials对象允许我们在每个时间步存储信息"""
trials = Trials()
"""fmin接受一个函数f进行最小化f；space4svm给定函数f的各个参数取值方式与范围；algo指定搜索策略；max_evals指定fmin的优化次数"""
best = fmin(f, space4svm, algo=tpe.suggest, max_evals=100, trials=trials)
print("best params is :", best)
"""打印最后10个时间步的参数信息"""
for trial in trials.trials[0:100]:
    print("trials is :", trial)


"""调参过程的可视化"""
xs = [t['tid'] for t in trials.trials]
C_ys = [t['misc']['vals']['C'] for t in trials.trials]
gamma_ys = [t['misc']['vals']['gamma'] for t in trials.trials]
plt.plot(xs, C_ys)
plt.show()
plt.plot(xs, gamma_ys)
plt.show()
