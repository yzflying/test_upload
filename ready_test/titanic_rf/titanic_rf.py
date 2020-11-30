import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns    #mpl基础上的一个图形包
import os
import sys
import time
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


"""读取数据，dataframe数据类型"""
train_valid_df = pd.read_csv('train.csv')     #默认header=0,即第0行为列名
test_df = pd.read_csv('test.csv')


"""将Sex、Embarked属性进行编码"""
encoder = LabelEncoder().fit(train_valid_df['Sex'])
train_valid_df['Sex'] = encoder.transform(train_valid_df['Sex'])   #对属性Sex进行数字化编码
train_valid_df['Embarked'] = train_valid_df['Embarked'].fillna('S')  #编码前对空值进行填充
encoder = LabelEncoder().fit(train_valid_df['Embarked'])
train_valid_df['Embarked'] = encoder.transform(train_valid_df['Embarked'])   #对行数字化编码


"""预处理，age填充"""
train_valid_df['Age'] = train_valid_df['Age'].fillna(train_valid_df['Age'].median())
"""训练"""
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]   #挑选出对标签有影响的属性


random_forest = RandomForestClassifier(n_estimators=100)
"""
n_estimators：随机森林中树的数量；较多的树能让模型性能更好，但时间复杂度高，默认10
max_features：树能使用的特征数量；默认没有限制
max_depth ：树深
'criterion': 0, 'max_depth': 8, 'max_features': 2, 'n_estimators': 13}
"""
random_forest.fit(train_valid_df[predictors], train_valid_df["Survived"])
print(random_forest.score(train_valid_df[predictors], train_valid_df["Survived"]))


# """特征重要性排序"""
# from sklearn.ensemble import RandomForestRegressor
#
#
# rf = RandomForestRegressor()
# rf.fit(train_valid_df[predictors], train_valid_df["Survived"])
# print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), predictors), reverse=True))


# """hyperopt模块调参"""
from hyperopt import hp
from hyperopt import fmin, tpe, Trials, STATUS_OK


X = train_valid_df[predictors]
y = train_valid_df["Survived"]


def hyperopt_train_test(params):
    X_ = X[:]
    # if 'normalize' in params:
    #     if params['normalize'] == 1:
    #         X_ = normalize(X_)
    #         del params['normalize']
    #
    # if 'scale' in params:
    #     if params['scale'] == 1:
    #         X_ = scale(X_)
    #         del params['scale']
    clf = RandomForestClassifier(**params)
    return cross_val_score(clf, X, y).mean()

space4rf = {
    'max_depth': hp.choice('max_depth', range(1,20)),
    'max_features': hp.choice('max_features', range(1,5)),
    'n_estimators': hp.choice('n_estimators', range(1,20)),
    'criterion': hp.choice('criterion', ["gini", "entropy"]),
    # 'scale': hp.choice('scale', [0, 1]),
    # 'normalize': hp.choice('normalize', [0, 1])
}


# best = 0
def f(params):
    # global best
    acc = hyperopt_train_test(params)
    # if acc > best:
    # best = acc
    return {'loss': -acc, 'status': STATUS_OK}

trials = Trials()
best = fmin(f, space4rf, algo=tpe.suggest, max_evals=300, trials=trials)
print(best)