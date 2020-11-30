"""
home-credit-default-risk：
还款能力预测，二分类问题
application.csv 主要的训练测试集。每行表示一笔贷款，由特征SK_ID_CURR标识
bureau.csv 客户历史贷款记录。每行表示一笔贷款，由特征SK_ID_BUREAU标识，application.csv中的每笔贷款客户在bureau.csv中可能有多个贷款记录
bureau_balance.csv 客户历史贷款记录bureau.csv中的每笔贷款的月度数据。一笔贷款SK_ID_BUREAU对应多个月
previous_application.csv  客户在本行的历史贷款记录。每行表示一笔贷款，由特征SK_ID_PREV标识，application.csv中的每笔贷款客户在本表中可能有多个贷款记录
POS_CASH_balance.csv  客户在本行的历史贷款记录每笔贷款的月度数据，一笔贷款SK_ID_PREV对应多个月
credit_card_balance.csv  信用卡客户在本行的历史贷款记录每笔贷款月度数据，每个信用卡可以对应多条数据
installments_payments.csv  客户在本行的历史贷款支付记录，每行代表一个支付

评价标准ROC/AUC：
当测试集中的正负样本的分布变化的时候，ROC曲线能够保持不变
ROC全称是“受试者工作特征”（Receiver Operating Characteristic）。ROC曲线的面积就是AUC（Area Under the Curve）。AUC用于衡量“二分类问题”机器学习算法性能（泛化能力）
TP：预测类别是P（正例），真实类别也是P
FP：预测类别是P，真实类别是N（反例）
TN：预测类别是N，真实类别也是N
FN：预测类别是N，真实类别是P
样本中的真实正例类别总数即TP+FN。TPR即True Positive Rate，TPR = TP/(TP+FN)
样本中的真实反例类别总数为FP+TN。FPR即False Positive Rate，FPR = FP/(TN+FP)
截断点取不同的值，TPR和FPR的计算结果也不同
将(FPR, TPR)作图得到曲线即ROC曲线，曲线围的面积即AUC，AUC的取值范围在0.5和1之间，目的是将AUC最大化
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import os


"""读取application.csv数据"""
app_train = pd.read_csv('application_train.csv')
print('Training data shape: ', app_train.shape)         # (307511, 122)
app_test = pd.read_csv('application_test.csv')
print('Testing data shape: ', app_test.shape)           # (48744, 121)


"""统计application.csv的target列的值分布，不平衡样本问题"""
# print('Distribution of the Target Column:', app_train['TARGET'].value_counts())    # 0：282686，1：24825
# sns.countplot(app_train['TARGET'])
# plt.show()


"""统计application.csv各属性空值"""
"""对于缺失比率较高的属性，可以舍弃，也可以利用xgboost遍历"""
def missing_values_table(df):
    """
    :param df: 待统计空值的表格
    :return: 输入表格df的各属性的空值数量Missing Values与比率
    """
    mis_val = df.isnull().sum()     # 各个属性的空值数量
    mis_val_percent = 100 * df.isnull().sum() / len(df)  # 各个属性的空值比率
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)  # 创建空值表mis_val_table，包含各属性的空值数量mis_val与比率mis_val_percent
    mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'Missing Values', 1: '% of Total Values'})
    # 筛选出mis_val_table有空值的属性，并依据空值比排序，所有数据保留一位小数
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values('% of Total Values', ascending=False).round(1)
    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"+"There are " + str(mis_val_table_ren_columns.shape[0])+" columns that have missing values.")
    # Return the dataframe with missing information
    return mis_val_table_ren_columns


missing_values = missing_values_table(app_train)   #121列中有67个属性含空值，空值最多的属性空值率达69.9%
# print(missing_values.head(20))


"""查看各属性数据类型"""
dtype_count = app_train.dtypes.reset_index()      # 各属性及其属性数据类型，reset_index()方法将series转为dataframe
dtype_count.columns = ['column_name', 'column_type']  # column重命名
print('column_type value_counts:\n', dtype_count['column_type'].value_counts())     # 各数据类型统计


"""查看object数据类型的各个属性取值范围个数"""
uni_num = app_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)  # object数据类型的各个属性去重后取值个数
print('object column_type nunique:\n', uni_num)


"""object数据类型的编码encoding"""
"""对于二类的object，使用label encoding(0、1)；对于多类的object，使用one-hot encoding"""
# 利用Scikit中的LabelEncoder来创建label encoder object
le = LabelEncoder()
le_count = 0
# 说明：EMERGENCYSTATE_MODE属性取值个数nunique有两个，另加部分空值，故计算unique列表为三个；不属于二类
for col in app_train:
    if app_train[col].dtype == 'object':
        if len(list(app_train[col].unique())) <= 2:         # 如果属性col去重unique后取值不大于2
            le.fit(app_train[col])                          # 对col编码
            app_train[col] = le.transform(app_train[col])   # 对数据集进行转化（属性值变为0、1）
            app_test[col] = le.transform(app_test[col])
            le_count += 1
print('%d columns were label encoded.' % le_count)
print('Training Features shape: ', app_train.shape)       # LabelEncoder编码后的列不变
print('Testing Features shape: ', app_test.shape)

# one-hot encoding
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)
print('Training Features shape: ', app_train.shape)  # one-hot编码会增加列(每个属性1列->每个属性unique列),属性值为0、1
print('Testing Features shape: ', app_test.shape)

# 因train与test数据集部分属性的unique不一致，导致one-hot后的列不一致；调整train与test数据集,使其有相同的特征名
train_labels = app_train['TARGET']
# Align the training and testing data, keep only columns present in both dataframes
app_train, app_test = app_train.align(app_test, join='inner', axis=1)   # inner join,删掉只在一个数据集中出现的columns
# Add the target back in
app_train['TARGET'] = train_labels
print('Training Features shape: ', app_train.shape)
print('Testing Features shape: ', app_test.shape)


"""数据探索"""
# 对'DAYS_BIRTH'属性(Client's age in days at the time of application)转化为贷款时客户年龄
app_train['DAYS_BIRTH'] = app_train['DAYS_BIRTH'].apply(lambda x: x/-365)
print(app_train['DAYS_BIRTH'].describe())       # 查看统计量，是否有异常值


# 对'DAYS_EMPLOYED'属性(How many days before the application the person started current employment)贷款时工作时长：days分析
print(app_train['DAYS_EMPLOYED'].describe())    # 可以看到最大工作时长较多样本为365243天，为异常值

# 探索'DAYS_EMPLOYED'属性是否异常与target列的关系，发现异常样本违约率较低(target均值)
anom = app_train[app_train['DAYS_EMPLOYED'] == 365243]          # 'DAYS_EMPLOYED'属性异常的样本
non_anom = app_train[app_train['DAYS_EMPLOYED'] != 365243]      # 'DAYS_EMPLOYED'属性无异常的样本
print('The non-anomalies default on %0.2f%% of loans' % (100 * non_anom['TARGET'].mean()))   # 正常样本target均值
print('The anomalies default on %0.2f%% of loans' % (100 * anom['TARGET'].mean()))          # 异常样本target均值
print('There are %d anomalous days of employment' % len(anom))      # 异常样本数量

# 创建一个新的column，'DAYS_EMPLOYED_ANOM'
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243   # True代表异常，False代表正常
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace=True)   # Replace the anomalous values with nan
app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace=True)
# # 'DAYS_EMPLOYED'值的分布直方图
# sns.countplot(app_train['DAYS_EMPLOYED'])
# # app_train['DAYS_EMPLOYED'].plot.hist(title='Days Employment Histogram')
# plt.show()


"""
探索各属性与target的相关性:spearman皮尔逊相关系数
可以发现年龄DAYS_BIRTH与target较正相关;EXT_SOURCE_1, EXT_SOURCE_2, and EXT_SOURCE_3与target较负相关
"""
# # app_train.corr()函数返回各属性之间的相关性，协方差矩阵形式
# correlations = app_train.corr()['TARGET'].sort_values()
# # Display correlations
# print('Most Positive Correlations:\n', correlations.tail(15))
# print('Most Negative Correlations:\n', correlations.head(15))
#
#
# # 客户年龄分段统计
# plt.hist(app_train['DAYS_BIRTH'], edgecolor='k', bins=25)
# plt.title('Age of Client')
# plt.xlabel('Age (years)')
# plt.ylabel('Count')
# plt.show()


"""
作kde图(核密度估计)来反映不同年龄与target的关系
核密度估计：即样本的分布密度函数，与样本的分段统计直方图类似，但避免了直方图分段数量的影响，直方图分布曲线不平滑的缺点。
"""
# plt.figure(figsize=(10, 8))
# # app_train.loc()函数选择target==0的行，column为DAYS_BIRTH的属性，即 mX1 大小，进行分布绘图
# sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, 'DAYS_BIRTH'], label='target == 0')
# # KDE plot of loans which were not repaid on time
# sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, 'DAYS_BIRTH'], label='target == 1')
# # Labeling of plot
# plt.xlabel('Age (years)')
# plt.ylabel('Density')
# plt.title('Distribution of Ages')
# plt.show()


"""对年龄进行分段，统计各段target==1(逾期)的比例"""
# 筛选出target、DAYS_BIRTH 属性
age_data = app_train[['TARGET', 'DAYS_BIRTH']]
# 对年龄DAYS_BIRTH分段，并统计各段target样本集的均值
age_data['DAYS_BIRTH'] = pd.cut(age_data['DAYS_BIRTH'], bins=np.linspace(20, 70, num=11))  # pd.cut()函数返回各样本DAYS_BIRTH属性的所属分段，并赋给DAYS_BIRTH属性
"""
   TARGET    DAYS_BIRTH
0       1  (25.0, 30.0]
"""
print(age_data.head(5))   # 示例如上

# age_data.groupby()函数将age_data数据集样本依据DAYS_BIRTH进行分组，返回一个DataFrameGroupBy对象；mean()方法计算各个组各个属性的平均值
age_groups = age_data.groupby('DAYS_BIRTH').mean()  # 对属性DAYS_BIRTH(即各分段年龄)统计TARGET均值
"""
                TARGET
DAYS_BIRTH            
(20.0, 25.0]  0.123036
"""
print(age_groups)
# 各分段年龄段的样本target均值，即target==1(逾期)的比例
# plt.figure(figsize=(8, 8))
# # Graph the age bins and the average of the target as a bar plot  依据age_groups.index和age_groups['TARGET']作条形图
# plt.bar(age_groups.index.astype(str), 100 * age_groups['TARGET'])
# # Plot labeling
# plt.xticks(rotation=75)
# plt.xlabel('Age Group (years)')
# plt.ylabel('Failure to Repay (%)')
# plt.title('Failure to Repay by Age Group')
# plt.show()


"""
探索EXT_SOURCE_1, EXT_SOURCE_2, and EXT_SOURCE_3与target的关系
可以发现以上三个属性与target呈现一种负相关的关系，且EXT_SOURCE_1与DAYS_BIRTH有一定程度的正相关
"""
# # 计算以上各属性之间的相关性
# ext_data = app_train[['TARGET', 'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
# ext_data_corrs = ext_data.corr()          # 相关系数矩阵
# print(ext_data_corrs)
# # 绘制颜色编码矩阵
# plt.figure(figsize=(8, 6))
# # Heatmap of correlations 将矩形数据绘制为颜色编码矩阵
# sns.heatmap(ext_data_corrs, cmap="RdYlBu_r", vmin=-0.25, annot=True, vmax=0.6)
# plt.title('Correlation Heatmap')
# plt.show()


"""
作kde图(核密度估计)来反映不同EXT_SOURCE与target的关系
"""
# plt.figure(figsize=(10, 12))
# # iterate through the sources
# for i, source in enumerate(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']):
#     # create a new subplot for each source
#     plt.subplot(3, 1, i + 1)
#     # plot repaid loans
#     sns.kdeplot(app_train.loc[app_train['TARGET'] == 0, source], label='target == 0')
#     # plot loans that were not repaid
#     sns.kdeplot(app_train.loc[app_train['TARGET'] == 1, source], label='target == 1')
#     # Label the plots
#     plt.title('Distribution of %s by Target Value' % source)
#     plt.xlabel('%s' % source)
#     plt.ylabel('Density')
# plt.tight_layout(h_pad=2.5)
# plt.show()


"""
特征工程：特征工程相较于模型和调参，有更大的回报
特征工程包含新特征的创建、已有特征的选择
同一个特征工程，在不同的模型可能对结果的提升不一样
"""
"""特征工程一：利用多项式PolynomialFeatures创建新特征（特征之间相乘来构造新特征）"""
from sklearn.preprocessing import Imputer   # 导入Imputer类，用于填充数据集缺失值
from sklearn.preprocessing import PolynomialFeatures


# # 创建poly_features、poly_features_test、poly_target数据集
# poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET']]
# poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']]
# poly_target = poly_features['TARGET']
# poly_features = poly_features.drop(columns=['TARGET'])      # poly_features用于特征工程的数据集，含4个关键属性
# # 缺失值填充
# imputer = Imputer(strategy='median')   # 创建imputer实例，用属性中值填充
# poly_features = imputer.fit_transform(poly_features)
# poly_features_test = imputer.transform(poly_features_test)
# # Create the polynomial object with specified degree
# poly_transformer = PolynomialFeatures(degree=3)    # 创建多项式的最高次幂，值越大则得到新特征越多
# # Train the polynomial features
# poly_transformer.fit(poly_features)
# # Transform the features
# poly_features = poly_transformer.transform(poly_features)
# poly_features_test = poly_transformer.transform(poly_features_test)
# print('Polynomial Features shape: ', poly_features.shape)         # 数据集新的维度为35
# print(poly_transformer.get_feature_names(input_features=['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))    # 打印各属性的默认名称


"""探索新创建的各特征与target的相关性,可以发现某些新特征比原始特征较target更有相关性"""
# # Create a dataframe of the features
# poly_features = pd.DataFrame(poly_features,
#                              columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))
# # Add in the target
# poly_features['TARGET'] = poly_target
# # Find the correlations with the target
# poly_corrs = poly_features.corr()['TARGET'].sort_values()
# # Display most negative and most positive
# print(poly_corrs.head(10))
# print(poly_corrs.tail(5))

"""对test数据集进行相同的处理，并对target与test合并"""
# # Put test features into dataframe
# poly_features_test = pd.DataFrame(poly_features_test,
#                                   columns=poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH']))
# # Merge polynomial features into training dataframe
# poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']
# app_train_poly = app_train.merge(poly_features, on='SK_ID_CURR', how='left')
# # Merge polnomial features into testing dataframe
# poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
# app_test_poly = app_test.merge(poly_features_test, on='SK_ID_CURR', how='left')
# # Align the dataframes
# app_train_poly, app_test_poly = app_train_poly.align(app_test_poly, join='inner', axis=1)   # inner join,删掉只在一个数据集中出现的columns
# # Print out the new shapes
# print('Training data with polynomial features shape: ', app_train_poly.shape)     # 特征个数为275
# print('Testing data with polynomial features shape:  ', app_test_poly.shape)


"""特征工程二：利用特征的实际代表意义进行新特征的创建"""
"""
CREDIT_INCOME_PERCENT: credit amount 相对于 client's income 的百分比
ANNUITY_INCOME_PERCENT: loan annuity 相对于 client's income 的百分比
CREDIT_TERM: the length of the payment in months (since the annuity is the monthly amount due
DAYS_EMPLOYED_PERCENT: days employed 相对于 client's age 的百分比
"""
app_train_domain = app_train.copy()
app_test_domain = app_test.copy()
# 创建新属性
app_train_domain['CREDIT_INCOME_PERCENT'] = app_train_domain['AMT_CREDIT'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['ANNUITY_INCOME_PERCENT'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_INCOME_TOTAL']
app_train_domain['CREDIT_TERM'] = app_train_domain['AMT_ANNUITY'] / app_train_domain['AMT_CREDIT']
app_train_domain['DAYS_EMPLOYED_PERCENT'] = app_train_domain['DAYS_EMPLOYED'] / app_train_domain['DAYS_BIRTH']
app_test_domain['CREDIT_INCOME_PERCENT'] = app_test_domain['AMT_CREDIT'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['ANNUITY_INCOME_PERCENT'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_INCOME_TOTAL']
app_test_domain['CREDIT_TERM'] = app_test_domain['AMT_ANNUITY'] / app_test_domain['AMT_CREDIT']
app_test_domain['DAYS_EMPLOYED_PERCENT'] = app_test_domain['DAYS_EMPLOYED'] / app_test_domain['DAYS_BIRTH']
# # 作kde图(核密度估计)来反映新属性与target的关系，没啥用
# plt.figure(figsize=(12, 20))
# # iterate through the new features
# for i, feature in enumerate(['CREDIT_INCOME_PERCENT', 'ANNUITY_INCOME_PERCENT', 'CREDIT_TERM', 'DAYS_EMPLOYED_PERCENT']):   # i为索引，feature为属性名
#     # create a new subplot for each source
#     plt.subplot(4, 1, i + 1)    # 4X1子图，占用其中的第i+1个子图(起始1，可选1、2、3、4)
#     # plot repaid loans
#     sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 0, feature], label='target == 0')
#     # plot loans that were not repaid
#     sns.kdeplot(app_train_domain.loc[app_train_domain['TARGET'] == 1, feature], label='target == 1')
#     # Label the plots
#     plt.title('Distribution of %s by Target Value' % feature)
#     plt.xlabel('%s' % feature)
#     plt.ylabel('Density')
# plt.tight_layout(h_pad=2.5)
# plt.show()


"""
创建逻辑回归模型，进行二分类,logistic regression成绩0.671
"""
from sklearn.preprocessing import MinMaxScaler, Imputer


"""首先进行缺失值填充和数据归一化"""
# # Drop the target from the training data
# if 'TARGET' in app_train:
#     train = app_train.drop(columns=['TARGET'])
# else:
#     train = app_train.copy()
# # Feature names
# features = list(train.columns)
# # Copy of the testing data
# test = app_test.copy()
# # Median imputation of missing values
# imputer = Imputer(strategy='median')
# # MinMaxScaler 归一化 x' = (x - X_min) / (X_max - X_min)
# # StandardScaler 标准化 x' = (x - μ)／σ
# scaler = MinMaxScaler(feature_range=(0, 1))
# # Fit on the training data
# imputer.fit(train)
# # Transform both training and testing data
# train = imputer.transform(train)
# test = imputer.transform(app_test)
# # Repeat with the scaler
# scaler.fit(train)
# train = scaler.transform(train)
# test = scaler.transform(test)
# print('Training data shape: ', train.shape)
# print('Testing data shape: ', test.shape)


"""建模训练"""
from sklearn.linear_model import LogisticRegression


# # Make the model with the specified regularization parameter
# log_reg = LogisticRegression(C=0.0001, class_weight=None, dual=False,
#           fit_intercept=True, intercept_scaling=1, max_iter=100,
#           multi_class='ovr', n_jobs=None, penalty='l2', random_state=None,
#           solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
# # Train on the training data
# log_reg.fit(train, train_labels)
# # Make predictions
# # Make sure to select the second column only
# # predict_proba()函数返回测试集test的预测结果，每个样本预测为每个结果的概率，列为分类数目，shape为(48744, 2)
# log_reg_pred = log_reg.predict_proba(test)[:, 1]
# # Submission dataframe
# submit = app_test[['SK_ID_CURR']]
# submit['TARGET'] = log_reg_pred
# # Save the submission to a csv file
# submit.to_csv('log_reg_baseline.csv', index=False)


"""使用随机森林Random Forest模型"""
from sklearn.ensemble import RandomForestClassifier


# # Make the random forest classifier
# random_forest = RandomForestClassifier(n_estimators=100, random_state=50, verbose=1, n_jobs=-1)
# # Train on the training data
# random_forest.fit(train, train_labels)
#
# # Extract feature importances
# feature_importance_values = random_forest.feature_importances_
# feature_importances = pd.DataFrame({'feature': features, 'importance': feature_importance_values})
#
# # Make predictions on the test data
# predictions = random_forest.predict_proba(test)[:, 1]
#
# # Make a submission dataframe
# submit = app_test[['SK_ID_CURR']]
# submit['TARGET'] = predictions
# # Save the submission dataframe
# submit.to_csv('random_forest_baseline.csv', index=False)


"""使用多项式特征工程数据进行随机森林建模"""
# poly_features_names = list(app_train_poly.columns)
# # Impute the polynomial features
# imputer = Imputer(strategy = 'median')
# poly_features = imputer.fit_transform(app_train_poly)
# poly_features_test = imputer.transform(app_test_poly)
#
# # Scale the polynomial features
# scaler = MinMaxScaler(feature_range = (0, 1))
#
# poly_features = scaler.fit_transform(poly_features)
# poly_features_test = scaler.transform(poly_features_test)
#
# random_forest_poly = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
#
# # Train on the training data
# random_forest_poly.fit(poly_features, train_labels)
#
# # Make predictions on the test data
# predictions = random_forest_poly.predict_proba(poly_features_test)[:, 1]
#
# # Make a submission dataframe
# submit = app_test[['SK_ID_CURR']]
# submit['TARGET'] = predictions
# # Save the submission dataframe
# submit.to_csv('random_forest_baseline_engineered.csv', index = False)


"""使用领域知识新建特征的特征工程数据进行随机森林建模"""
# app_train_domain = app_train_domain.drop(columns = 'TARGET')
# domain_features_names = list(app_train_domain.columns)
#
# # Impute the domainnomial features
# imputer = Imputer(strategy = 'median')
#
# domain_features = imputer.fit_transform(app_train_domain)
# domain_features_test = imputer.transform(app_test_domain)
#
# # Scale the domainnomial features
# scaler = MinMaxScaler(feature_range = (0, 1))
#
# domain_features = scaler.fit_transform(domain_features)
# domain_features_test = scaler.transform(domain_features_test)
#
# random_forest_domain = RandomForestClassifier(n_estimators = 100, random_state = 50, verbose = 1, n_jobs = -1)
#
# # Train on the training data
# random_forest_domain.fit(domain_features, train_labels)
#
# # Extract feature importances
# feature_importance_values_domain = random_forest_domain.feature_importances_
# feature_importances_domain = pd.DataFrame({'feature': domain_features_names, 'importance': feature_importance_values_domain})
#
# # Make predictions on the test data
# predictions = random_forest_domain.predict_proba(domain_features_test)[:, 1]
#
# # Make a submission dataframe
# submit = app_test[['SK_ID_CURR']]
# submit['TARGET'] = predictions
# # Save the submission dataframe
# submit.to_csv('random_forest_baseline_domain.csv', index = False)


"""作图比较特征重要性（无特征工程和领域知识新建特征工程）"""
def plot_feature_importances(df):
    """
    Args:df(dataframe),数据集包含两列，features(各个特征名称)和importance(该特征的重要性)
    Returns:
        shows a plot of the 15 most importance features
        df (dataframe): feature importances sorted by importance (highest to lowest)
        with a column for normalized importance
    """
    # Sort features according to importance
    df = df.sort_values('importance', ascending=False).reset_index()
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))), df['importance_normalized'].head(15), align='center', edgecolor='k')
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    # Plot labeling
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importances')
    plt.show()
    return df


# # Show the feature importances for the default features
# feature_importances_sorted = plot_feature_importances(feature_importances)
# feature_importances_domain_sorted = plot_feature_importances(feature_importances_domain)


"""特征工程进阶"""
"""introduction-to-manual-feature-engineering"""
"""bureau数据集分析。新建previous_loan_counts每个客户历史贷款次数，并校验其分布与target的关系"""
# Read in bureau 导入客户历史贷款记录数据
bureau = pd.read_csv('bureau.csv')

# 对client id (SK_ID_CURR)进行分组、排序，查看每个客户历史贷款SK_ID_BUREAU次数
# Groupby the client id (SK_ID_CURR), count the number of previous loans, and rename the column
previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index=False)['SK_ID_BUREAU'].count().rename(columns={'SK_ID_BUREAU': 'previous_loan_counts'})
print('previous_loan_counts:\n', previous_loan_counts.head())
"""
    SK_ID_CURR  previous_loan_counts
0      100001                     7
"""
# 新建previous_loan_counts属性，并融合在train数据集
# Join to the training dataframe
train = pd.read_csv('application_train.csv')
train = train.merge(previous_loan_counts, on='SK_ID_CURR', how='left')
# Fill the missing values with 0
train['previous_loan_counts'] = train['previous_loan_counts'].fillna(0)


# 用相关系数来衡量新建属性与target的相关性
# 用KDE还查看属性分布情况
# Plots the disribution of a variable colored by value of the target
def kde_target(var_name, df):
    # Calculate the correlation coefficient between the new variable and the target
    # 计算var_name分布与target各类的相关性，并作图
    corr = df['TARGET'].corr(df[var_name])

    # Calculate medians for repaid vs not repaid
    avg_repaid = df.loc[df['TARGET'] == 0, var_name].median()
    avg_not_repaid = df.loc[df['TARGET'] == 1, var_name].median()

    plt.figure(figsize=(12, 6))

    # Plot the distribution for target == 0 and target == 1
    sns.kdeplot(df.loc[df['TARGET'] == 0, var_name], label='TARGET == 0')
    sns.kdeplot(df.loc[df['TARGET'] == 1, var_name], label='TARGET == 1')

    # label the plot
    plt.xlabel(var_name)
    plt.ylabel('Density')
    plt.title('%s Distribution' % var_name)
    plt.legend()
    plt.show()
    # print out the correlation
    print('The correlation between %s and the TARGET is %0.4f' % (var_name, corr))
    # Print out average values
    print('Median value for loan that was not repaid = %0.4f' % avg_not_repaid)
    print('Median value for loan that was repaid =     %0.4f' % avg_repaid)


# kde_target('EXT_SOURCE_3', train)
# kde_target('previous_loan_counts', train)   # 可以看到previous_loan_counts分布与target并无关联


"""针对数值型columns属性，新建特征统计量，进行特征融合，并计算相关性cor"""
# 对client id (SK_ID_CURR)分组，统计组下各个属性所有样本的统计值
# Group by the client id (SK_ID_CURR), calculate aggregation statistics
bureau_agg = bureau.drop(columns=['SK_ID_BUREAU']).groupby('SK_ID_CURR', as_index=False).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()
print('bureau_agg:\n', bureau_agg.head())
"""
    SK_ID_CURR DAYS_CREDIT                    ...  AMT_ANNUITY                       
                   count         mean  max  ...         mean      max  min      sum
0     100001           7  -735.000000  -49  ...  3545.357143  10822.5  0.0  24817.5
1     100002           8  -874.000000 -103  ...     0.000000      0.0  0.0      0.0
"""
# 二级索引(各种统计量)与一级索引(各属性)，合并为一层属性
# List of column names
columns = ['SK_ID_CURR']
# Iterate through the variables names
# bureau_agg.columns.levels[0]即数据集bureau_agg的columns列表，不含统计名称columns
# bureau_agg.columns.levels[1]即['count', 'mean', 'max', 'min', 'sum', '']
for var in bureau_agg.columns.levels[0]:
    # Skip the id name
    if var != 'SK_ID_CURR':
        # Iterate through the stat names
        for stat in bureau_agg.columns.levels[1][:-1]:
            # Make a new column name for the variable and stat
            columns.append('bureau_%s_%s' % (var, stat))

# Assign the list of columns names as the dataframe column names
bureau_agg.columns = columns    # 替换columns名称
print('bureau_agg:\n', bureau_agg.head())    # 数据与之前的bureau_agg一样，只是一二级属性名称统一，替换为columns

# Merge with the training data
train = train.merge(bureau_agg, on='SK_ID_CURR', how='left')

"""计算columns元素与target的相关性，并保存到new_corrs"""
# List of new correlations
new_corrs = []
# Iterate through the columns
for col in columns:
    # Calculate correlation with the target
    corr = train['TARGET'].corr(train[col])
    # Append the list as a tuple
    new_corrs.append((col, corr))

# Sort the correlations by the absolute value
# Make sure to reverse to put the largest values at the front of list
new_corrs = sorted(new_corrs, key=lambda x: abs(x[1]), reverse=True)
print("前15个相关性较大的统计型属性：\n", new_corrs[:15])        # 也没啥用，相关性值都比较小
# kde_target('bureau_DAYS_CREDIT_mean', train)


def agg_numeric(df, group_var, df_name):
    """
    对前面新建特征统计量，并生成数据集bureau_agg的工作进行函数封装
    输入：原始数据集df，分组的属性group_var，新属性的一级名称df_name
    """
    # step 1：删除原始数据集df的一些ID类属性
    for col in df:
        if col != group_var and 'SK_ID' in col:
            df = df.drop(columns=col)
    # step 2：删除数据集df非数值类型的属性
    group_ids = df[group_var]
    numeric_df = df.select_dtypes('number')
    numeric_df[group_var] = group_ids
    # step 3：聚合agg，初步生成数据集numeric_df各属性的统计量
    agg = numeric_df.groupby(group_var).agg(['count', 'mean', 'max', 'min', 'sum']).reset_index()
    # step 4： 创建 new column names
    columns = [group_var]
    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        # Skip the grouping variable
        if var != group_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))
    # step 5：替换columns名称
    agg.columns = columns
    return agg


def target_corrs(df):
    """
    对前面计算columns元素与target的相关性，并保存到new_corrs的工作进行函数封装
    输入：数据集df
    """
    # List of correlations
    corrs = []
    # Iterate through the columns
    for col in df.columns:
        print(col)
        # Skip the target column
        if col != 'TARGET':
            # Calculate correlation with the target
            corr = df['TARGET'].corr(df[col])

            # Append the list as a tuple
            corrs.append((col, corr))

    # Sort by absolute magnitude of correlations
    corrs = sorted(corrs, key=lambda x: abs(x[1]), reverse=True)
    return corrs


# bureau_agg_new = agg_numeric(bureau.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bureau')
# new_corrs = target_corrs(train)


"""
针对bureau数据集中的离散型数据类型，使用one-hot encoding，根据客户聚合并计算统计量['sum', 'mean']
"""
# 对离散数据使用one-hot encoding，生成categorical数据集
categorical = pd.get_dummies(bureau.select_dtypes('object'))
categorical['SK_ID_CURR'] = bureau['SK_ID_CURR']
print(categorical.head())
# 对categorical数据集依据客户SK_ID_CURR进行分组，计算统计量['sum', 'mean']
categorical_grouped = categorical.groupby('SK_ID_CURR').agg(['sum', 'mean'])
print(categorical_grouped.head())

"""agg_numeric函数功能，将聚合得到的数据集categorical_grouped属性，二级索引(各种统计量)与一级索引(各属性)，合并为一层属性"""
group_var = 'SK_ID_CURR'
# Need to create new column names
columns = []
# Iterate through the variables names
for var in categorical_grouped.columns.levels[0]:
    # Skip the grouping variable
    if var != group_var:
        # Iterate through the stat names
        for stat in ['count', 'count_norm']:
            # Make a new column name for the variable and stat
            columns.append('%s_%s' % (var, stat))
#  Rename the columns
categorical_grouped.columns = columns
# Merge with the training data
train = train.merge(categorical_grouped, left_on = 'SK_ID_CURR', right_index = True, how = 'left')


def count_categorical(df, group_var, df_name):
    """
    agg_numeric函数功能类似
    将聚合得到的数据集categorical_grouped属性，二级索引(各种统计量)与一级索引(各属性)，合并为一层属性
    """
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('object'))
    # Make sure to put the identifying id on the column
    categorical[group_var] = df[group_var]
    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(group_var).agg(['sum', 'mean'])
    column_names = []
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['count', 'count_norm']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    categorical.columns = column_names
    return categorical


# bureau_counts = count_categorical(bureau, group_var='SK_ID_CURR', df_name='bureau')


"""
bureau_balance数据集分析
属性SK_ID_BUREAU表示每一笔贷款，在表中对应多行月记录；先根据SK_ID_BUREAU对每一笔贷款进行聚合；再根据SK_ID_CURR对每个客户进行聚合
"""
# Read in bureau balance
bureau_balance = pd.read_csv('bureau_balance.csv')
print(bureau_balance.head())

"""对STATUS属性(object类型)进行分析
Counts of each type of status for each previous loan 统计每一笔贷款SK_ID_BUREAU的各个STATUS的['count', 'count_norm']
columns属性名包含['bureau_balance_STATUS_0_count', 'bureau_balance_STATUS_0_count_norm','bureau_balance_STATUS_1_count'...]
"""
bureau_balance_counts = count_categorical(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
print(bureau_balance_counts.head())

"""对MONTHS_BALANCE属性(numeric类型)进行分析
Calculate value count statistics for each `SK_ID_CURR` 统计每一笔贷款SK_ID_BUREAU的['count', 'mean', 'max', 'min', 'sum']
"""
bureau_balance_agg = agg_numeric(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
print(bureau_balance_agg.head())

"""将bureau_balance数据集得到的bureau_balance_counts、bureau_balance_agg根据每一笔贷款记录SK_ID_BUREAU进行merge，每个样本行代表一笔贷款记录"""
bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index=True, left_on='SK_ID_BUREAU', how='outer')

# Merge to include the SK_ID_CURR 与bureau数据集进行merge
bureau_by_loan = bureau_by_loan.merge(bureau[['SK_ID_BUREAU', 'SK_ID_CURR']], on='SK_ID_BUREAU', how='left')
print(bureau_by_loan.head())

"""根据每个客户SK_ID_CURR进一步聚合"""
bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'client')
print(bureau_balance_by_client.head())


"""
python垃圾回收机制Garbage collection介绍
python采用的是引用计数机制为主，标记-清除和分代收集两种机制为辅的策略
1.引用计数机制
python里每一个东西都是对象，它们的核心就是一个结构体：PyObject
结构体中的ob_refcnt就是做为引用计数。当一个对象被其他对象引用时，它的ob_refcnt就会增加1；当ob_refcnt为0时，该对象生命就结束了
优缺点：
实时性：一旦没有引用，内存就直接释放
维护引用计数ob_refcnt消耗资源
不能解决循环引用的问题
更多内容请参考：https://www.jianshu.com/p/1e375fb40506
"""
import gc


gc.enable()
del train, bureau, bureau_balance, bureau_agg, bureau_balance_agg, bureau_balance_counts, bureau_by_loan, bureau_balance_by_client
print('gc collect Num:', gc.collect())     # 处理这些循环引用一共释放掉的对象个数


"""读取数据文件"""
# Read in new copies of all the dataframes
train = pd.read_csv('application_train.csv')
bureau = pd.read_csv('bureau.csv')
bureau_balance = pd.read_csv('bureau_balance.csv')

"""对bureau、bureau_balance数据处理，生成四个数据集"""
bureau_counts = count_categorical(bureau, group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_agg = agg_numeric(bureau.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'bureau')
bureau_balance_counts = count_categorical(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
bureau_balance_agg = agg_numeric(bureau_balance, group_var = 'SK_ID_BUREAU', df_name = 'bureau_balance')
"""bureau_balance处理，并根据客户SK_ID_CURR分组，生成bureau_balance_by_client"""
bureau_by_loan = bureau_balance_agg.merge(bureau_balance_counts, right_index = True, left_on = 'SK_ID_BUREAU', how = 'outer')
bureau_by_loan = bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(bureau_by_loan, on = 'SK_ID_BUREAU', how = 'left')
bureau_balance_by_client = agg_numeric(bureau_by_loan.drop(columns = ['SK_ID_BUREAU']), group_var = 'SK_ID_CURR', df_name = 'client')
# 原始数据train的columns， 122
original_features = list(train.columns)
print('Original Number of Features: ', len(original_features))
"""将train与bureau_counts、bureau_agg、bureau_balance_by_client进行merge"""
train = train.merge(bureau_counts, on = 'SK_ID_CURR', how = 'left')
train = train.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
train = train.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')
# merge后的数据train的columns， 333
new_features = list(train.columns)
print('Number of features using previous loans from other institutions data: ', len(new_features))


"""查看缺失值的百分比，变量与目标的相关性，以及变量与其他变量的相关性"""
"""计算train的缺失值情况"""
def missing_values_table(df):
    """
    Function to calculate missing values by column
    :param df:
    :return: 各个属性的数量'Missing Values'和比例'% of Total Values'
    """
    # Total missing values
    mis_val = df.isnull().sum()        # 各个属性缺失值数量
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)          # 各个属性缺失值比例
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                              "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    # Return the dataframe with missing information
    return mis_val_table_ren_columns


missing_train = missing_values_table(train)
print('missing_train:\n', missing_train.head(10))

# 同样的方法处理test数据集
test = pd.read_csv('application_test.csv')
test = test.merge(bureau_counts, on = 'SK_ID_CURR', how = 'left')
test = test.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
test = test.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')
print('Shape of Testing Data: ', test.shape)
train_labels = train['TARGET']
# Align the dataframes, this will remove the 'TARGET' column
train, test = train.align(test, join = 'inner', axis = 1)
train['TARGET'] = train_labels

# 删除train中缺失值比例在90%以上的属性
missing_columns = list(missing_train.index[missing_train['% of Total Values'] > 90])
print('There are %d columns with more than 90%% missing in either the training or testing data.' % len(missing_columns))
train = train.drop(columns=missing_columns)

"""计算各个属性与target的相关性"""
# # Calculate all correlations in dataframe
# corrs = train.corr()
# corrs = corrs.sort_values('TARGET', ascending = False)
# # Ten most positive correlations
# print(pd.DataFrame(corrs['TARGET'].head(10)))
# # Ten most negative correlations
# print(pd.DataFrame(corrs['TARGET'].dropna().tail(10)))
# # 利用kde_target函数来作参数属性分布与target==0、target==1的关系，并作图
# kde_target(var_name='bureau_CREDIT_ACTIVE_Active_count_norm', df=train)

"""计算各个属性之间的相关性，并将相关性超过0.8的属性删除其一"""
# # Set the threshold  相关性阈值
# threshold = 0.8
# # Empty dictionary to hold correlated variables
# above_threshold_vars = {}
# # 对每个属性col，查找所有与其相关性大于0.8的其他属性(列表)，形成一个键值对
# for col in corrs:
#     above_threshold_vars[col] = list(corrs.index[corrs[col] > threshold])
# # Track columns to remove and columns already examined
# cols_to_remove = []
# cols_seen = []
# cols_to_remove_pair = []
#
# # Iterate through columns and correlated columns
# for key, value in above_threshold_vars.items():
#     # Keep track of columns already examined
#     cols_seen.append(key)
#     for x in value:
#         if x == key:
#             pass
#         else:
#             # Only want to remove one in a pair
#             if x not in cols_seen:
#                 cols_to_remove.append(x)
#                 cols_to_remove_pair.append(key)
#
# cols_to_remove = list(set(cols_to_remove))
# print('Number of columns to remove: ', len(cols_to_remove))
#
# train_corrs_removed = train.drop(columns = cols_to_remove)
# test_corrs_removed = test.drop(columns = cols_to_remove)
#
# print('Training Corrs Removed Shape: ', train_corrs_removed.shape)
# print('Testing Corrs Removed Shape: ', test_corrs_removed.shape)


"""建模lightgbm """
def model(features, test_features, encoding='ohe', n_folds=5):
    """Train and test a light gradient boosting model using cross validation.
    Parameters
        features (pd.DataFrame):
            dataframe of training features to use for training a model. Must include the TARGET column.
        test_features (pd.DataFrame):
            dataframe of testing features to use for making predictions with the model.
        encoding (str, default = 'ohe'):
            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
            n_folds (int, default = 5): number of folds to use for cross validation
    Return
        submission (pd.DataFrame):
            dataframe with `SK_ID_CURR` and `TARGET` probabilities
            predicted by the model.
        feature_importances (pd.DataFrame):
            dataframe with the feature importances from the model.
        valid_metrics (pd.DataFrame):
            dataframe with training and validation metrics (ROC AUC) for each fold and overall.
    """
    # Extract the ids
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    # Extract the labels for training
    labels = features['TARGET']
    # Remove the ids and target
    features = features.drop(columns=['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns=['SK_ID_CURR'])
    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join='inner', axis=1)
        # No categorical indices to record
        cat_indices = 'auto'
    # Integer label encoding
    elif encoding == 'le':
        # Create a label encoder
        label_encoder = LabelEncoder()
        # List for storing categorical indices
        cat_indices = []
        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))
                # Record the categorical indices
                cat_indices.append(i)
    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    # Extract feature names
    feature_names = list(features.columns)
    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)
    # Create the kfold object
    k_fold = KFold(n_splits=n_folds, shuffle=False, random_state=50)
    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))
    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])
    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []
    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        # Create the model
        model = lgb.LGBMClassifier(n_estimators=10000, objective='binary',
                                   class_weight='balanced', learning_rate=0.05,
                                   reg_alpha=0.1, reg_lambda=0.1,
                                   subsample=0.8, n_jobs=-1, random_state=50)
        # Train the model
        model.fit(train_features, train_labels, eval_metric='auc',
                  eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names=['valid', 'train'], categorical_feature=cat_indices,
                  early_stopping_rounds=100, verbose=200)
        # Record the best iteration
        best_iteration = model.best_iteration_
        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration=best_iteration)[:, 1] / k_fold.n_splits
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration=best_iteration)[:, 1]
        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)
    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})
    return submission, feature_importances, metrics


def plot_feature_importances(df):
    """
    Plot importances returned by a model. This can work with any measure of
    feature importance provided that higher importance is better.
    Args:
        df (dataframe): feature importances. Must have the features in a column
        called `features` and the importances in a column called `importance
    Returns:
        shows a plot of the 15 most importance features

        df (dataframe): feature importances sorted by importance (highest to lowest)
        with a column for normalized importance
        """
    # Sort features according to importance
    df = df.sort_values('importance', ascending=False).reset_index()
    # Normalize the feature importances to add up to one
    df['importance_normalized'] = df['importance'] / df['importance'].sum()
    # Make a horizontal bar chart of feature importances
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()
    # Need to reverse the index to plot most important on top
    ax.barh(list(reversed(list(df.index[:15]))),
            df['importance_normalized'].head(15),
            align='center', edgecolor='k')
    # Set the yticks and labels
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df['feature'].head(15))
    # Plot labeling
    plt.xlabel('Normalized Importance')
    plt.title('Feature Importances')
    plt.show()
    return df


"""根据模型进行训练，可以利用原始train数据集、添加历史贷款记录的数据集、删除高度相关性之后的数据集用于训练，对比效果"""
# submission_raw, fi_raw, metrics_raw = model(train, test)
# fi_raw_sorted = plot_feature_importances(fi_raw)
# submission_raw.to_csv('test_one.csv', index=False)


"""
特征工程进阶，利用previous_application, POS_CASH_balance, installments_payments, and credit_card_balance四个文件进一步提取信息
"""
import sys


def return_size(df):
    """Return size of dataframe in gigabytes"""
    return round(sys.getsizeof(df) / 1e9, 2)


def convert_types(df, print_info=False):
    """对df数据集数据类型进行转换"""
    original_memory = df.memory_usage().sum()
    # Iterate through each column
    for c in df:
        # Convert ids and booleans to integers
        if ('SK_ID' in c):
            df[c] = df[c].fillna(0).astype(np.int32)
        # Convert objects to category
        elif (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype('category')
        # Booleans mapped to integers
        elif list(df[c].unique()) == [1, 0]:
            df[c] = df[c].astype(bool)
        # Float64 to float32
        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)
        # Int64 to int32
        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)
    new_memory = df.memory_usage().sum()
    if print_info:
        print(f'Original Memory Usage: {round(original_memory / 1e9, 2)} gb.')
        print(f'New Memory Usage: {round(new_memory / 1e9, 2)} gb.')
    return df


"""
对previous_application文件进行特征工程处理
读取previous_application文件，根据客户SK_ID_CURR进行聚合，生成previous_agg、previous_counts
"""
# previous = pd.read_csv('previous_application.csv')
# previous = convert_types(previous, print_info=True)
#
# # Calculate aggregate statistics for each numeric column
# previous_agg = agg_numeric(previous, 'SK_ID_CURR', 'previous')
# print('Previous aggregation shape: ', previous_agg.shape)
#
# # Calculate value counts for each categorical column
# previous_counts = count_categorical(previous, 'SK_ID_CURR', 'previous')
# print('Previous counts shape: ', previous_counts.shape)
#
# """读取主数据集application_train，并将数据previous_agg、previous_counts进行merge；最后进行内存释放"""
# train = pd.read_csv('application_train.csv')
# train = convert_types(train)
# test = pd.read_csv('application_test.csv')
# test = convert_types(test)
# # Merge in the previous information
# train = train.merge(previous_counts, on ='SK_ID_CURR', how = 'left')
# train = train.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')
# test = test.merge(previous_counts, on ='SK_ID_CURR', how = 'left')
# test = test.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')
# # Remove variables to free memory
# gc.enable()
# del previous, previous_agg, previous_counts
# print(gc.collect())


"""
定义两个函数，进行缺失值处理
missing_values_table函数统计各属性缺失值数量及占比
remove_missing_columns函数实现对df缺失值占比大于threshold的属性删除
"""
# Function to calculate missing values by column# Funct
def missing_values_table(df, print_info=False):
    # Total missing values
    mis_val = df.isnull().sum()
    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    if print_info:
        # Print some summary information
        print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                                  "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
    # Return the dataframe with missing information
    return mis_val_table_ren_columns


def remove_missing_columns(train, test, threshold=90):
    # Calculate missing stats for train and test (remember to calculate a percent!)
    train_miss = pd.DataFrame(train.isnull().sum())
    train_miss['percent'] = 100 * train_miss[0] / len(train)

    test_miss = pd.DataFrame(test.isnull().sum())
    test_miss['percent'] = 100 * test_miss[0] / len(test)
    # list of missing columns for train and test
    missing_train_columns = list(train_miss.index[train_miss['percent'] > threshold])
    missing_test_columns = list(test_miss.index[test_miss['percent'] > threshold])

    # Combine the two lists together
    missing_columns = list(set(missing_train_columns + missing_test_columns))
    # Print information
    print('There are %d columns with greater than %d%% missing values.' % (len(missing_columns), threshold))
    # Drop the missing columns and return
    train = train.drop(columns=missing_columns)
    test = test.drop(columns=missing_columns)
    return train, test


# train, test = remove_missing_columns(train, test)


"""
POS_CASH_balance数据集处理
"""
def aggregate_client(df, group_vars, df_names):
    """Aggregate a dataframe with data at the loan level at the client level
    Args:
        df (dataframe): data at the loan level
        group_vars (list of two strings): grouping variables for the loan and then the client (example ['SK_ID_PREV', 'SK_ID_CURR'])
        names (list of two strings): names to call the resulting columns
        (example ['cash', 'client'])
    Returns:
        df_client (dataframe): aggregated numeric stats at the client level.
        Each client will have a single row with all the numeric data aggregated
    """
    # Aggregate the numeric columns
    df_agg = agg_numeric(df, parent_var=group_vars[0], df_name=df_names[0])
    # If there are categorical variables
    if any(df.dtypes == 'category'):
        # Count the categorical columns
        df_counts = count_categorical(df, parent_var=group_vars[0], df_name=df_names[0])
        # Merge the numeric and categorical
        df_by_loan = df_counts.merge(df_agg, on=group_vars[0], how='outer')
        gc.enable()
        del df_agg, df_counts
        gc.collect()
        # Merge to get the client id in dataframe
        df_by_loan = df_by_loan.merge(df[[group_vars[0], group_vars[1]]], on=group_vars[0], how='left')
        # Remove the loan id
        df_by_loan = df_by_loan.drop(columns=[group_vars[0]])
        # Aggregate numeric stats by column
        df_by_client = agg_numeric(df_by_loan, parent_var=group_vars[1], df_name=df_names[1])
    # No categorical variables
    else:
        # Merge to get the client id in dataframe
        df_by_loan = df_agg.merge(df[[group_vars[0], group_vars[1]]], on=group_vars[0], how='left')
        gc.enable()
        del df_agg
        gc.collect()
        # Remove the loan id
        df_by_loan = df_by_loan.drop(columns=[group_vars[0]])
        # Aggregate numeric stats by column
        df_by_client = agg_numeric(df_by_loan, parent_var=group_vars[1], df_name=df_names[1])
    # Memory management
    gc.enable()
    del df, df_by_loan
    gc.collect()
    return df_by_client


"""加载POS_CASH_balance数据集，并根据SK_ID_CURR聚合"""
# cash = pd.read_csv('POS_CASH_balance.csv')
# cash = convert_types(cash, print_info=True)
# cash_by_client = aggregate_client(cash, group_vars = ['SK_ID_PREV', 'SK_ID_CURR'], df_names = ['cash', 'client'])
# """将cash_by_client与主数据集进行merge"""
# print('Cash by Client Shape: ', cash_by_client.shape)
# train = train.merge(cash_by_client, on = 'SK_ID_CURR', how = 'left')
# test = test.merge(cash_by_client, on = 'SK_ID_CURR', how = 'left')
# """清除缓存"""
# gc.enable()
# del cash, cash_by_client
# gc.collect()
# """删除缺失值较多的属性"""
# train, test = remove_missing_columns(train, test)


"""
credit_card_balance数据集处理
"""
"""加载credit_card_balance数据集，并根据SK_ID_CURR聚合"""
# credit = pd.read_csv('credit_card_balance.csv')
# credit = convert_types(credit, print_info = True)
# credit_by_client = aggregate_client(credit, group_vars = ['SK_ID_PREV', 'SK_ID_CURR'], df_names = ['credit', 'client'])
# """将credit_by_client与主数据集进行merge"""
# print('Credit by client shape: ', credit_by_client.shape)
# train = train.merge(credit_by_client, on = 'SK_ID_CURR', how = 'left')
# test = test.merge(credit_by_client, on = 'SK_ID_CURR', how = 'left')
# """清除缓存"""
# gc.enable()
# del credit, credit_by_client
# gc.collect()
# """删除缺失值较多的属性"""
# train, test = remove_missing_columns(train, test)


"""
installments_payments数据集处理
"""
"""加载installments_payments数据集，并根据SK_ID_CURR聚合"""
# installments = pd.read_csv('installments_payments.csv')
# installments = convert_types(installments, print_info = True)
# installments_by_client = aggregate_client(installments, group_vars = ['SK_ID_PREV', 'SK_ID_CURR'], df_names = ['installments', 'client'])
# """将installments_by_client与主数据集进行merge"""
# print('Installments by client shape: ', installments_by_client.shape)
# train = train.merge(installments_by_client, on = 'SK_ID_CURR', how = 'left')
# test = test.merge(installments_by_client, on = 'SK_ID_CURR', how = 'left')
# """清除缓存"""
# gc.enable()
# del installments, installments_by_client
# gc.collect()
# """删除缺失值较多的属性"""
# train, test = remove_missing_columns(train, test)


"""打印对所有表格进行特征收集之后的train.shape"""
# print('Final Training Shape: ', train.shape)
# print('Final Testing Shape: ', test.shape)

"""根据模型进行训练"""
# submission_raw, fi_raw, metrics_raw = model(train, test)
# fi_raw_sorted = plot_feature_importances(fi_raw)
# submission_raw.to_csv('test_one.csv', index=False)


"""
特征工程基本方法
自动特征工程工具包 featuretools模块 的基本使用
"""
import featuretools as ft


# Read in the datasets and limit to the first 1000 rows (sorted by SK_ID_CURR)
app_train = pd.read_csv('application_train.csv').sort_values('SK_ID_CURR').reset_index(drop = True).loc[:1000, :]
app_test = pd.read_csv('application_test.csv').sort_values('SK_ID_CURR').reset_index(drop = True).loc[:1000, :]
bureau = pd.read_csv('bureau.csv').sort_values(['SK_ID_CURR', 'SK_ID_BUREAU']).reset_index(drop = True).loc[:1000, :]
bureau_balance = pd.read_csv('bureau_balance.csv').sort_values('SK_ID_BUREAU').reset_index(drop = True).loc[:1000, :]
cash = pd.read_csv('POS_CASH_balance.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:1000, :]
credit = pd.read_csv('credit_card_balance.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:1000, :]
previous = pd.read_csv('previous_application.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:1000, :]
installments = pd.read_csv('installments_payments.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:1000, :]

# 创建新属性set，区分样本训练集与测试集
app_train['set'] = 'train'
app_test['set'] = 'test'
app_test["TARGET"] = np.nan

# 合并
app = app_train.append(app_test, ignore_index = True)


"""创建一个空的实体集es，一个实体集可以包含多个实体，一个实体就是一张表"""
es = ft.EntitySet(id = 'clients')


"""
向实体es中添加实体entity
定义实体entity，如果实体有索引index，则需要传入index；如果实体没有索引，则make_index = True
featuretools会自动判断属性类型，也可以手动定义
"""
# Entities with a unique index
es = es.entity_from_dataframe(entity_id='app', dataframe=app, index='SK_ID_CURR')
es = es.entity_from_dataframe(entity_id='bureau', dataframe=bureau, index='SK_ID_BUREAU')
es = es.entity_from_dataframe(entity_id='previous', dataframe=previous, index='SK_ID_PREV')
# Entities that do not have a unique index
es = es.entity_from_dataframe(entity_id='bureau_balance', dataframe=bureau_balance, make_index=True, index='bureaubalance_index')
es = es.entity_from_dataframe(entity_id='cash', dataframe=cash, make_index=True, index='cash_index')
es = es.entity_from_dataframe(entity_id='installments', dataframe=installments, make_index=True, index = 'installments_index')
es = es.entity_from_dataframe(entity_id='credit', dataframe=credit, make_index=True, index='credit_index')


"""
对实体集es中的各实体建立关联
app：每行代表一个客户，包含每个客户样本的一些基本信息，通过SK_ID_CURR标识
bureau：每行代表一笔贷款，包含每笔贷款的基本信息，通过SK_BUREAU_ID标识(每个客户SK_ID_CURR可以有多笔贷款SK_BUREAU_ID)
bureau_balance：每行代表一笔贷款单月还款信息，包含每笔贷款的分月还款情况(每笔贷款SK_BUREAU_ID可以有多行分月还款)
previous：每行代表一笔历史贷款，包含每笔贷款基本信息，通过SK_ID_PREV标识(每个客户SK_ID_CURR可以有多笔历史贷款SK_ID_PREV)
cash：每行代表一笔历史贷款单月还款信息，包含每笔贷款的分月还款情况(每笔历史贷款SK_ID_PREV可以有多行分月还款，每个客户SK_ID_CURR可以有多笔历史贷款SK_ID_PREV)
credit： (类似cash)每行代表一张信用卡单月还款信息，包含每张信用卡分月还款情况(每笔历史贷款SK_ID_PREV可以有多行分月还款，每个客户SK_ID_CURR可以有多笔历史贷款SK_ID_PREV)
installments：(类似cash)每行代表一次还款信息，包含每笔贷款的多次还款情况(每笔历史贷款SK_ID_PREV可以有多次还款信息，每个客户SK_ID_CURR可以有多笔历史贷款SK_ID_PREV)
"""
# Relationship between app and bureau
r_app_bureau = ft.Relationship(es['app']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'])
# Relationship between bureau and bureau balance
r_bureau_balance = ft.Relationship(es['bureau']['SK_ID_BUREAU'], es['bureau_balance']['SK_ID_BUREAU'])

# Relationship between current app and previous apps
r_app_previous = ft.Relationship(es['app']['SK_ID_CURR'], es['previous']['SK_ID_CURR'])
# Relationships between previous apps and cash, installments, and credit
r_previous_cash = ft.Relationship(es['previous']['SK_ID_PREV'], es['cash']['SK_ID_PREV'])
r_previous_installments = ft.Relationship(es['previous']['SK_ID_PREV'], es['installments']['SK_ID_PREV'])
r_previous_credit = ft.Relationship(es['previous']['SK_ID_PREV'], es['credit']['SK_ID_PREV'])

# Add in the defined relationships
es = es.add_relationships([r_app_bureau, r_bureau_balance, r_app_previous, r_previous_cash, r_previous_installments, r_previous_credit])


"""
primitives常见的两种基本特征创建方法
    name	    type	    description
0	any	        aggregation	Test if any value is 'True'.
1	num_unique	aggregation	Returns the number of unique categorical variables.
2	last	    aggregation	Returns the last value.
3	max	        aggregation	Finds the maximum non-null value of a numeric feature.
4	num_true	aggregation	Finds the number of 'True' values in a boolean.
5	std	        aggregation	Finds the standard deviation of a numeric feature ignoring null values.
6	min	        aggregation	Finds the minimum non-null value of a numeric feature.
7	skew	    aggregation	Computes the skewness of a data set.
8	sum	        aggregation	Counts the number of elements of a numeric or boolean feature.
9	trend	    aggregation	Calculates the slope of the linear trend of variable overtime.

    name	    type	    description
19	absolute	transform	Absolute value of base feature.
20	year	    transform	Transform a Datetime feature into the year.
21	compare	    transform	For each value, determine if it is equal to another value.
22	month	    transform	Transform a Datetime feature into the month.
23	week	    transform	Transform a Datetime feature into the week.
24	days_since	transform	For each value of the base feature, compute the number of days between it
25	hours	    transform	Transform a Timedelta feature into the number of hours.
26	minute	    transform	Transform a Datetime feature into the minute.
27	seconds 	transform	Transform a Timedelta feature into the number of seconds.
28	cum_min	    transform	Calculates the min of previous values of an instance for each value in a time-dependent entity
"""
# List the primitives in a dataframe
primitives = ft.list_primitives()
pd.options.display.max_colwidth = 100
primitives[primitives['type'] == 'aggregation'].head(10)
primitives[primitives['type'] == 'transform'].head(10)


"""
深度特征合成DFS
"""
# Default primitives from featuretools
default_agg_primitives =  ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]
default_trans_primitives =  ["day", "year", "month", "weekday", "haversine", "numwords", "characters"]

# # DFS with specified primitives
# feature_names = ft.dfs(entityset = es, target_entity = 'app',
#                        trans_primitives = default_trans_primitives,
#                        agg_primitives=default_agg_primitives,
#                        max_depth = 2, features_only=True)
#
# print('%d Total Features' % len(feature_names))


"""
接下来省略的一些步骤：
查看各属性之间的相关性
查看各属性与target的相关性
查看各属性按target分类的分布kde图
输出数据集进行训练
"""


"""
featuretools进阶
"""
"""对于部分属性，属性值为整数，但属性值去重后只有1-2个，此类属性为布尔类型属性"""
app_types = {}
# Iterate through the columns and record the Boolean columns
for col in app_train:
    # If column is a number with only two values, encode it as a Boolean
    if (app_train[col].dtype != 'object') and (len(app_train[col].unique()) <= 2):
        app_types[col] = ft.variable_types.Boolean
print('Number of boolean variables: ', len(app_types))
# Record ordinal variables
app_types['REGION_RATING_CLIENT'] = ft.variable_types.Ordinal
app_types['REGION_RATING_CLIENT_W_CITY'] = ft.variable_types.Ordinal
app_test_types = app_types.copy()
del app_test_types['TARGET']
# Record boolean variables in the previous data
previous_types= {'NFLAG_LAST_APPL_IN_DAY': ft.variable_types.Boolean,
                 'NFLAG_INSURED_ON_APPROVAL': ft.variable_types.Boolean}


"""对于时间类型属性"""
import re


"""对所有df，将365243替换为nan"""
def replace_day_outliers(df):
    """Replace 365243 with np.nan in any columns with DAYS"""
    for col in df.columns:
        if "DAYS" in col:
            df[col] = df[col].replace({365243: np.nan})
    return df

app_train = replace_day_outliers(app_train)
app_test = replace_day_outliers(app_test)
bureau = replace_day_outliers(bureau)
bureau_balance = replace_day_outliers(bureau_balance)
credit = replace_day_outliers(credit)
cash = replace_day_outliers(cash)
previous = replace_day_outliers(previous)
installments = replace_day_outliers(installments)


"""将时间类型属性转换为timedelta时间段类型，根据基准日期start_date更新"""
# Establish a starting date for all applications at Home Credit   创建一个基准日期
start_date = pd.Timestamp("2016-01-01")

"""将时间类型属性转换为timedelta时间段类型，单位为天Day"""
for col in ['DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', 'DAYS_CREDIT_UPDATE']:
    bureau[col] = pd.to_timedelta(bureau[col], 'D')   # 将bureau[col]转换为timedelta时间段类型，单位为天Day

print(bureau[['DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', 'DAYS_CREDIT_UPDATE']].head())

# Create the date columns
bureau['bureau_credit_application_date'] = start_date + bureau['DAYS_CREDIT']
bureau['bureau_credit_end_date'] = start_date + bureau['DAYS_CREDIT_ENDDATE']
bureau['bureau_credit_close_date'] = start_date + bureau['DAYS_ENDDATE_FACT']
bureau['bureau_credit_update_date'] = start_date + bureau['DAYS_CREDIT_UPDATE']


"""
根据bureau_credit_end_date、bureau_credit_application_date计算各样本贷款时长，并作图
可以发现有一波数据，时长明显异常
"""
import matplotlib.pyplot as plt
import seaborn as sns
# Set up default plot styles
plt.rcParams['font.size'] = 26
plt.style.use('fivethirtyeight')

# Drop the time offset columns
bureau = bureau.drop(columns = ['DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT', 'DAYS_CREDIT_UPDATE'])

plt.figure(figsize = (10, 8))
sns.distplot((bureau['bureau_credit_end_date'] - bureau['bureau_credit_application_date']).dropna().dt.days)
plt.xlabel('Length of Loan (Days)', size = 24); plt.ylabel('Density', size = 24); plt.title('Loan Length', size = 30)
# plt.show()


"""
将bureau_balance数据集的MONTHS_BALANCE属性进行转换，对'SK_ID_BUREAU' == 5001709的贷款样本的分月还款作图
"""
bureau_balance['MONTHS_BALANCE'] = pd.to_timedelta(bureau_balance['MONTHS_BALANCE'], 'M')

# Make a date column
bureau_balance['bureau_balance_date'] = start_date + bureau_balance['MONTHS_BALANCE']
bureau_balance = bureau_balance.drop(columns = ['MONTHS_BALANCE'])

# Select one loan and plot
example_credit = bureau_balance[bureau_balance['SK_ID_BUREAU'] == 5001709]
plt.plot(example_credit['bureau_balance_date'], example_credit['STATUS'], 'ro')
plt.title('Loan 5001709 over Time')
plt.xlabel('Date')
plt.ylabel('Status')
# plt.show()


"""
特征选择方法：
1.剔除相关性比较大的属性之一
2.删除空值率较高的属性
3.选择特征重要性排序较高的属性
4.运用降维算法，例如PCA
"""
from sklearn.decomposition import PCA
from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline


# Make sure to drop the ids and target
train = train.drop(columns = ['SK_ID_CURR', 'TARGET'])
test = test.drop(columns = ['SK_ID_CURR'])

# Make a pipeline with imputation and pca
pipeline = Pipeline(steps = [('imputer', Imputer(strategy = 'median')), ('pca', PCA())])
# Fit and transform on the training data
train_pca = pipeline.fit_transform(train)
# transform the testing data
test_pca = pipeline.transform(test)
# Extract the pca object
pca = pipeline.named_steps['pca']
# Plot the cumulative variance explained
plt.figure(figsize = (10, 8))
plt.plot(list(range(train.shape[1])), np.cumsum(pca.explained_variance_ratio_), 'r-')
plt.xlabel('Number of PC')
plt.ylabel('Cumulative Variance Explained')
plt.title('Cumulative Variance Explained with PCA')
# plt.show()


"""
hyperopt模块是一个调参模块，主要包含以下步骤：
1.定义目标函数
2.定义域空间
3.参数搜索（主要算法有TPE(Tree Parzen Estimation)算法、随机搜索算法）
4.结果保存

网格调参与随机搜索调参
"""
import pandas as pd
import numpy as np
# Modeling
import lightgbm as lgb
# Splitting data
from sklearn.model_selection import train_test_split


N_FOLDS = 5
MAX_EVALS = 5


"""获取训练集"""
features = pd.read_csv('application_train.csv')
# Sample 16000 rows (10000 for training, 6000 for testing)
features = features.sample(n = 16000, random_state = 42)
# Only numeric features
features = features.select_dtypes('number')
# Extract the labels
labels = np.array(features['TARGET'].astype(np.int32)).reshape((-1, ))
features = features.drop(columns = ['TARGET', 'SK_ID_CURR'])
# Split into training and testing data
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 6000, random_state = 50)

"""Cross Validation and Early Stopping"""
# Create a training and testing dataset
train_set = lgb.Dataset(data = train_features, label = train_labels)
test_set = lgb.Dataset(data = test_features, label = test_labels)

"""base model"""
# Get default hyperparameters
model = lgb.LGBMClassifier()
default_params = model.get_params()

# Remove the number of estimators because we set this to 10000 in the cv call
del default_params['n_estimators']

# Cross validation with early stopping
cv_results = lgb.cv(default_params, train_set, num_boost_round = 10000, early_stopping_rounds = 100, metrics = 'auc', nfold = N_FOLDS, seed = 42)

print('The maximum validation ROC AUC was: {:.5f} with a standard deviation of {:.5f}.'.format(cv_results['auc-mean'][-1], cv_results['auc-stdv'][-1]))
print('The optimal number of boosting rounds (estimators) was {}.'.format(len(cv_results['auc-mean'])))


"""base model 测试"""
from sklearn.metrics import roc_auc_score


# Optimal number of esimators found in cv
model.n_estimators = len(cv_results['auc-mean'])

# Train and make predicions with model
model.fit(train_features, train_labels)
preds = model.predict_proba(test_features)[:, 1]
baseline_auc = roc_auc_score(test_labels, preds)
print('The baseline model scores {:.5f} ROC AUC on the test set.'.format(baseline_auc))


"""
定义目标函数
"""
def objective(hyperparameters, iteration):
    """Objective function for grid and random search. Returns
       the cross validation score from a set of hyperparameters."""

    # Number of estimators will be found using early stopping
    if 'n_estimators' in hyperparameters.keys():
        del hyperparameters['n_estimators']

    # Perform n_folds cross validation
    cv_results = lgb.cv(hyperparameters, train_set, num_boost_round=10000, nfold=N_FOLDS, early_stopping_rounds=100, metrics='auc', seed=42)

    # results to retun
    score = cv_results['auc-mean'][-1]
    estimators = len(cv_results['auc-mean'])
    hyperparameters['n_estimators'] = estimators
    return [score, hyperparameters, iteration]


score, params, iteration = objective(default_params, 1)
print('The cross-validation ROC AUC was {:.5f}.'.format(score))


"""定义超参数"""
# Create a default model
model = lgb.LGBMModel()
model.get_params()
# Hyperparameter grid
param_grid = {
    'boosting_type': ['gbdt', 'goss', 'dart'],
    'num_leaves': list(range(20, 150)),
    'learning_rate': list(np.logspace(np.log10(0.005), np.log10(0.5), base = 10, num = 1000)),
    'subsample_for_bin': list(range(20000, 300000, 20000)),
    'min_child_samples': list(range(20, 500, 5)),
    'reg_alpha': list(np.linspace(0, 1)),
    'reg_lambda': list(np.linspace(0, 1)),
    'colsample_bytree': list(np.linspace(0.6, 1, 10)),
    'subsample': list(np.linspace(0.5, 1, 100)),
    'is_unbalance': [True, False]
}


import random

random.seed(50)
# Randomly sample a boosting type
boosting_type = random.sample(param_grid['boosting_type'], 1)[0]   # 在param_grid['boosting_type']列表中随机取1个值

# Set subsample depending on boosting type
subsample = 1.0 if boosting_type == 'goss' else random.sample(param_grid['subsample'], 1)[0]
print('Boosting type: ', boosting_type)
print('Subsample ratio: ', subsample)


import matplotlib.pyplot as plt
import seaborn as sns


"""学习率搜索范围，作图显示 0.005-0.05 与 0.05-0.5分布频数一致"""
# Learning rate histogram
plt.hist(param_grid['learning_rate'], bins = 20, color = 'r', edgecolor = 'k')
plt.xlabel('Learning Rate', size = 14)
plt.ylabel('Count', size = 14)
plt.title('Learning Rate Distribution', size = 18)


"""叶子数 参数范围定义，作图分析"""
plt.hist(param_grid['num_leaves'], color = 'm', edgecolor = 'k')
plt.xlabel('Learning Number of Leaves', size = 14)
plt.ylabel('Count', size = 14)
plt.title('Number of Leaves Distribution', size = 18)


# Dataframes for random and grid search，用于保存训练过程中的一些得分、参数值信息
random_results = pd.DataFrame(columns = ['score', 'params', 'iteration'], index = list(range(MAX_EVALS)))
grid_results = pd.DataFrame(columns = ['score', 'params', 'iteration'], index = list(range(MAX_EVALS)))


"""查看参数组合形式一共有多少种"""
com = 1
for x in param_grid.values():
    com *= len(x)
print('There are {} combinations'.format(com))


import itertools


"""
网格搜索算法
"""
def grid_search(param_grid, max_evals=MAX_EVALS):
    """Grid search algorithm (with limit on max evals)"""
    # 定义df，进行参数值、得分等信息的存储
    results = pd.DataFrame(columns=['score', 'params', 'iteration'], index=list(range(MAX_EVALS)))
    # 生成参数组合
    keys, values = zip(*param_grid.items())
    i = 0    # Iterate次数
    # Iterate through every possible combination of hyperparameters
    for v in itertools.product(*values):
        # Create a hyperparameter dictionary
        hyperparameters = dict(zip(keys, v))
        # Set the subsample ratio accounting for boosting type
        hyperparameters['subsample'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters['subsample']
        # Evalute the hyperparameters
        eval_results = objective(hyperparameters, i)
        results.loc[i, :] = eval_results
        i += 1
        # Normally would not limit iterations
        if i > MAX_EVALS:
            break
    # Sort with best score on top
    results.sort_values('score', ascending=False, inplace=True)
    results.reset_index(inplace=True)
    return results


grid_results = grid_search(param_grid)
print('The best validation score was {:.5f}'.format(grid_results.loc[0, 'score']))
print('\nThe best hyperparameters were:')

import pprint
pprint.pprint(grid_results.loc[0, 'params'])


"""利用网格搜索算法得到的参数，进行test预测"""
# Get the best parameters
grid_search_params = grid_results.loc[0, 'params']

# Create, train, test model
model = lgb.LGBMClassifier(**grid_search_params, random_state=42)
model.fit(train_features, train_labels)
preds = model.predict_proba(test_features)[:, 1]
print('The best model from grid search scores {:.5f} ROC AUC on the test set.'.format(roc_auc_score(test_labels, preds)))


"""
随机搜索算法
"""
"""生成一组随机参数"""
random.seed(50)
# Randomly sample from dictionary
random_params = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
# Deal with subsample ratio
random_params['subsample'] = 1.0 if random_params['boosting_type'] == 'goss' else random_params['subsample']
print(random_params)


def random_search(param_grid, max_evals=MAX_EVALS):
    """Random search for hyperparameter optimization"""
    # Dataframe for results
    results = pd.DataFrame(columns=['score', 'params', 'iteration'],
                           index=list(range(MAX_EVALS)))
    # Keep searching until reach max evaluations
    for i in range(MAX_EVALS):
        # Choose random hyperparameters
        hyperparameters = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}
        hyperparameters['subsample'] = 1.0 if hyperparameters['boosting_type'] == 'goss' else hyperparameters['subsample']
        # Evaluate randomly selected hyperparameters
        eval_results = objective(hyperparameters, i)
        results.loc[i, :] = eval_results
    # Sort with best score on top
    results.sort_values('score', ascending=False, inplace=True)
    results.reset_index(inplace=True)
    return results


random_results = random_search(param_grid)
print('The best validation score was {:.5f}'.format(random_results.loc[0, 'score']))
print('\nThe best hyperparameters were:')
import pprint
pprint.pprint(random_results.loc[0, 'params'])


"""利用随机搜索算法得到的参数进行test预测"""
# Get the best parameters
random_search_params = random_results.loc[0, 'params']
# Create, train, test model
model = lgb.LGBMClassifier(**random_search_params, random_state = 42)
model.fit(train_features, train_labels)
preds = model.predict_proba(test_features)[:, 1]
print('The best model from random search scores {:.5f} ROC AUC on the test set.'.format(roc_auc_score(test_labels, preds)))