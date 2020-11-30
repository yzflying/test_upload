"""
airbnb-recruiting-new-user-bookings   对旅行目的地预测，多分类（12分类）问题
data-science-bowl-2019
favorita-grocery-sales-forecasting
home-credit-default-risk
santander-value-prediction-challenge

id: user id （用户id）
date_account_created（帐号注册时间）: the date of account creation
timestamp_first_active（首次活跃时间）: timestamp of the first activity, note that it can be earlier than date_account_created or date_first_booking because a user can search before signing up
date_first_booking（首次订房时间）: date of first booking
gender（性别）
age（年龄）
signup_method（注册方式）
signup_flow（注册页面）: the page a user came to signup up from
language（语言）: international language preference
affiliate_channel（付费市场渠道）: what kind of paid marketing
affiliate_provider（付费市场渠道名称）: where the marketing is e.g. google, craigslist, other
first_affiliate_tracked（注册前第一个接触的市场渠道）: whats the first marketing the user interacted with before the signing up
signup_app（注册app）
first_device_type(设备类型)
first_browser（浏览器类型）
country_destination（订房国家-需要预测的量）（12分类）

sessions.csv：web sessions log for users（网页浏览数据）
user_id（用户id）: to be joined with the column ‘id’ in users table
action(用户行为)
action_type（用户行为类型）
action_detail（用户行为具体）
device_type（设备类型）
secs_elapsed（停留时长）
"""


"""预测目的地，多分类预测问题"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.preprocessing import StandardScaler  # 标准化工具
from xgboost import XGBClassifier


#Loading the Data
df_train = pd.read_csv('train_users_2.csv')
df_test = pd.read_csv('test_users.csv')
print('the shape of train and test:\n', df_train.shape, df_test.shape)        # ((213451, 16), (62096, 15))


# 合并数据集
labels = df_train.country_destination.values   # numpy.ndarray类型
id_test = df_test.id          # pandas.Series类型
df_train.drop(['country_destination'], axis=1)
# Concatenating train and test data for EDA
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)   # axis=0 横向，操作前后行变化


"""
查看用户增长情况
"""
# dac_train = df_train.date_account_created.value_counts()      # 各个注册时间点，计数
# dac_test = df_test.date_account_created.value_counts()
# # 将时间数据类型转换为datatime类型
# dac_train_date = pd.to_datetime(df_train.date_account_created.value_counts().index)
# dac_test_date = pd.to_datetime(df_test.date_account_created.value_counts().index)
# # 计算离首次注册时间相差的天数
# dac_train_day = dac_train_date - dac_train_date.min()
# dac_test_day = dac_test_date - dac_train_date.min()
# # motplotlib作图
# plt.scatter(dac_train_day.days, dac_train.values, color = 'r', label = 'train dataset')
# plt.scatter(dac_test_day.days, dac_test.values, color = 'b', label = 'test dataset')
# plt.title("Accounts created vs day")
# plt.xlabel("Days")
# plt.ylabel("Accounts created")
# plt.legend(loc='upper left')
# plt.show()


"""
缺失值统计与处理，删除无实际意义的columns，属性异常值处理，
离散变量的编码；时间类型属性的转化与分离，对年龄进行分段bin
"""
# 查看各属性是否有缺失值
for i in df_all.columns:
    ab = df_all[i].isnull().sum()
    if ab != 0:
        print(i + " has {} null values.".format(ab))
        print()


# 删除ID、date_first_booking、country_destination等列
df_all = df_all.drop(['id', 'country_destination', 'date_first_booking'], axis=1)   # axis=1 纵向，操作前后列变化


# gender性别分布
print('the value count of gender:\n', df_all.gender.value_counts())


# age异常值处理
df_all.age.describe()
df_all[df_all['age'] < 15].age = np.nan
df_all[df_all['age'] >= 100].age = np.nan


# 所有样本的性别分布，并计算占比，作图
plt.figure(figsize=(14, 8))
order1 = df_all['gender'].value_counts().index
print(order1)
sns.countplot(data=df_all, x='gender', order=order1)   # 数据df_all根据gender分组统计，gender属性值按order1顺序排列(使用条形显示每个分箱器中的观察计数)
plt.xlabel('Gender')
plt.ylabel('Count')
plt.title('Gender Distribution')

order2 = df_all['gender'].value_counts()
for i in range(order2.shape[0]):
    count = order2[i]
    strt = '{:0.1f}%'.format(100*count / df_all.shape[0])
    plt.text(i, count+1000, strt, ha='center')        # 通过for循环给每个bar添加标记，在第i个bar的纵向count+1000处添加strt
# plt.show()


# 所有训练样本的目的地分布，并计算占比，作图
plt.figure(figsize=(14, 8))
order1 = df_train['country_destination'].value_counts().index
sns.countplot(data=df_train, x='country_destination', order=order1)
plt.xlabel('Destination')
plt.ylabel('Count')
plt.title('Destination Distribution')

order2 = df_train['country_destination'].value_counts()
for i in range(order2.shape[0]):
    count = order2[i]
    strt='{:0.1f}%'.format(100*count / df_train.shape[0])
    plt.text(i, count+1000, strt, ha='center')
# plt.show()


"""
统计year分布，year下month分布
"""
# # 根据属性date_account_created分离新建一个year属性
# df_all['acc_year'] = df_all['date_account_created'].dt.year   # 时间日期类型属性to_datetime之后就可以用pandas.Series.dt.day等方法
#
# df1 = df_all.groupby('acc_year').count()    # 各个acc_year分组下的样本计数
# print('df1:', df1.head())
# print(df1.columns)
# """
#           affiliate_channel  ...  timestamp_first_active
# acc_year                     ...
# 2010                   2788  ...                    2788
# 2011                  11775  ...                   11775
# 2012                  39462  ...                   39462
# 2013                  82960  ...                   82960
# 2014                 138562  ...                  138562
# """
#
# # groupby聚合之后，acc_year属性变为索引，通过reset_index方法将acc_year转为属性
# df1.reset_index(inplace=True)
#
# print('df1:', df1.head())
# print(df1.columns)
# """
#   acc_year  affiliate_channel  ...  signup_method  timestamp_first_active
# 0     2010               2788  ...           2788                    2788
# 1     2011              11775  ...          11775                   11775
# 2     2012              39462  ...          39462                   39462
# 3     2013              82960  ...          82960                   82960
# 4     2014             138562  ...         138562                  138562
# """
#
# # 属性acc_year分布，并计算占比
# plt.figure(figsize=[14, 8])
# sns.barplot(data=df1, x='acc_year', y='affiliate_provider')   # y取任意属性，即代表该分组下的样本数量
# plt.title('Year wise distribution')
# plt.xlabel('Year')
# plt.ylabel('Counts')
# for i in range(df1.shape[0]):
#     count = df1.iloc[i]['affiliate_provider']
#     strt = '{:0.2f}%'.format(100*count/df_all.shape[0])
#     plt.text(i, count+1000, strt, ha='center')
# # plt.show()
#
#
# # 统计2014年每个月份占比
# df2 = df_all.loc[df_all['acc_year'] == 2014, :]     # df.loc[row, cor]来筛选样本及属性
# df2['monthYear14'] = df2['date_account_created'].map(lambda x: x.strftime('%m-%Y'))
#
# df2 = df2.groupby('monthYear14').count()
# # Number of accounts created in different month of 2014
# plt.figure(figsize=[14, 8])
# sns.barplot(data=df2, x=df2.index, y='affiliate_provider')   # 次数x直接用的index，没有将聚合后的monthYear14转化为属性
# plt.title('2014 month wise distribution')
# plt.xlabel('Month-Year')
# plt.ylabel('Counts')
# for i in range(df2.shape[0]):
#     count = df2.iloc[i]['affiliate_provider']
#     strt = '{:0.2f}%'.format(100*count/df_all.shape[0])
#     plt.text(i, count+100, strt, ha='center')
# # plt.show()


# # 新建离散属性member_age_bins，保存年龄的分段结果
# df_all['member_age_bins'] = df_all['age'].apply(lambda x: '18 - 20' if 18 < x <= 20
#                                                   else '20 - 30' if 20 < x <= 30
#                                                   else '30 - 40' if 30 < x <= 40
#                                                   else '40 - 50' if 40 < x <= 50
#                                                   else '50 - 60' if 50 < x <= 60
#                                                   else '60-70' if 60 < x <= 70
#                                                   else '70+' if 70 < x <= 100
#                                                   else np.nan)
#
#
# # 对age属性所有样本，绘制分段直方图
# plt.figure(figsize=[14, 8])
# sns.distplot(df_all.age.dropna(), bins=np.arange(18, 100+5, 5), kde=False)
# plt.xlabel('Age of members')
# plt.ylabel('Count')
# plt.title('Age distribution of Users')
# plt.xlim(18, 100)
# # plt.show()
#
#
# # Destination-Age distribution plot   绘制箱型图age、country_destination
# plt.figure(figsize=[14, 8])
# sns.boxplot(data=df_train, y='age', x='country_destination')
# plt.ylim(18, 100)
# plt.xlabel('Country')
# plt.ylabel('Age')
# plt.title('Country-Age Distribution')
# # plt.show()
#
#
# # Gender-Age Distribution plot      绘制箱型图age、gender
# plt.figure(figsize=[14, 8])
# sns.boxplot(data=df_all, y='age', x='gender')
# plt.ylim(18, 100)
# plt.xlabel('Gender')
# plt.ylabel('Age')
# plt.title('Gender-Age Distribution')
# # plt.show()
#
#
# # 按性别区分，绘制各目的地样本个数
# plt.figure(figsize=(14, 8))
# order1 = df_train['country_destination'].value_counts().index
# sns.countplot(data=df_train, x='country_destination', order=order1, hue='gender')
# plt.xlabel('Destination')
# plt.ylabel('Count')
# plt.title('Gender-Destination Distribution')
# order2 = df_train['country_destination'].value_counts()
# # plt.show()
#
#
# """查看2013年数据集，分各年龄段的统计数目随时间变化情况"""
# df3 = df_all[df_all['date_account_created'].dt.year == 2013]
# df3['monthYear13'] = df3['date_account_created'].map(lambda x: x.strftime('%m-%Y'))
# df3 = df3.groupby(['monthYear13', 'member_age_bins']).count()
#
# df3.reset_index(inplace=True)   # 将'monthYear13', 'member_age_bins'由index转为columns
#
# plt.figure(figsize=[14, 8])
# sns.pointplot(data=df3, x='monthYear13', y='affiliate_provider', hue='member_age_bins')   # 折线图，x为月份monthYear13，y为样本数量count，分年龄段绘制
# plt.title('2013 month-age wise distribution')
# plt.xlabel('Month-Year')
# plt.ylabel('Counts')
# # plt.show()


# """
# 查看train数据集中country_destination==DNF(即没有旅游过)用户和有旅游过的用户随时间增长情况
# """
# df_train['date_account_created'] = pd.to_datetime(df_train['date_account_created'])
#
# sns.set_style("whitegrid", {'axes.edgecolor': '0'})
# sns.set_context("poster", font_scale=1.1)
# plt.figure(figsize=(12, 6))
#
# df_train[df_train['country_destination'] != 'NDF']['date_account_created'].value_counts().plot(kind='line', linewidth=1, color='green')
# df_train[df_train['country_destination'] == 'NDF']['date_account_created'].value_counts().plot(kind='line', linewidth=1, color='red')
# plt.show()


"""
特征工程
"""
"""
# 对时间类型属性转为datetime数据类型
df_all['date_account_created'] = pd.to_datetime(df_all['date_account_created'])
df_all['timestamp_first_active'] = pd.to_datetime(df_all['timestamp_first_active'], format='%Y%m%d%H%M%S')
"""


# date_account_created特征拆分为三个年月日特征，并删除date_account_created 属性
dac = np.vstack(df_all.date_account_created.astype(str).apply(lambda x: list(map(int, x.split('-')))).values)  # np.vstack，横向合并，类似axis=0
df_all['dac_year'] = dac[:, 0]
df_all['dac_month'] = dac[:, 1]
df_all['dac_day'] = dac[:, 2]
df_all = df_all.drop(['date_account_created'], axis=1)

# timestamp_first_active特征拆分为三个年月日特征，并删除timestamp_first_active 属性
tfa = np.vstack(df_all.timestamp_first_active.astype(str).apply(lambda x: list(map(int, [x[:4], x[4:6], x[6:8]]))).values)
df_all['tfa_year'] = tfa[:, 0]
df_all['tfa_month'] = tfa[:, 1]
df_all['tfa_day'] = tfa[:, 2]
df_all = df_all.drop(['timestamp_first_active'], axis=1)


"""
# 对离散变量的属性进行编码
categorical_features = [
    'affiliate_channel',
    'affiliate_provider',
    'first_affiliate_tracked',
    'first_browser',
    'first_device_type',
    'gender',
    'language',
    'signup_app',
    'signup_method',
    'signup_flow'
]
for categorical_feature in categorical_features:
    df_all[categorical_feature] = df_all[categorical_feature].astype('category')
"""


# One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel',
             'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type', 'first_browser']
for f in ohe_feats:
    dfWork_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, dfWork_dummy), axis=1)


"""
XGBClassifier建模
"""
# # Splitting train and test
# vals = df_all.values
# X = vals[:df_train.shape[0]]
# le = LabelEncoder()
# y = le.fit_transform(labels)
# X_test = vals[df_train.shape[0]:]


"""
Classifier
max_depth：树深，一般3-10
n_estimators：总共迭代的次数，即决策树的个数
objective:返回值类型，多分类：概率
subsample：训练每棵树时，使用的数据占全部训练集的比例。默认值为1，典型值为0.5-1
colsample_bytree：训练每棵树时，使用的特征占全部特征的比例。默认值为1，典型值为0.5-1
"""
# xgb = XGBClassifier(max_depth=6, learning_rate=0.3, n_estimators=25,
#                     objective='multi:softprob', subsample=0.5, colsample_bytree=0.5, seed=0)
# xgb.fit(X, y)
#
# # Prediction
# y_pred = xgb.predict_proba(X_test)   # 每个样本的预测结果为一行12列矩阵，对应每个label的可能性值
#
# # Taking the 5 classes with highest probabilities
# ids = []  # list of ids
# cts = []  # list of countries
# for i in range(1):
#     idx = id_test[i]
#     ids += [idx] * 5     # 最终ids是一个mX5的二维数组(m为test样本个数，每行的5列数据一致)
#     # np.argsort(y_pred[i])[::-1]是将i样本预测结果根据值由大到小对参数排序
#     # inverse_transform方法是对编码后的label进行逆转化为编码前label，最终cts是一个mX5的二维数组(m为test样本个数)
#     cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()
#
# # Generate submission
# sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
# sub.to_csv('sub.csv', index=False)


"""
sesion文件文件处理部分
"""
# user_id更改为id，方便后面的数据合并
df_sessions = pd.read_csv('sessions.csv')
df_sessions['id'] = df_sessions['user_id']
df_sessions = df_sessions.drop(['user_id'], axis=1)

# 对action(用户行为)、action_type（用户行为类型）、action_detail（用户行为具体）属性填充，secs_elapsed（停留时长）空值稍后处理
df_sessions.action = df_sessions.action.fillna('NAN')
df_sessions.action_type = df_sessions.action_type.fillna('NAN')
df_sessions.action_detail = df_sessions.action_detail.fillna('NAN')
df_sessions.isnull().sum()

# action有多种，且最少的发生次数只有1，可以对用户发生次数较少的行为列为OTHER一类
act_freq = 100  # Threshold of frequency
# np.unique(df_sessions.action, return_counts=True) 取以数组形式返回非重复的action值和它的数量
# zip（*（a,b））a,b种元素一一对应，返回zip object
act = dict(zip(*np.unique(df_sessions.action, return_counts=True)))
df_sessions.action = df_sessions.action.apply(lambda x: 'OTHER' if act[x] < act_freq else x)

"""
首先将用户的特征根据用户id进行分组
**特征action：**统计每个用户总的action出现的次数，各个action类型的数量，平均值以及标准差
**特征action_detail：**统计每个用户总的action_detail出现的次数，各个action_detail类型的数量，平均值以及标准差
**特征action_type：**统计每个用户总的action_type出现的次数，各个action_type类型的数量，平均值，标准差以及总的停留时长（进行log处理）
**特征device_type：**统计每个用户总的device_type出现的次数，各个device_type类型的数量，平均值以及标准差
**特征secs_elapsed：**对缺失值用0填充，统计每个用户secs_elapsed时间的总和，平均值，标准差以及中位数（进行log处理），（总和/平均数），secs_elapsed（log处理后）各个时间出现的次数
经过特征提取后，session文件由6个特征变为458个特征
"""
# 对action特征进行细化
f_act = df_sessions.action.value_counts().argsort()         # action各属性值(按照统计计数大小排序)
f_act_detail = df_sessions.action_detail.value_counts().argsort()
f_act_type = df_sessions.action_type.value_counts().argsort()
f_dev_type = df_sessions.device_type.value_counts().argsort()

# 按照id进行分组
dgr_sess = df_sessions.groupby(['id'])
# Loop on dgr_sess to create all the features.
samples = []  # samples列表
ln = len(dgr_sess)  # 计算分组后df_sessions的长度


for g in dgr_sess:  # 对dgr_sess中每个id的数据进行遍历
    gr = g[1]  # data frame that comtains all the data for a groupby value 'zzywmcn0jv'

    l = []  # 建一个空列表，临时存放特征

    # the id    for example:'zzywmcn0jv'
    l.append(g[0])  # 将id值放入空列表中

    # number of total actions
    l.append(len(gr))  # 将id对应数据的长度放入列表

    # secs_elapsed 特征中的缺失值用0填充再获取具体的停留时长值
    sev = gr.secs_elapsed.fillna(0).values  # These values are used later.

    # action features 特征-用户行为
    # 每个用户行为出现的次数，各个行为类型的数量，平均值以及标准差
    c_act = [0] * len(f_act)
    for i, v in enumerate(gr.action.values):  # i是从0-1对应的位置，v 是用户行为特征的值
        c_act[f_act[v]] += 1
    _, c_act_uqc = np.unique(gr.action.values, return_counts=True)
    # 计算用户行为行为特征各个类型数量的长度，平均值以及标准差
    c_act += [len(c_act_uqc), np.mean(c_act_uqc), np.std(c_act_uqc)]
    l = l + c_act

    # action_detail features 特征-用户行为具体
    # (how many times each value occurs, numb of unique values, mean and std)
    c_act_detail = [0] * len(f_act_detail)
    for i, v in enumerate(gr.action_detail.values):
        c_act_detail[f_act_detail[v]] += 1
    _, c_act_det_uqc = np.unique(gr.action_detail.values, return_counts=True)
    c_act_detail += [len(c_act_det_uqc), np.mean(c_act_det_uqc), np.std(c_act_det_uqc)]
    l = l + c_act_detail

    # action_type features  特征-用户行为类型 click等
    # (how many times each value occurs, numb of unique values, mean and std
    # + log of the sum of secs_elapsed for each value)
    l_act_type = [0] * len(f_act_type)
    c_act_type = [0] * len(f_act_type)
    for i, v in enumerate(gr.action_type.values):
        l_act_type[f_act_type[v]] += sev[i]  # sev = gr.secs_elapsed.fillna(0).values ，求每个行为类型总的停留时长
        c_act_type[f_act_type[v]] += 1
    l_act_type = np.log(1 + np.array(l_act_type)).tolist()  # 每个行为类型总的停留时长，差异比较大，进行log处理
    _, c_act_type_uqc = np.unique(gr.action_type.values, return_counts=True)
    c_act_type += [len(c_act_type_uqc), np.mean(c_act_type_uqc), np.std(c_act_type_uqc)]
    l = l + c_act_type + l_act_type

    # device_type features 特征-设备类型
    # (how many times each value occurs, numb of unique values, mean and std)
    c_dev_type = [0] * len(f_dev_type)
    for i, v in enumerate(gr.device_type.values):
        c_dev_type[f_dev_type[v]] += 1
    c_dev_type.append(len(np.unique(gr.device_type.values)))
    _, c_dev_type_uqc = np.unique(gr.device_type.values, return_counts=True)
    c_dev_type += [len(c_dev_type_uqc), np.mean(c_dev_type_uqc), np.std(c_dev_type_uqc)]
    l = l + c_dev_type

    # secs_elapsed features  特征-停留时长
    l_secs = [0] * 5
    l_log = [0] * 15
    if len(sev) > 0:
        # Simple statistics about the secs_elapsed values.
        l_secs[0] = np.log(1 + np.sum(sev))
        l_secs[1] = np.log(1 + np.mean(sev))
        l_secs[2] = np.log(1 + np.std(sev))
        l_secs[3] = np.log(1 + np.median(sev))
        l_secs[4] = l_secs[0] / float(l[1])  #

        # Values are grouped in 15 intervals. Compute the number of values
        # in each interval.
        # sev = gr.secs_elapsed.fillna(0).values
        log_sev = np.log(1 + sev).astype(int)
        # np.bincount():Count number of occurrences of each value in array of non-negative ints.
        l_log = np.bincount(log_sev, minlength=15).tolist()
    l = l + l_secs + l_log

    # The list l has the feature values of one sample.
    samples.append(l)

# preparing objects
samples = np.array(samples)
samp_ar = samples[:, 1:].astype(np.float16)  # 取除id外的特征数据
samp_id = samples[:, 0]  # 取id，id位于第一列

# 为提取的特征创建一个dataframe
col_names = []  # name of the columns
for i in range(len(samples[0]) - 1):  # 减1的原因是因为有个id
    col_names.append('c_' + str(i))  # 起名字的方式
df_agg_sess = pd.DataFrame(samp_ar, columns=col_names)
df_agg_sess['id'] = samp_id
df_agg_sess.index = df_agg_sess.id  # 将id作为index


"""数据归一化"""
X_scaler = StandardScaler()
df_all = X_scaler.fit_transform(df_all)


"""
评分模型：
NDCG是一种衡量排序质量的评价指标，该指标考虑了所有元素的相关性
由于我们预测的目标变量并不是二分类变量，故我们用NDGG模型来进行模型评分，判断模型优劣
"""
from sklearn.metrics import make_scorer


def dcg_score(y_true, y_score, k=5):
    """
    y_true : array, shape = [n_samples] #数据
        Ground truth (true relevance labels).
    y_score : array, shape = [n_samples, n_classes] #预测的分数
        Predicted scores.
    k : int
    """
    order = np.argsort(y_score)[::-1]  # 分数从高到低排序
    y_true = np.take(y_true, order[:k])  # 取出前k[0,k）个分数

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)


def ndcg_score(ground_truth, predictions, k=5):
    """
    Parameters
    ----------
    ground_truth : array, shape = [n_samples]
        Ground truth (true labels represended as integers).
    predictions : array, shape = [n_samples, n_classes]
        Predicted probabilities. 预测的概率
    k : int
        Rank.
    """
    lb = LabelBinarizer()
    lb.fit(range(len(predictions) + 1))
    T = lb.transform(ground_truth)
    scores = []
    # Iterate over each y_true and compute the DCG score
    for y_true, y_score in zip(T, predictions):
        actual = dcg_score(y_true, y_score, k)
        best = dcg_score(y_true, y_true, k)
        score = float(actual) / float(best)
        scores.append(score)

    return np.mean(scores)


"""
建模部分
"""
"""
逻辑回归
The training score is: 0.7595244143892934
The cv score is: 0.7416926026958558
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


"""
C:正则化强度的倒数
penalty：惩罚项，L2
"""
lr = LogisticRegression(C=1.0, penalty='l2', multi_class='ovr')

RANDOM_STATE = 2017  # 随机种子

# k-fold cross validation（k-折叠交叉验证）
kf = KFold(n_splits=5, random_state=RANDOM_STATE)  # 分成5个组
train_score = []
cv_score = []

# select a k  (value how many y):
k_ndcg = 3
# kf.split: Generate indices to split data into training and test set.
for train_index, test_index in kf.split(xtrain_new, ytrain_new):
    # 训练集数据分割为训练集和测试集，y是目标变量
    X_train, X_test = xtrain_new[train_index, :], xtrain_new[test_index, :]
    y_train, y_test = ytrain_new[train_index], ytrain_new[test_index]

    lr.fit(X_train, y_train)

    y_pred = lr.predict_proba(X_test)
    train_ndcg_score = ndcg_score(y_train, lr.predict_proba(X_train), k=k_ndcg)
    cv_ndcg_score = ndcg_score(y_test, y_pred, k=k_ndcg)

    train_score.append(train_ndcg_score)
    cv_score.append(cv_ndcg_score)

print("\nThe training score is: {}".format(np.mean(train_score)))
print("\nThe cv score is: {}".format(np.mean(cv_score)))


"""
Tree模型
"""
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import *
from sklearn.svm import SVC, LinearSVC, NuSVC


LEARNING_RATE = 0.1
N_ESTIMATORS = 50
RANDOM_STATE = 2017
MAX_DEPTH = 9

# 建了一个tree字典
clf_tree = {
    'RF': RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                 max_depth=MAX_DEPTH,
                                 random_state=RANDOM_STATE),

    'AdaBoost': AdaBoostClassifier(n_estimators=N_ESTIMATORS,
                                   learning_rate=LEARNING_RATE,
                                   random_state=RANDOM_STATE),

    'GraBoost': GradientBoostingClassifier(learning_rate=LEARNING_RATE,
                                           max_depth=MAX_DEPTH,
                                           n_estimators=N_ESTIMATORS,
                                           random_state=RANDOM_STATE)
}
train_score = []
cv_score = []

kf = KFold(n_splits=3, random_state=RANDOM_STATE)

k_ndcg = 5

for key in clf_tree.keys():

    clf = clf_tree.get(key)

    train_score_iter = []
    cv_score_iter = []

    for train_index, test_index in kf.split(xtrain_new, ytrain_new):
        X_train, X_test = xtrain_new[train_index, :], xtrain_new[test_index, :]
        y_train, y_test = ytrain_new[train_index], ytrain_new[test_index]

        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)
        train_ndcg_score = ndcg_score(y_train, clf.predict_proba(X_train), k=k_ndcg)
        cv_ndcg_score = ndcg_score(y_test, y_pred, k=k_ndcg)

        train_score_iter.append(train_ndcg_score)
        cv_score_iter.append(cv_ndcg_score)

    train_score.append(np.mean(train_score_iter))
    cv_score.append(np.mean(cv_score_iter))


train_score_tree = train_score    # 训练得分
cv_score_tree = cv_score          # 交叉验证得分


"""
svm模型
"""
TOL = 1e-4
MAX_ITER = 1000

clf_svm = {

    'SVM-rbf': SVC(kernel='rbf',
                   max_iter=MAX_ITER,
                   tol=TOL, random_state=RANDOM_STATE,
                   decision_function_shape='ovr'),
}

train_score_svm = []
cv_score_svm = []

kf = KFold(n_splits=3, random_state=RANDOM_STATE)

k_ndcg = 5

for key in clf_svm.keys():

    clf = clf_svm.get(key)

    train_score_iter = []
    cv_score_iter = []

    for train_index, test_index in kf.split(xtrain_new, ytrain_new):
        X_train, X_test = xtrain_new[train_index, :], xtrain_new[test_index, :]
        y_train, y_test = ytrain_new[train_index], ytrain_new[test_index]

        clf.fit(X_train, y_train)

        y_pred = clf.decision_function(X_test)
        train_ndcg_score = ndcg_score(y_train, clf.decision_function(X_train), k=k_ndcg)
        cv_ndcg_score = ndcg_score(y_test, y_pred, k=k_ndcg)

        train_score_iter.append(train_ndcg_score)
        cv_score_iter.append(cv_ndcg_score)

    train_score_svm.append(np.mean(train_score_iter))
    cv_score_svm.append(np.mean(cv_score_iter))


"""
xgboost模型
The training score is: 0.803445955699075
The cv score is: 0.7721491602424301
"""
import xgboost as xgb


def customized_eval(preds, dtrain):
    labels = dtrain.get_label()
    top = []
    for i in range(preds.shape[0]):
        top.append(np.argsort(preds[i])[::-1][:5])
    mat = np.reshape(np.repeat(labels, np.shape(top)[1]) == np.array(top).ravel(), np.array(top).shape).astype(int)
    score = np.mean(np.sum(mat / np.log2(np.arange(2, mat.shape[1] + 2)), axis=1))
    return 'ndcg5', score


# xgboost parameters

NUM_XGB = 200

params = {}
params['colsample_bytree'] = 0.6
params['max_depth'] = 6
params['subsample'] = 0.8
params['eta'] = 0.3
params['seed'] = RANDOM_STATE
params['num_class'] = 12
params['objective'] = 'multi:softprob'  # output the probability instead of class.
train_score_iter = []
cv_score_iter = []

kf = KFold(n_splits=3, random_state=RANDOM_STATE)

k_ndcg = 5

for train_index, test_index in kf.split(xtrain_new, ytrain_new):
    X_train, X_test = xtrain_new[train_index, :], xtrain_new[test_index, :]
    y_train, y_test = ytrain_new[train_index], ytrain_new[test_index]

    train_xgb = xgb.DMatrix(X_train, label=y_train)
    test_xgb = xgb.DMatrix(X_test, label=y_test)

    watchlist = [(train_xgb, 'train'), (test_xgb, 'test')]

    bst = xgb.train(params,
                    train_xgb,
                    NUM_XGB,
                    watchlist,
                    feval=customized_eval,
                    verbose_eval=3,
                    early_stopping_rounds=5)

    # bst = xgb.train( params, dtrain, num_round, evallist )

    y_pred = np.array(bst.predict(test_xgb))
    y_pred_train = np.array(bst.predict(train_xgb))
    train_ndcg_score = ndcg_score(y_train, y_pred_train, k=k_ndcg)
    cv_ndcg_score = ndcg_score(y_test, y_pred, k=k_ndcg)

    train_score_iter.append(train_ndcg_score)
    cv_score_iter.append(cv_ndcg_score)

train_score_xgb = np.mean(train_score_iter)
cv_score_xgb = np.mean(cv_score_iter)

print("\nThe training score is: {}".format(train_score_xgb))
print("The cv score is: {}\n".format(cv_score_xgb))
