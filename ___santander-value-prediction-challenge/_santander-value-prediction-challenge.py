"""
airbnb-recruiting-new-user-bookings
data-science-bowl-2019
favorita-grocery-sales-forecasting
home-credit-default-risk
santander-value-prediction-challenge  银行对客户的潜在交易价值进行预测，对客户提供个性化、及时的服务，回归类预测问题
"""

"""
方法一：使用lightgbm模型预测
"""
"""导入相关包"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing, model_selection, metrics
import lightgbm as lgb


"""打开数据集"""
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
print("Train rows and columns : ", train_df.shape)   # 4459, 4993  属性比样本多，需要对属性进行处理
print("Test rows and columns : ", test_df.shape)     # 49342, 4992

train_df.head()    # 属性名称统一编码，不知代表意义；训练值很多零值，稀疏数组


"""
对target列排序，作散点图：查看target有无离群点、target值分布范围
无离群点、target取值范围较大
"""
plt.figure(figsize=(8,6))      # figure尺寸大小，单位英寸
plt.scatter(range(train_df.shape[0]), np.sort(train_df['target'].values))   # target值分布散点图
plt.xlabel('index', fontsize=12)
plt.ylabel('Target', fontsize=12)
plt.title("Target Distribution", fontsize=14)
plt.show()

"""
对target列分段，作直方图：查看target各取值区间的样本数目
target偏态分布，主要集中在target较小的区间
"""
plt.figure(figsize=(12,8))
sns.distplot(train_df["target"].values, bins=50, kde=False)  # target分段，直方图
plt.xlabel('Target', fontsize=12)
plt.title("Target Histogram", fontsize=14)
plt.show()

"""
对target列进行分段，作直方图：查看target各取值区间（对数区间）的样本数目
"""
plt.figure(figsize=(12,8))
sns.distplot( np.log1p(train_df["target"].values), bins=50, kde=False)   # 对target进行ln(x+1)处理，分段作直方图
plt.xlabel('Target', fontsize=12)
plt.title("Log of Target Histogram", fontsize=14)
plt.show()


"""
统计各属性缺失值数量
无缺失值
"""
missing_df = train_df.isnull().sum(axis=0).reset_index()   # 按列统计缺失值，得到Series；reset_index()方法将Series转为DataFrame并添加列索引
missing_df.columns = ['column_name', 'missing_count']      # 对列columns重命名
missing_df = missing_df[missing_df['missing_count']>0]     # 删掉缺失值统计missing_count列为0的样本
missing_df = missing_df.sort_values(by='missing_count')    # 依据缺失值统计missing_count排序
print(missing_df)


"""
统计各属性的数据类型
数值型属性较多
"""
dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df_cnt = dtype_df.groupby("Column Type").aggregate('count').reset_index()   # 对Column Type分组，统计每组数量count
print(dtype_df_cnt)


"""
统计各属性的取值个数
删掉取值个数为1，即常值的属性，共256个属性
"""
unique_df = train_df.nunique().reset_index()        # unique()返回各属性去重后的值；nunique()返回各属性取值去重后的个数
unique_df.columns = ["col_name", "unique_count"]
constant_df = unique_df[unique_df["unique_count"]==1]
print(str(constant_df.col_name.tolist()))           # 属性名转列表


"""
spearman相关系数:衡量两个变量（各属性与标签）的依赖性指标。对于大于500左右的数据集.
它利用单调方程评价两个统计变量的相关性。
如果数据中没有重复值，并且当两个变量完全单调相关时，斯皮尔曼相关系数则为+1或−1;相关性越小，相关系数趋近0
ρ = (∑(xi-x)(yi-y))/(∑(xi-x)²*∑(yi-y)²)½
"""
from scipy.stats import spearmanr


labels = []
values = []
for col in train_df.columns:
    if col not in ["ID", "target"]:
        labels.append(col)
        #计算各数值类属性与target的斯皮尔曼相关系数,忽略空值
        values.append(spearmanr(train_df[col].values, train_df["target"].values, nan_policy='omit')[0])
# 创建DataFrame，含col_labels与corr_values；并根据corr_values排序
corr_df = pd.DataFrame({'col_labels':labels, 'corr_values':values})
corr_df = corr_df.sort_values(by='corr_values')
# 筛选相关系数大于0.1的属性
corr_df = corr_df[(corr_df['corr_values']>0.1) | (corr_df['corr_values']<-0.1)]

# 各属性col_labels与相关系数corr_values作图
ind = np.arange(corr_df.shape[0])
fig, ax = plt.subplots(figsize=(12, 30)) # fig, ax = plt.subplots(2, 2, figsize=(12, 30))  # 定义一个2*2的图
ax.barh(ind, np.array(corr_df.corr_values.values), color='b')  # ax即位置，将ax替换为ax[0]即对特定位置subplot作图
ax.set_yticks(ind)   # y的bar数量
ax.set_yticklabels(corr_df.col_labels.values, rotation='horizontal') # y的bar名称label
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient of the variables")
plt.show()


"""
对上述与标签相关性较大的各个属性，作图显示各属性之间的斯皮尔曼相关性
对相关性较大的各属性，可只取一个，舍弃其余属性
"""
# 筛选相关性较大的属性
cols_to_use = corr_df[(corr_df['corr_values']>0.11) | (corr_df['corr_values']<-0.11)].col_labels.tolist()
temp_df = train_df[cols_to_use]
# 计算各属性之间的斯皮尔曼相关性
corrmat = temp_df.corr(method='spearman')
# 作图，将矩形数据corrmat绘制为颜色编码矩阵
sns.heatmap(corrmat, vmax=1., square=True, cmap="YlGnBu", annot=True)
plt.title("Important variables correlation map", fontsize=15)
plt.show()


"""
建立非线性模型极端随机森林回归ExtraTreesRegressor，衡量各属性的重要性
"""
from sklearn import ensemble


# 数据预处理
train_X = train_df.drop(constant_df.col_name.tolist() + ["ID", "target"], axis=1)  # 删掉常量属性，ID属性
test_X = test_df.drop(constant_df.col_name.tolist() + ["ID"], axis=1)
train_y = np.log1p(train_df["target"].values)   # 对偏态分布的target取对数
# 建立模型并拟合
model = ensemble.ExtraTreesRegressor(n_estimators=200, max_depth=20, max_features=0.5, n_jobs=-1, random_state=0)
model.fit(train_X, train_y)
# 获取属性importances值并作图
feat_names = train_X.columns.values  		# 获取columns名称列表，注意与train_df["target"].values区别
importances = model.feature_importances_  	# 各属性重要性值，list类型
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)		# 各属性在各tree误差的标准差
indices = np.argsort(importances)[::-1][:20]  # 各属性importances值排序，取前20属性索引
# 作图
plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")  #条形图，带误差yerr
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()


"""
建立非线性模型Light GBM，衡量各属性的重要性
"""
def run_lgb(train_X, train_y, val_X, val_y, test_X):
    """
    输入训练集、验证集、测试集
    输出测试集结果，模型，验证集结果
    """
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 30,
        "learning_rate": 0.01,
        "bagging_fraction": 0.7,
        "feature_fraction": 0.7,
        "bagging_frequency": 5,
        "bagging_seed": 2018,
        "verbosity": -1
    }

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=200,
                      evals_result=evals_result)

    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result


# KFold交叉验证
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=2017)
pred_test_full = 0
# 5折交叉验证，对pred_test取平均值
for dev_index, val_index in kf.split(train_X):
    dev_X, val_X = train_X.loc[dev_index,:], train_X.loc[val_index,:]
    dev_y, val_y = train_y[dev_index], train_y[val_index]
    pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
    pred_test_full += pred_test
pred_test_full /= 5.
# 对pred_test取指数
pred_test_full = np.expm1(pred_test_full)


# Making a submission file #
sub_df = pd.DataFrame({"ID":test_df["ID"].values})
sub_df["target"] = pred_test_full
sub_df.to_csv("baseline_lgb.csv", index=False)


# Light GBM，衡量Feature Importance
fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()


"""
方法二：使用xgboost and lightgbm模型联合预测
"""
"""打开数据集"""
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
test_ID = test['ID']


"""
数据预处理
"""
# 对偏态分布y_train取对数
y_train = train['target']
y_train = np.log1p(y_train)
# 删掉ID属性
train.drop("ID", axis = 1, inplace = True)
train.drop("target", axis = 1, inplace = True)
test.drop("ID", axis = 1, inplace = True)
# 删掉取值个数为1，即常值的属性
cols_with_onlyone_val = train.columns[train.nunique() == 1]
train.drop(cols_with_onlyone_val.values, axis=1, inplace=True)
test.drop(cols_with_onlyone_val.values, axis=1, inplace=True)
# 对各属性值四舍五入，返回 x 的小数点四舍五入到n个数字
NUM_OF_DECIMALS = 32
train = train.round(NUM_OF_DECIMALS)
test = test.round(NUM_OF_DECIMALS)
# 删掉取值重复的columns
colsToRemove = []
columns = train.columns
for i in range(len(columns)-1):
    v = train[columns[i]].values
    dupCols = []
    for j in range(i + 1,len(columns)):
        if np.array_equal(v, train[columns[j]].values):
            colsToRemove.append(columns[j])
train.drop(colsToRemove, axis=1, inplace=True)
test.drop(colsToRemove, axis=1, inplace=True)


"""
特征工程
"""
"""利用RandomForestRegressor对各属性重要性排序，取top 1000特征"""
from sklearn import model_selection
from sklearn import ensemble


NUM_OF_FEATURES = 1000   # 超参数
# 拆分训练测试集
x1, x2, y1, y2 = model_selection.train_test_split(train, y_train.values, test_size=0.20, random_state=5)
# 训练模型
model = ensemble.RandomForestRegressor(n_jobs=-1, random_state=7)
model.fit(x1, y1)
# 测试集x2预测
pred_y2 = model.predict(x2)
# 计算测试集平均误差
mean_err = np.sqrt(np.mean(np.power(y2 - pred_y2, 2)))
print(mean_err)
# 各属性feature的重要性importance
importance_df = pd.DataFrame({'importance': model.feature_importances_, 'feature': train.columns})
# 各属性feature依据重要性importance排序，取前NUM_OF_FEATURES个属性的属性名称feature
col = importance_df.sort_values(by=['importance'], ascending=[False])[:NUM_OF_FEATURES]['feature'].values
train = train[col]
test = test[col]


"""
用Kolmogorov-Smirnov来测试训练数据和测试数据。这是对原假设的双面检验，即两个独立样本是否来自同一连续分布。
如果一个特征在训练集中和在测试集中有不同的分布，我们应该去除这个特征，因为我们在训练中所学的不能泛化。
THRESHOLD_P_VALUE和THRESHOLD_STATISTIC是超参数。
K-S检验原理：
作图，x坐标范围为样本取值范围，y坐标范围为0-1；
点(x,y)中的y代表样本中小于x的样本占总样本比率；点(x,y)连线是一条凸曲线
两条凸曲线的最大距离即统计量statistic，代表两组数据的差异
"""
from scipy.stats import ks_2samp
THRESHOLD_P_VALUE = 0.01 # 显著性水平阈值，ks_2samp方法返回的pvalue值大于该值，则认为是同一分布
THRESHOLD_STATISTIC = 0.3 #统计量，描述了两组数据的差异，ks_2samp方法返回的statistic值小于该值，则认为是同一分布
diff_cols = []
for col in train.columns:
    statistic, pvalue = ks_2samp(train[col].values, test[col].values)
    if pvalue <= THRESHOLD_P_VALUE and np.abs(statistic) > THRESHOLD_STATISTIC:
        diff_cols.append(col)
for col in diff_cols:
    if col in train.columns:
        train.drop(col, axis=1, inplace=True)
        test.drop(col, axis=1, inplace=True)


"""
在原始特征上添加了一些额外的统计特征。此外，我们还添加了低维表示作为特征。NUM_OF_COM是超参数
"""
from sklearn import random_projection


ntrain = len(train)  # train样本数量
ntest = len(test)
tmp = pd.concat([train,test])
weight = ((train != 0).sum()/len(train)).values

tmp_train = train[train!=0]
tmp_test = test[test!=0]
# 添加新的属性：各属性的一些统计量
train["weight_count"] = (tmp_train*weight).sum(axis=1)
test["weight_count"] = (tmp_test*weight).sum(axis=1)
train["count_not0"] = (train != 0).sum(axis=1)
test["count_not0"] = (test != 0).sum(axis=1)
train["sum"] = train.sum(axis=1)
test["sum"] = test.sum(axis=1)
train["var"] = tmp_train.var(axis=1)
test["var"] = tmp_test.var(axis=1)
train["median"] = tmp_train.median(axis=1)
test["median"] = tmp_test.median(axis=1)
train["mean"] = tmp_train.mean(axis=1)
test["mean"] = tmp_test.mean(axis=1)
train["std"] = tmp_train.std(axis=1)
test["std"] = tmp_test.std(axis=1)
train["max"] = tmp_train.max(axis=1)
test["max"] = tmp_test.max(axis=1)
train["min"] = tmp_train.min(axis=1)
test["min"] = tmp_test.min(axis=1)
train["skew"] = tmp_train.skew(axis=1)
test["skew"] = tmp_test.skew(axis=1)
train["kurtosis"] = tmp_train.kurtosis(axis=1)
test["kurtosis"] = tmp_test.kurtosis(axis=1)

del(tmp_train)
del(tmp_test)
NUM_OF_COM = 100 #need tuned
transformer = random_projection.SparseRandomProjection(n_components = NUM_OF_COM)
RP = transformer.fit_transform(tmp)
rp = pd.DataFrame(RP)
columns = ["RandomProjection{}".format(i) for i in range(NUM_OF_COM)]
rp.columns = columns

rp_train = rp[:ntrain]
rp_test = rp[ntrain:]
rp_test.index = test.index

#concat RandomProjection and raw data
train = pd.concat([train,rp_train],axis=1)
test = pd.concat([test,rp_test],axis=1)
del(rp_train)
del(rp_test)


"""
交叉验证与模型融合
"""
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

#define evaluation method for a given model. we use k-fold cross validation on the training set.
#the loss function is root mean square logarithm error between target and prediction
#note: train and y_train are feeded as global variables
NUM_FOLDS = 5 #need tuned


def rmsle_cv(model):
    """
	train, y_train在model的训练误差
	"""
    kf = KFold(NUM_FOLDS, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    # the reason of clone is avoiding affect the original base models
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)
        return self

    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([ model.predict(X) for model in self.models_ ])
        return np.mean(predictions, axis=1)


# 建模model_xgb、model_lgb
model_xgb = xgb.XGBRegressor(colsample_bytree=0.055, colsample_bylevel =0.5,
                             gamma=1.5, learning_rate=0.02, max_depth=32,
                             objective='reg:linear',booster='gbtree',
                             min_child_weight=57, n_estimators=1000, reg_alpha=0,
                             reg_lambda = 0,eval_metric = 'rmse', subsample=0.7,
                             silent=1, n_jobs = -1, early_stopping_rounds = 14,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=144,
                              learning_rate=0.005, n_estimators=720, max_depth=13,
                              metric='rmse',is_training_metric=True,
                              max_bin = 55, bagging_fraction = 0.8,verbose=-1,
                              bagging_freq = 5, feature_fraction = 0.9)
# 调用rmsle_cv函数获取各模型训练集误差
score = rmsle_cv(model_xgb)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
score = rmsle_cv(model_lgb)
print("LGBM score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))
averaged_models = AveragingModels(models = (model_xgb, model_lgb))
score = rmsle_cv(averaged_models)
print("averaged score: {:.4f} ({:.4f})\n" .format(score.mean(), score.std()))


"""
模型融合与提交数据
"""
averaged_models.fit(train.values, y_train)
pred = np.expm1(averaged_models.predict(test.values))
ensemble = pred
sub = pd.DataFrame()
sub['ID'] = test_ID
sub['target'] = ensemble
sub.to_csv('submission.csv',index=False)

#Xgboost score: 1.3582 (0.0640)
#LGBM score: 1.3437 (0.0519)
#averaged score: 1.3431 (0.0586)

#Xgboost score: 1.3566 (0.0525)
#LGBM score: 1.3477 (0.0497)
#averaged score: 1.3438 (0.0516)

#Xgboost score: 1.3540 (0.0621)
#LGBM score: 1.3463 (0.0485)
#averaged score: 1.3423 (0.0556)


"""
方法三：数据分析
"""
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection


"""
导入部分数据
"""
SAMPLE_SIZE = 4459
train_df = pd.read_csv('train.csv').sample(SAMPLE_SIZE)
test_df = pd.read_csv('test.csv').sample(SAMPLE_SIZE)
# 数据集合并，删除target、ID属性
total_df = pd.concat([train_df.drop('target', axis=1), test_df], axis=0).drop('ID', axis=1)
# 删除常量属性，样本去重(代码略)
"""
total_df：		删除target、ID属性；删除常量属性，样本去重；对所有属性非零值进行归一化；对统计量3σ外的属性异常值取对数；
total_df_all：	删除target、ID属性；删除常量属性，样本去重；对所有属性所有值进行归一化；
"""
total_df_all = deepcopy(total_df)
# 异常值处理与归一化
for col in total_df.columns:

    # 对属性值，查看是否有在3σ外的值
    data = total_df[col].values
    data_mean, data_std = np.mean(data), np.std(data)
    cut_off = data_std * 3
    lower, upper = data_mean - cut_off, data_mean + cut_off
    outliers = [x for x in data if x < lower or x > upper]

    # 如果有，对该属性所有值取对数
    if len(outliers) > 0:
        non_zero_idx = data != 0
        total_df.loc[non_zero_idx, col] = np.log(data[non_zero_idx])

    # 对所有属性非零值进行归一化
    nonzero_rows = total_df[col] != 0
    total_df.loc[nonzero_rows, col] = scale(total_df.loc[nonzero_rows, col])

    # Scale all column values
    total_df_all[col] = scale(total_df_all[col])
    gc.collect()

# Train and test index
train_idx = range(0, len(train_df))
test_idx = range(len(train_df), len(total_df))


"""
PCA降维
"""
def test_pca(data, create_plots=True):
    """Run PCA analysis, return embedding"""
    # 创建一个PCA对象，降维至1000
    pca = PCA(n_components=1000)
    # 用data进行降维训练，返回PCA降维后的矩阵pca_trafo
    pca_trafo = pca.fit_transform(data)
    # 将pca_trafo转换为df数据类型，pca_df
    pca_df = pd.DataFrame(
        pca_trafo,
        index=total_df.index,
        columns=["PC" + str(i + 1) for i in range(pca_trafo.shape[1])]
    )
# Only construct plots if requested
    if create_plots:

        # Create two plots next to each other
        _, axes = plt.subplots(2, 2, figsize=(20, 15))
        axes = list(itertools.chain.from_iterable(axes))

        # "explained_variance_ratio_",各主成分方差值占总方差的比率;Plot the explained variance# Plot t
        axes[0].plot(
            pca.explained_variance_ratio_, "--o", linewidth=2,
            label="Explained variance ratio"
        )
        # Plot the cumulative explained variance
        axes[0].plot(
            pca.explained_variance_ratio_.cumsum(), "--o", linewidth=2,
            label="Cumulative explained variance ratio"
        )

        # Show legend
        axes[0].legend(loc="best", frameon=True)


        # 绘制axes[1]、axes[2]、axes[3]；Show biplots
        for i in range(1, 4):
            # Components to be plottet
            x, y = "PC"+str(i), "PC"+str(i+1)
            # Plot biplots
            settings = {'kind': 'scatter', 'ax': axes[i], 'alpha': 0.2, 'x': x, 'y': y}
            pca_df.iloc[train_idx].plot(label='Train', c='#ff7f0e', **settings)
            pca_df.iloc[test_idx].plot(label='Test',  c='#1f77b4', **settings)
        plt.show()
    return pca_df


# Run the PCA and get the embedded dimension
pca_df = test_pca(total_df)
pca_df_all = test_pca(total_df_all, create_plots=False)


"""
t-SNE:一种非线性降维方法，将多维数据降维到2-3维并可视化
一、SNE是先将欧几里得距离转换为条件概率来表达点与点之间的相似度
高维空间中的两个数据点xi和xj，xi以条件概率Pj|i选择xj作为它的邻近点。考虑以xi为中心点的高斯分布，若xj越靠近xi，则Pj|i越大。反之若两者相距较远，则Pj|i极小。
概率pij，正比于xi和xj之间的相似度，即：
pj∣i = exp(−∣∣xi−xj∣∣2/(2σ2i))∑k≠iexp(−∣∣xi−xk∣∣2/(2σ2i))
σ2i表示以xi为中心点的高斯分布的方差
二、假设高维数据点xi和xj在低维空间的映射点分别为yi和yj
y维度较低时，指定方差σ2i为1/2，得到相似度：
qj∣i=exp(−∣∣xi−xj∣∣2)∑k≠iexp(−∣∣xi−xk∣∣2)
三、如果降维效果较小，则pj∣i 与qj∣i相等，因此降维算法的损失函数（KL散度）为：
∑i∑j(pj∣i)log(pj∣i/qj∣i)
四、损失函数KL散度的不对称性
根据损失函数可以看到p和q的影响不一样。当距离较远的两个点来表达距离较近的两个点会产生更大的cost，相反，用较近的两个点来表达较远的两个点产生的cost相对较小。
使用联合概率分布替代条件概率分布，即对于任意i,pij=pji,qij=qji
五、t分布
在低维空间下，我们使用更加偏重长尾分布的方式来将距离转换为概率分布，使得高维度下中低等的距离在映射后能够有一个较大的距离
只在非零项上归一化total_df，训练集和测试集看起来更类似。对所有条目进行归一化total_df_all，训练集和测试集似乎更加分离
"""
def test_tsne(data, ax=None, title='t-SNE'):
    """Run t-SNE and return embedding"""
    # Run t-SNE
    tsne = TSNE(n_components=2, init='pca')   # 降维后为2维，PCA初始化
    Y = tsne.fit_transform(data)

    # Create plot
    for name, idx in zip(["Train", "Test"], [train_idx, test_idx]):
        ax.scatter(Y[idx, 0], Y[idx, 1], label=name, alpha=0.2)
        ax.set_title(title)
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
    ax.legend()
    return Y

# Run t-SNE on PCA embedding
_, axes = plt.subplots(1, 2, figsize=(20, 8))

tsne_df = test_tsne(
    pca_df, axes[0],
    title='t-SNE: Scaling on non-zeros'
)

tsne_df_unique = test_tsne(
    pca_df_all, axes[1],
    title='t-SNE: Scaling on all entries'
)

plt.axis('tight')
plt.show()


"""
For the training set it may be interesting to see how the different target values are separated on the embedded two dimensions
"""
# Create plot
fig, axes = plt.subplots(1, 1, figsize=(10, 8))
sc = axes.scatter(tsne_df[train_idx, 0], tsne_df[train_idx, 1], alpha=0.2, c=np.log1p(train_df.target), cmap=cm)
cbar = fig.colorbar(sc, ax=axes)
cbar.set_label('Log1p(target)')
axes.set_title("t-SNE colored by target")
axes.xaxis.set_major_formatter(NullFormatter())
axes.yaxis.set_major_formatter(NullFormatter())


"""
新建一个结果标签区分训练集与测试集；使用随机森林模型对训练和测试数据集进行区分，查看两者差异
可以看到数据集之间有明显差异
"""
def test_prediction(data):
    """Try to classify train/test samples from total dataframe"""

    # Create a target which is 1 for training rows, 0 for test rows
    y = np.zeros(len(data))
    y[train_idx] = 1

    # Perform shuffled CV predictions of train/test label
    predictions = cross_val_predict(
        ExtraTreesClassifier(n_estimators=100, n_jobs=4),
        data, y,
        cv=StratifiedKFold(
            n_splits=10,
            shuffle=True,
            random_state=42
        )
    )

    # Show the classification report
    print(classification_report(y, predictions))

# Run classification on total raw data
test_prediction(total_df_all)


"""
使用ks_2samp方法查看不同数据集的各属性分布，对有明显差异的属性删除
"""
def get_diff_columns(train_df, test_df, show_plots=True, show_all=False, threshold=0.1):
    """Use KS to estimate columns where distributions differ a lot from each other"""

    # Find the columns where the distributions are very different
    diff_data = []
    for col in tqdm(train_df.columns):
        statistic, pvalue = ks_2samp(
            train_df[col].values,
            test_df[col].values
        )
        if pvalue <= 0.05 and np.abs(statistic) > threshold:
            diff_data.append({'feature': col, 'p': np.round(pvalue, 5), 'statistic': np.round(np.abs(statistic), 2)})

    # Put the differences into a dataframe
    diff_df = pd.DataFrame(diff_data).sort_values(by='statistic', ascending=False)

    if show_plots:
        # Let us see the distributions of these columns to confirm they are indeed different
        n_cols = 7
        if show_all:
            n_rows = int(len(diff_df) / 7)
        else:
            n_rows = 2
        _, axes = plt.subplots(n_rows, n_cols, figsize=(20, 3*n_rows))
        axes = [x for l in axes for x in l]

        # Create plots
        for i, (_, row) in enumerate(diff_df.iterrows()):
            if i >= len(axes):
                break
            extreme = np.max(np.abs(train_df[row.feature].tolist() + test_df[row.feature].tolist()))
            train_df.loc[:, row.feature].apply(np.log1p).hist(
                ax=axes[i], alpha=0.5, label='Train', density=True,
                bins=np.arange(-extreme, extreme, 0.25)
            )
            test_df.loc[:, row.feature].apply(np.log1p).hist(
                ax=axes[i], alpha=0.5, label='Test', density=True,
                bins=np.arange(-extreme, extreme, 0.25)
            )
            axes[i].set_title(f"Statistic = {row.statistic}, p = {row.p}")
            axes[i].set_xlabel(f'Log({row.feature})')
            axes[i].legend()

        plt.tight_layout()
        plt.show()

    return diff_df

# Get the columns which differ a lot between test and train
diff_df = get_diff_columns(total_df.iloc[train_idx], total_df.iloc[test_idx])












