"""
data-science-bowl-2019
考察5种评估类型ird Measurer, Cart Balancer, Cauldron Filler, Chest Sorter, and Mushroom Sorter来判断是否掌握某种技能,以优化历史学习
需要根据样本历史学习情况来预测能否通过某种评估，几次通过；为多分类(四分类)问题

train.csv：
event_id(随机生成的标记，意义不详，与game_session无必然关系)
game_session(游戏记录唯一编号，每完成一个游戏对应的随机编号)，完成一个游戏记录可能包含多个样本行
timestamp
event_data  event_count	event_code	game_time
installation_id(安装的app唯一编号，随机生成，可以理解为单一用户)
title(游戏、活动或评估的名称)
type(游戏、活动或评估)
world(游戏、活动或评估的益智类型，'NONE'(at app's start screen), TREETOPCITY'(Length/Height), 'MAGMAPEAK'(Capacity/Displacement), 'CRYSTALCAVES'(Weight))

train_labels.csv：
game_session(游戏记录唯一编号，每完成一个游戏对应的随机编号)
installation_id(安装的app唯一编号，随机生成，可以理解为单一用户)
title(评估类型名称)
num_correct	num_incorrect	accuracy	accuracy_group(通过尝试次数)

specs.csv：event_id	  info	 args

sample_submission.csv：installation_id	accuracy_group
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
from IPython.display import HTML
pd.set_option('max_columns', 100)


# 读取数据集
train = pd.read_csv('train.csv')
train_labels = pd.read_csv('train_labels.csv')
test = pd.read_csv('test.csv')
specs = pd.read_csv('specs.csv')
ss = pd.read_csv('sample_submission.csv')

# 对train进行样本随机取样
train_ = train.sample(1000000)

# 查看train_labels中的得分accuracy_group分布(先对accuracy_group分组，查看每组的样本game_session个数)
train_labels.head()
train_labels.groupby('accuracy_group')['game_session'].count().plot(kind='barh', figsize=(15, 5), title='Target (accuracy group)')
plt.show()

# 将随机数event_id、game_session转换为整数
train['event_id_as_int'] = train['event_id'].apply(lambda x: int(x, 16))
train['game_session_as_int'] = train['game_session'].apply(lambda x: int(x, 16))

# Format and make date / hour features     对timestamp属性分离出weekday_name、date、hour等信息
train['timestamp'] = pd.to_datetime(train['timestamp'])
train['date'] = train['timestamp'].dt.date
train['hour'] = train['timestamp'].dt.hour
train['weekday_name'] = train['timestamp'].dt.weekday_name
# Same for test
test['timestamp'] = pd.to_datetime(test['timestamp'])
test['date'] = test['timestamp'].dt.date
test['hour'] = test['timestamp'].dt.hour
test['weekday_name'] = test['timestamp'].dt.weekday_name


# 查看installation_id属性值，nunique数目为17000
train['installation_id'].nunique()
train.groupby('installation_id').count()['event_id'].plot(kind='hist',bins=40,figsize=(15, 5), title='Count of Observations by installation_id')
plt.show()

train.groupby('installation_id').count()['event_id'].apply(np.log1p).plot(kind='hist', bins=40, figsize=(15, 5), title='Log(Count) of Observations by installation_id')
plt.show()

# 查看计数较多的installation_id
train.groupby('installation_id').count()['event_id'].sort_values(ascending=False).head(5)


# event codes的分布
train.groupby('event_code') .count()['event_id'].sort_values().plot(kind='bar', figsize=(15, 5),title='Count of different event codes.')
plt.show()

# title的分布
train.groupby('title')['event_id'].count().sort_values().plot(kind='barh',title='Count of Observation by Game/Video title', figsize=(15, 15))
plt.show()


"""对installation_id聚合，对hour属性求统计值"""
aggs = {'hour': ['max','min','mean']}

train_aggs = train.groupby('installation_id').agg(aggs)
train_aggs = train_aggs.reset_index()
train_aggs.columns = ['_'.join(col).strip() for col in train_aggs.columns.values]
train_aggs = train_aggs.rename(columns={'installation_id_' : 'installation_id'})

# Hmmm... not 1:1   合并数据集属性
train_aggs.merge(train_labels[['installation_id','accuracy_group']],how='left')


"""定义一个函数，查看数据集的缺失值"""
def missing_data(data):
    """返回每个属性的空值、空值占比、属性数据类型。['Total', 'Percent', 'Types']"""
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))


missing_data(train)


"""定义一个函数，查看数据集每个属性去重前后的元素个数"""
def unique_values(data):
    """返回每个属性去重前后的元素个数['Total', 'Uniques']"""
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in data.columns:
        unique = data[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    return(np.transpose(tt))


unique_values(train)


"""定义一个函数，查看数据集每个属性重复次数最多的元素"""
def most_frequent_values(data):
    """返回每个属性重复次数最多的元素、重复次数、占样本比率['Total', 'Most frequent item', 'Frequence','Percent from total']"""
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    for col in data.columns:
        itm = data[col].value_counts().index[0]
        val = data[col].value_counts().values[0]
        items.append(itm)
        vals.append(val)
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals/total * 100, 3)
    return(np.transpose(tt))


"""计算数据库df的属性feature下各属性值次数与占比"""
def plot_count(feature, title, df, size=1):
    f, ax = plt.subplots(1,1, figsize=(4*size,4))
    total = float(len(df))
    g = sns.countplot(df[feature], order = df[feature].value_counts().index[:20], palette='Set3')
    g.set_title("Number and percentage of {}".format(title))
    if(size > 2):
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2.,
                height + 3,
                '{:1.2f}%'.format(100*height/total),
                ha="center")
    plt.show()


plot_count('title', 'title (first most frequent 20 values - train)', train, size=4)


"""
检查目标target在训练集与测试集分布是否有差异
Kappa一致性系数:用于衡量多分类(二分类用准确率、ROC曲线来衡量预测与真实值、模型好坏)系统预测值、真实值，来确定模型好坏；kappa = (p_o - p_e) / (1 - p_e)
kappa介于0-1，可分为五组来表示不同级别的一致性：0.0~0.20极低、0.21~0.40一般、0.41~0.60 中等、0.61~0.80 高度的一致性、0.81~1几乎完全一致
"""
from collections import Counter
import sklearn


def eval_qwk_lgb_regr(y_true, y_pred):
    # Counter，统计列表中各元素及其次数，字典类型。元素名称:统计次数
    dist = Counter(train['accuracy_group'])     # accuracy_group属性元素有0、1、2、3
    # disk[k]由元素k的次数转换为元素k的占比
    for k in dist:
        dist[k] /= len(train)
    train['accuracy_group'].hist()

    acum = 0
    bound = {}
    for i in range(3):
        acum += dist[i]
        bound[i] = np.percentile(y_pred, acum * 100)   # np.percentile(list, percent) 将list正向排序，返回列表的percent%百分位位置的数

    def classify(x):
        if x <= bound[0]:
            return 0
        elif x <= bound[1]:
            return 1
        elif x <= bound[2]:
            return 2
        else:
            return 3

    y_pred = np.array(list(map(classify, y_pred))).reshape(y_true.shape)
    return 'cappa', sklearn.metrics.cohen_kappa_score(y_true, y_pred, weights='quadratic'), True


def cohenkappa(ypred, y):
    """根据ypred、y 计算损失loss"""
    y = y.get_label().astype("int")
    ypred = ypred.reshape((4, -1)).argmax(axis = 0)
    loss = sklearn.metrics.cohen_kappa_score(y, ypred, weights = 'quadratic')
    return "cappa", loss, True


def encode_title(train, test, train_labels):
    """将title、world等属性编码"""
    # map(fuc, list),依据fuc(list[i])生成一个新列表。新建属性title_event_code，为str(x) + '_' + str(y)的形式(其中x，y为train['title'], train['event_code'])
    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))
    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))
    # x.union(y)，合并并去重x，y俩集合的元素
    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))
    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))
    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))
    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))
    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))
    # zip(a, b) 将a、b列表打包为基本元素为元组的列表 [(a1,b1),(a2,b2)]，dict()函数转为字典形式
    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))
    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))
    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))
    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))
    # replace the text titles with the number titles from the dict
    train['title'] = train['title'].map(activities_map)  # activities_map是{title: index}的形式。此处将train['title']转化为activities_map[train['title']]，即index
    test['title'] = test['title'].map(activities_map)
    train['world'] = train['world'].map(activities_world)
    test['world'] = test['world'].map(activities_world)
    train_labels['title'] = train_labels['title'].map(activities_map)
    # convert text into datetime
    train['timestamp'] = pd.to_datetime(train['timestamp'])
    test['timestamp'] = pd.to_datetime(test['timestamp'])

    return train, test, train_labels, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code


class Base_Model(object):

    def __init__(self, train_df, test_df, features, categoricals=[], n_splits=5, verbose=True):
        self.train_df = train_df
        self.test_df = test_df
        self.features = features
        self.n_splits = n_splits
        self.categoricals = categoricals
        self.target = 'accuracy_group'
        self.cv = self.get_cv()
        self.verbose = verbose
        self.params = self.get_params()
        self.y_pred, self.score, self.model = self.fit()

    def train_model(self, train_set, val_set):
        raise NotImplementedError

    def get_cv(self):
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        return cv.split(self.train_df, self.train_df[self.target])

    def get_params(self):
        raise NotImplementedError

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        raise NotImplementedError

    def convert_x(self, x):
        return x

    def fit(self):
        oof_pred = np.zeros((len(reduce_train), ))
        y_pred = np.zeros((len(reduce_test), ))
        for fold, (train_idx, val_idx) in enumerate(self.cv):
            x_train, x_val = self.train_df[self.features].iloc[train_idx], self.train_df[self.features].iloc[val_idx]
            y_train, y_val = self.train_df[self.target][train_idx], self.train_df[self.target][val_idx]
            train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)
            model = self.train_model(train_set, val_set)
            conv_x_val = self.convert_x(x_val)
            oof_pred[val_idx] = model.predict(conv_x_val).reshape(oof_pred[val_idx].shape)
            x_test = self.convert_x(self.test_df[self.features])
            y_pred += model.predict(x_test).reshape(y_pred.shape) / self.n_splits
            print('Partial score of fold {} is: {}'.format(fold, eval_qwk_lgb_regr(y_val, oof_pred[val_idx])[1]))
        _, loss_score, _ = eval_qwk_lgb_regr(self.train_df[self.target], oof_pred)
        if self.verbose:
            print('Our oof cohen kappa score is: ', loss_score)
        return y_pred, loss_score, model


class Lgb_Model(Base_Model):

    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0
        return lgb.train(self.params, train_set, valid_sets=[train_set, val_set], verbose_eval=verbosity)

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = lgb.Dataset(x_train, y_train, categorical_feature=self.categoricals)
        val_set = lgb.Dataset(x_val, y_val, categorical_feature=self.categoricals)
        return train_set, val_set

    def get_params(self):
        params = {'n_estimators':5000,
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'metric': 'rmse',
                    'subsample': 0.75,
                    'subsample_freq': 1,
                    'learning_rate': 0.01,
                    'feature_fraction': 0.9,
                    'max_depth': 15,
                    'lambda_l1': 1,
                    'lambda_l2': 1,
                    'early_stopping_rounds': 100
                    }
        return params


class Xgb_Model(Base_Model):

    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0
        return xgb.train(self.params, train_set,
                         num_boost_round=5000, evals=[(train_set, 'train'), (val_set, 'val')],
                         verbose_eval=verbosity, early_stopping_rounds=100)

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = xgb.DMatrix(x_train, y_train)
        val_set = xgb.DMatrix(x_val, y_val)
        return train_set, val_set

    def convert_x(self, x):
        return xgb.DMatrix(x)

    def get_params(self):
        params = {'colsample_bytree': 0.8,
            'learning_rate': 0.01,
            'max_depth': 10,
            'subsample': 1,
            'objective':'reg:squarederror',
            #'eval_metric':'rmse',
            'min_child_weight':3,
            'gamma':0.25,
            'n_estimators':5000}

        return params


import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder

class Nn_Model(Base_Model):

    def __init__(self, train_df, test_df, features, categoricals=[], n_splits=5, verbose=True):
        features = features.copy()
        if len(categoricals) > 0:
            for cat in categoricals:
                enc = OneHotEncoder()
                train_cats = enc.fit_transform(train_df[[cat]])
                test_cats = enc.transform(test_df[[cat]])
                cat_cols = ['{}_{}'.format(cat, str(col)) for col in enc.active_features_]
                features += cat_cols
                train_cats = pd.DataFrame(train_cats.toarray(), columns=cat_cols)
                test_cats = pd.DataFrame(test_cats.toarray(), columns=cat_cols)
                train_df = pd.concat([train_df, train_cats], axis=1)
                test_df = pd.concat([test_df, test_cats], axis=1)
        scalar = MinMaxScaler()
        train_df[features] = scalar.fit_transform(train_df[features])
        test_df[features] = scalar.transform(test_df[features])
        print(train_df[features].shape)
        super().__init__(train_df, test_df, features, categoricals, n_splits, verbose)

    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(train_set['X'].shape[1],)),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.LayerNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='relu')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=4e-4), loss='mse')
        print(model.summary())
        save_best = tf.keras.callbacks.ModelCheckpoint('nn_model.w8', save_weights_only=True, save_best_only=True, verbose=1)
        early_stop = tf.keras.callbacks.EarlyStopping(patience=20)
        model.fit(train_set['X'],
                train_set['y'],
                validation_data=(val_set['X'], val_set['y']),
                epochs=100,
                 callbacks=[save_best, early_stop])
        model.load_weights('nn_model.w8')
        return model

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = {'X': x_train, 'y': y_train}
        val_set = {'X': x_val, 'y': y_val}
        return train_set, val_set

    def get_params(self):
        return None


from random import choice

class Cnn_Model(Base_Model):

    def __init__(self, train_df, test_df, features, categoricals=[], n_splits=5, verbose=True):
        features = features.copy()
        if len(categoricals) > 0:
            for cat in categoricals:
                enc = OneHotEncoder()
                train_cats = enc.fit_transform(train_df[[cat]])
                test_cats = enc.transform(test_df[[cat]])
                cat_cols = ['{}_{}'.format(cat, str(col)) for col in enc.active_features_]
                features += cat_cols
                train_cats = pd.DataFrame(train_cats.toarray(), columns=cat_cols)
                test_cats = pd.DataFrame(test_cats.toarray(), columns=cat_cols)
                train_df = pd.concat([train_df, train_cats], axis=1)
                test_df = pd.concat([test_df, test_cats], axis=1)
        scalar = MinMaxScaler()
        train_df[features] = scalar.fit_transform(train_df[features])
        test_df[features] = scalar.transform(test_df[features])
        self.create_feat_2d(features)
        super().__init__(train_df, test_df, features, categoricals, n_splits, verbose)

    def create_feat_2d(self, features, n_feats_repeat=50):
        self.n_feats = len(features)
        self.n_feats_repeat = n_feats_repeat
        self.mask = np.zeros((self.n_feats_repeat, self.n_feats), dtype=np.int32)
        for i in range(self.n_feats_repeat):
            l = list(range(self.n_feats))
            for j in range(self.n_feats):
                c = l.pop(choice(range(len(l))))
                self.mask[i, j] = c
        self.mask = tf.convert_to_tensor(self.mask)
        print(self.mask.shape)


    def train_model(self, train_set, val_set):
        verbosity = 100 if self.verbose else 0

        inp = tf.keras.layers.Input(shape=(self.n_feats))
        x = tf.keras.layers.Lambda(lambda x: tf.gather(x, self.mask, axis=1))(inp)
        x = tf.keras.layers.Reshape((self.n_feats_repeat, self.n_feats, 1))(x)
        x = tf.keras.layers.Conv2D(18, (50, 50), strides=50, activation='relu')(x)
        x = tf.keras.layers.Flatten()(x)
        #x = tf.keras.layers.Dense(200, activation='relu')(x)
        #x = tf.keras.layers.LayerNormalization()(x)
        #x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(100, activation='relu')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(50, activation='relu')(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        out = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(inp, out)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse')
        print(model.summary())
        save_best = tf.keras.callbacks.ModelCheckpoint('nn_model.w8', save_weights_only=True, save_best_only=True, verbose=1)
        early_stop = tf.keras.callbacks.EarlyStopping(patience=20)
        model.fit(train_set['X'],
                train_set['y'],
                validation_data=(val_set['X'], val_set['y']),
                epochs=100,
                 callbacks=[save_best, early_stop])
        model.load_weights('nn_model.w8')
        return model

    def convert_dataset(self, x_train, y_train, x_val, y_val):
        train_set = {'X': x_train, 'y': y_train}
        val_set = {'X': x_val, 'y': y_val}
        return train_set, val_set

    def get_params(self):
        return None
