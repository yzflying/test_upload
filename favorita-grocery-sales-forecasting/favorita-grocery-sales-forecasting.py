import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
color = sns.color_palette()
import subprocess
from subprocess import check_output
import gc


"""
favorita-grocery-sales-forecasting   回归问题
"""


# """打开数据集"""
# test = pd.read_csv("test.csv")
# stores = pd.read_csv("stores.csv")
# items = pd.read_csv("items.csv")
# transactions = pd.read_csv("transactions.csv")
# oil = pd.read_csv("oil.csv")
# holiday = pd.read_csv("holidays_events.csv")
#
#
# """优化内存消耗"""
# """查看test数据集内存消耗"""
# mem_test = test.memory_usage(index=True).sum()
# print("test dataset uses ", mem_test/1024**2, " MB")
#
#
# """改变属性数据类型,节省40%内存以上"""
# """将整数型数据类型统一为uint8"""
# test['store_nbr'] = test['store_nbr'].astype(np.uint8)
# test['id'] = test['id'].astype(np.uint32)
# test['item_nbr'] = test['item_nbr'].astype(np.uint32)
# """将时间数据类型统一为datatime"""
# test['date'] = pd.to_datetime(test['date'], format="%Y-%m-%d")
# # 对比改进前后内存消耗
# print(test.memory_usage(index=True))   # series类型数据，包含各属性内存消耗
# new_mem_test = test.memory_usage(index=True).sum()
# print("test dataset uses ", new_mem_test/1024**2, " MB after changes")
#
#
# """
# train数据集处理
# store_nbr  ：商店编号
# item_nbr  ：商品编号
# unit_sales ：预测目标，某种商品销量，为负则代表退货
# onpromotion  : 描述当前商品、商店在当前日期是否处于促销状态(存在较多空值)
# Year、Month、Day ：销售日期
# """
# # 定义对应属性名称的目标数据类型
# dtype_dict = {"id": np.uint32, "store_nbr": np.uint8, "item_nbr": np.uint32, "unit_sales": np.float32}
# # 加载train数据集的part1属性
# train_part1 = pd.read_csv("train.csv", dtype=dtype_dict, usecols=[0, 2, 3, 4])
#
# # 加载train数据集的part2属性
# train_part2 = pd.read_csv("train.csv", dtype=dtype_dict, usecols=[1, 5], parse_dates=[0])
# # 时间类型属性分离,并转化为uint8类型
# train_part2['Year'] = pd.DatetimeIndex(train_part2['date']).year
# train_part2['Month'] = pd.DatetimeIndex(train_part2['date']).month
# train_part2['Day'] = pd.DatetimeIndex(train_part2['date']).day.astype(np.uint8)
# del(train_part2['date'])
# train_part2['Day'] = train_part2['Day'].astype(np.uint8)
# train_part2['Month'] = train_part2['Month'].astype(np.uint8)
# train_part2['Year'] = train_part2['Year'].astype(np.uint16)
# # 缺失值填充0
# train_part2["onpromotion"].fillna(0, inplace=True)
# train_part2["onpromotion"] = train_part2["onpromotion"].astype(np.int8)
#
# # 对train数据集的两个part进行合并
# train = pd.concat([train_part1.reset_index(drop=True), train_part2], axis=1)
# # drop temp files
# del(train_part1)
# del(train_part2)
# # Further Id is just an indicator column, hence not required for analysis
# id = train['id']
# del(train['id'])
# # check memory
# new_mem_train = train.memory_usage().sum()
# print("Train dataset uses ", new_mem_train/1024**2, " MB after changes")
#
#
# """特征工程"""
# """按year属性来分离train样本"""
# train_2013 = train.loc[train['Year'] == 2013]
# train_2014 = train.loc[train['Year'] == 2014]
# train_2015 = train.loc[train['Year'] == 2015]
# train_2016 = train.loc[train['Year'] == 2016]
# train_2017 = train.loc[train['Year'] == 2017]
#
#
# def aggregate_level1(df):
#     """
#     writing a function to get item and store level summary metrics for a specific year
#     :param df: train部分样本的数据集
#     :return: sale_day_store_level、sale_day_item_level、sale_store_item_level
#     """
#     # day-store level 按时间和商店聚类，查看商店销售总量、销售商品数量
#     sale_day_store_level = df.groupby(['Year','Month','Day','store_nbr'], as_index=False)['unit_sales'].agg(['sum','count'])
#     # drop index and rename
#     sale_day_store_level = sale_day_store_level.reset_index().rename(columns={'sum':'store_sales', 'count':'item_variety'})
#     # day-item level 按时间和商品聚类，查看商品销售总量、销售商店数量
#     sale_day_item_level = df.groupby(['Year','Month','Day','item_nbr'],as_index=False)['unit_sales'].agg(['sum','count'])
#     # drop index and rename
#     sale_day_item_level = sale_day_item_level.reset_index().rename(columns={'sum':'item_sales', 'count':'store_spread'})
#     # store item level 按时间year和商品、商店聚类，查看商品销售量、销售次数
#     sale_store_item_level = df.groupby(['Year','store_nbr','item_nbr'],as_index=False)['unit_sales'].agg(['sum','count'])
#     # drop index and rename
#     sale_store_item_level = sale_store_item_level.reset_index().rename(columns={'sum':'item_sales', 'count':'entries'})
#     return sale_day_store_level, sale_day_item_level, sale_store_item_level
#
#
# # 查看2013年的一些聚类统计量 run for 2013
# sale_day_store_level_2013, sale_day_item_level_2013, sale_store_item_level_2013 = aggregate_level1(train_2013)
# sale_day_store_level_2014, sale_day_item_level_2014, sale_store_item_level_2014 = aggregate_level1(train_2014)
# sale_day_store_level_2015, sale_day_item_level_2015, sale_store_item_level_2015 = aggregate_level1(train_2015)
# sale_day_store_level_2016, sale_day_item_level_2016, sale_store_item_level_2016 = aggregate_level1(train_2016)
# sale_day_store_level_2017, sale_day_item_level_2017, sale_store_item_level_2017 = aggregate_level1(train_2017)
#
#
# """将不同年份的train统计量合并"""
# sale_day_store_level = pd.concat([sale_day_store_level_2013, sale_day_store_level_2014,
#                                 sale_day_store_level_2015, sale_day_store_level_2016,
#                                 sale_day_store_level_2017])
# sale_day_item_level = pd.concat([sale_day_item_level_2013, sale_day_item_level_2014,
#                                 sale_day_item_level_2015, sale_day_item_level_2016,
#                                 sale_day_item_level_2017])
# sale_store_item_level = pd.concat([sale_store_item_level_2013, sale_store_item_level_2014,
#                                 sale_store_item_level_2015, sale_store_item_level_2016,
#                                 sale_store_item_level_2017])
#
#
# """清理内存"""
# del(sale_day_store_level_2013)
# del(sale_day_store_level_2014)
# del(sale_day_store_level_2015)
# del(sale_day_store_level_2016)
# del(sale_day_store_level_2017)
# del(sale_day_item_level_2013)
# del(sale_day_item_level_2014)
# del(sale_day_item_level_2015)
# del(sale_day_item_level_2016)
# del(sale_day_item_level_2017)
# del(sale_store_item_level_2013)
# del(sale_store_item_level_2014)
# del(sale_store_item_level_2015)
# del(sale_store_item_level_2016)
# del(sale_store_item_level_2017)
# gc.collect()
#
#
# """将train生成的数据永久化"""
# sale_day_store_level.to_csv("sale_day_store_level.csv")
# sale_day_item_level.to_csv("sale_day_item_level.csv")
# sale_store_item_level.to_csv("sale_store_item_level.csv")
#
#
# """所有年份的train数据根据商店聚类,求每个商店的销量与商品数量"""
# sale_store_level = sale_day_store_level.groupby(['store_nbr'],as_index=False)['store_sales','item_variety'].agg(['sum'])
# # Here the group by gives a multiindex , removing that
# sale_store_level.columns = sale_store_level.columns.droplevel(1)
# sale_store_level = sale_store_level.reset_index()
#
# sale_item_level=sale_day_item_level.groupby(['item_nbr'],as_index=False)['item_sales'].agg(['sum'])
# sale_item_level=sale_item_level.reset_index()
# sale_item_level.head()
#
#
# """销量前十商店作图，store_nbr：store_sales"""
# # Sorting by sales
# temp=sale_store_level.sort_values('store_sales',ascending=False).reset_index(drop=True)
# temp=temp.set_index('store_nbr').head(10)
#
# plt.figure(figsize=(12,8))
# sns.barplot(temp.index,temp.store_sales, alpha=0.8, color=color[2],)
# plt.ylabel('Overall Sales', fontsize=12)
# plt.xlabel('Store Number', fontsize=12)
# plt.title('Top Stores by Overall sale', fontsize=15)
# plt.show()
#
#
# """销量前十商品作图，item_nbr：sum"""
# temp1=sale_item_level.sort_values('sum',ascending=False).reset_index(drop=True)
# temp1=temp1.set_index('item_nbr').head(10)
#
# plt.figure(figsize=(12,8))
# x=temp1.index.values
# y=temp1['sum'].values
# sns.barplot(x,y, alpha=0.8, color=color[8])
# plt.ylabel('Overall Sales', fontsize=12)
# plt.xlabel('Store Number', fontsize=12)
# plt.title('Top Items by Overall sale', fontsize=15)
# plt.show()
#
#
# """按时间聚类，查看当前时间所有商店总销量"""
# # year over year sales
# temp=sale_day_store_level.groupby('Year')['store_sales'].sum()
# plt.figure(figsize=(13,4))
# sns.pointplot(temp.index,temp.values, alpha=0.8, color=color[1],)
# plt.ylabel('Overall Sales', fontsize=12)
# plt.xlabel('Year', fontsize=12)
# plt.title('Sale YOY', fontsize=15)
# plt.xticks(rotation='vertical')
# plt.show()
# # month over month sales
# temp=sale_day_store_level.groupby(['Year','Month'])['store_sales'].sum()
# plt.figure(figsize=(13,4))
# sns.pointplot(temp.index,temp.values, alpha=0.8, color=color[2],)
# plt.ylabel('Overall Sales', fontsize=12)
# plt.xlabel('Month', fontsize=12)
# plt.title('Monthly sales variation', fontsize=15)
# plt.xticks(rotation='vertical')
# plt.show()
#
#
# """按月统计当前时间总销量，作图查看 时间：销量 变化"""
# # month over month sales
# temp=sale_day_store_level.groupby(['Year','Month']).aggregate({'store_sales':np.sum,'Year':np.min,'Month':np.min})
# temp=temp.reset_index(drop=True)
# sns.set(style="whitegrid", color_codes=True)
# # temp
# plt.figure(figsize=(15,8))
# plt.plot(range(1,13),temp.iloc[0:12,0],label="2013")
# plt.plot(range(1,13),temp.iloc[12:24,0],label="2014")
# plt.plot(range(1,13),temp.iloc[24:36,0],label="2015")
# plt.plot(range(1,13),temp.iloc[36:48,0],label="2015")
# plt.ylabel('Overall Sales', fontsize=12)
# plt.xlabel('Month', fontsize=12)
# plt.title('Monthly sales variation', fontsize=15)
# plt.xticks(rotation='vertical')
# plt.legend(['2013', '2014', '2015', '2016'], loc='upper left')
# plt.show()
#
#
# """
# Oil数据集包含date(时间),dcoilwtico(当前价格)两个属性
# 按时间Month聚类，查看当前Month时间oil平均价,并作图
# """
# # also checking the oil price change
# oil['date'] = pd.to_datetime(oil['date'])
# oil['Year'] = oil['date'].dt.year
# oil['Month'] = oil['date'].dt.month
# # Oil price variation over month
# temp=oil.groupby(['Year','Month']).agg(['sum', 'count'])
# temp.columns = temp.columns.droplevel(0)
# temp['avg']=temp['sum']/temp['count']
# #plot
# plt.figure(figsize=(13,4))
# sns.pointplot(temp.index,temp.avg, alpha=0.8, color=color[4],)
# plt.ylabel('Oil price', fontsize=12)
# plt.xlabel('Month', fontsize=12)
# plt.title('Monthly variation in oil price', fontsize=15)
# plt.xticks(rotation='vertical')
# plt.show()
#
#
# """
# stores数据集，描述了所有商店信息
# store_nbr(商店编号,一共54个)、city(所在城市)、state(所在州)、type(商店类型)、cluster(相似商店簇)
# """
# plt.figure(figsize=(15, 12))
# # row col plotnumber - 121
# """拥有各个簇的商店个数"""
# plt.subplot(221)
# temp = stores['cluster'].value_counts()
# sns.barplot(temp.index,temp.values,color=color[5])
# plt.ylabel('Count of stores', fontsize=12)
# plt.xlabel('Cluster', fontsize=12)
# plt.title('Store distribution across cluster', fontsize=15)
# """拥有各个类型的商店个数"""
# plt.subplot(222)
# temp = stores['type'].value_counts()
# sns.barplot(temp.index,temp.values,color=color[7])
# plt.ylabel('Count of stores', fontsize=12)
# plt.xlabel('Type of store', fontsize=12)
# plt.title('Store distribution across store types', fontsize=15)
# """各个州的商店个数"""
# plt.subplot(223)
# # Count of stores for each type
# temp = stores['state'].value_counts()
# sns.barplot(temp.index,temp.values,color=color[8])
# plt.ylabel('Count of stores', fontsize=12)
# plt.xlabel('state', fontsize=12)
# plt.title('Store distribution across states', fontsize=15)
# plt.xticks(rotation='vertical')
# """各个城市的商店个数"""
# plt.subplot(224)
# # Count of stores for each type
# temp = stores['city'].value_counts()
# sns.barplot(temp.index,temp.values,color=color[9])
# plt.ylabel('Count of stores', fontsize=12)
# plt.xlabel('City', fontsize=12)
# plt.title('Store distribution across cities', fontsize=15)
# plt.xticks(rotation='vertical')
# plt.show()
#
#
# """
# 对sale_store_level、stores进行merge，并按照一定分类算总销量并作图
# """
# sale_store_level = sale_store_level.iloc[:,0:2]
# merge = pd.merge(sale_store_level,stores,how='left',on='store_nbr')
# # Sale of stores in different types and clusters
# plt.figure(figsize=(15,12))
# # row col plotnumber - 121
# plt.subplot(221)
# """Sale of stores for each type"""
# temp = merge.groupby(['cluster'])['store_sales'].sum()
# sns.barplot(temp.index,temp.values,color=color[5])
# plt.ylabel('Sales', fontsize=12)
# plt.xlabel('Cluster', fontsize=12)
# plt.title('Cumulative sales across store clusters', fontsize=15)
#
# plt.subplot(222)
# """sale of stores for each type"""
# temp = merge.groupby(['type'])['store_sales'].sum()
# sns.barplot(temp.index,temp.values,color=color[7])
# plt.ylabel('sales', fontsize=12)
# plt.xlabel('Type of store', fontsize=12)
# plt.title('Cumulative sales across store types', fontsize=15)
#
# plt.subplot(223)
# """sale of stores for each type"""
# temp = merge.groupby(['state'])['store_sales'].sum()
# sns.barplot(temp.index,temp.values,color=color[8])
# plt.ylabel('sales', fontsize=12)
# plt.xlabel('state', fontsize=12)
# plt.title('Cumulative sales across states', fontsize=15)
# plt.xticks(rotation='vertical')
#
# plt.subplot(224)
# """sale of stores for city"""
# temp = merge.groupby(['city'])['store_sales'].sum()
# sns.barplot(temp.index,temp.values,color=color[9])
# plt.ylabel('sales', fontsize=12)
# plt.xlabel('City', fontsize=12)
# plt.title('Cumulative sales across cities', fontsize=15)
# plt.xticks(rotation='vertical')
# plt.show()
#
#
# """
# transactions 各商店特定时间的交易量
# date(时间)、store_nbr(商店)、transactions(交易量)
# """
# # month over month sales
# transactions['date'] = pd.to_datetime(transactions['date'])
# """特定时间的商店数量、交易总量"""
# temp=transactions.groupby(['date']).aggregate({'store_nbr':'count','transactions':np.sum})
# temp=temp.reset_index()
# temp_2013=temp[temp['date'].dt.year==2013].reset_index(drop=True)
# temp_2014=temp[temp['date'].dt.year==2014].reset_index(drop=True)
# temp_2015=temp[temp['date'].dt.year==2015].reset_index(drop=True)
# temp_2016=temp[temp['date'].dt.year==2016].reset_index(drop=True)
# temp_2017=temp[temp['date'].dt.year==2017].reset_index(drop=True)
#
# sns.set(style="whitegrid", color_codes=True)
# """各年的商店数量作图"""
# plt.figure(figsize=(15,14))
# plt.subplot(211)
# plt.plot(temp_2013['date'],temp_2013.iloc[:,1],label="2013")
# plt.plot(temp_2014['date'],temp_2014.iloc[:,1],label="2014")
# plt.plot(temp_2015['date'],temp_2015.iloc[:,1],label="2015")
# plt.plot(temp_2016['date'],temp_2016.iloc[:,1],label="2016")
# plt.plot(temp_2017['date'],temp_2017.iloc[:,1],label="2017")
# plt.ylabel('Number of stores open', fontsize=12)
# plt.xlabel('Time', fontsize=12)
# plt.title('Number of stores open', fontsize=15)
# plt.xticks(rotation='vertical')
# plt.legend(['2013', '2014', '2015', '2016'], loc='lower right')
# """各年每天的商店数量作图"""
# plt.subplot(212)
# plt.plot(temp_2013.index,temp_2013.iloc[:,1],label="2013")
# plt.plot(temp_2014.index,temp_2014.iloc[:,1],label="2014")
# plt.plot(temp_2015.index,temp_2015.iloc[:,1],label="2015")
# plt.plot(temp_2016.index,temp_2016.iloc[:,1],label="2016")
# plt.plot(temp_2017.index,temp_2017.iloc[:,1],label="2017")
# plt.ylabel('Number of stores open', fontsize=12)
# plt.xlabel('Day of year', fontsize=12)
# plt.title('Number of stores open', fontsize=15)
# plt.xticks(rotation='vertical')
# plt.legend(['2013', '2014', '2015', '2016'], loc='lower right')
# plt.show()
#
#
# """对商店聚类，找到该店第一次交易时间(开张时间),并作图"""
# temp=transactions.groupby(['store_nbr']).agg({'date':[np.min,np.max]}).reset_index()
# temp['store_age']=temp['date']['amax']-temp['date']['amin']
# temp['open_year']=temp['date']['amin'].dt.year
# data=temp['open_year'].value_counts()
# #print(data)
# plt.figure(figsize=(12,4))
# sns.barplot(data.index,data.values, alpha=0.8, color=color[0])
# plt.ylabel('Stores', fontsize=12)
# plt.xlabel('Store opening Year', fontsize=12)
# plt.title('When were the stores started?', fontsize=15)
# plt.xticks(rotation='vertical')
# plt.show()
#
#
# """
# items数据集,描述商品信息
# item_nbr(商品编号)、family(大类)、class(类型)、perishable(是否易腐烂，二分类数据类型)
# """
# """sale_store_item_level描述了某商品某商店某年的销售量,与items 、stores、进行merge"""
# store_items=pd.merge(sale_store_item_level,items,on='item_nbr')
# store_items=pd.merge(store_items,stores,on='store_nbr')
# store_items['item_sales']=store_items['item_sales']
# # top selling items by store type
# top_items_by_type=store_items.groupby(['type','item_nbr'])['item_sales'].sum()
# top_items_by_type=top_items_by_type.reset_index().sort_values(['type','item_sales'],ascending=[True,False])
# # top selling item class by store type
# top_class_by_type=store_items.groupby(['type','class'])['item_sales'].sum()
# top_class_by_type=top_class_by_type.reset_index().sort_values(['type','item_sales'],ascending=[True,False])
# # top selling item family by store type
# top_family_by_type=store_items.groupby(['type','family'])['item_sales'].sum()
# top_family_by_type=top_family_by_type.reset_index().sort_values(['type','item_sales'],ascending=[True,False])
#
#
# """Top 5 item families across different store types"""
# plt.figure(figsize=(12,5))
# x=top_family_by_type.pivot(index='type',columns='family')
# x.plot.bar(stacked=True,figsize=(12,5))
# y=x.columns.droplevel(0).values
# plt.ylabel('Sales', fontsize=12)
# plt.xlabel('Top 5 item families', fontsize=12)
# plt.title('Top 5 item families across different store types', fontsize=15)
# plt.xticks(rotation='vertical')
# plt.legend(y)
# plt.show()
#
#
# """Top 5 item classes across different store types"""
# plt.figure(figsize=(12,5))
# x=top_class_by_type.pivot(index='type',columns='class')
# x.plot.bar(stacked=True,figsize=(12,5))
# y=x.columns.droplevel(0).values
# plt.ylabel('Sales', fontsize=12)
# plt.xlabel('Top 5 item classes', fontsize=12)
# plt.title('Top 5 item classes across different store types', fontsize=15)
# plt.xticks(rotation='vertical')
# plt.legend(y)
# plt.show()
#
#
# """Top 5 items across different store types"""
# plt.figure(figsize=(12,5))
# x=top_items_by_type.pivot(index='type',columns='item_nbr')
# x.plot.bar(stacked=True,figsize=(12,5))
# y=x.columns.droplevel(0).values
# plt.ylabel('Sales', fontsize=12)
# plt.xlabel('Top 5 items ', fontsize=12)
# plt.title('Top 5 items across different store types', fontsize=15)
# plt.xticks(rotation='vertical')
# plt.legend(y)
# plt.show()
#
#
# """Performance of Item families across stores of different type"""
# top_family_by_type=store_items.groupby(['type','family'])['item_sales'].sum()
# top_family_by_type=top_family_by_type.reset_index().sort_values(['type','item_sales'],ascending=[True,False])
# x=top_family_by_type.pivot(index='family',columns='type')
# cm = sns.light_palette("orange", as_cmap=True)
# x = x.style.background_gradient(cmap=cm)
#
#
# """Performance of Item class across stores of different type"""
# top_class_by_type=store_items.groupby(['type','class'])['item_sales'].sum()
# top_class_by_type=top_class_by_type.reset_index().sort_values(['type','item_sales'],ascending=[True,False])
# top_class_by_type=top_class_by_type.groupby(['class']).head(20)
# x=top_class_by_type.pivot(index='class',columns='type')
# x['total']=x.sum(axis=1)
# x=x.sort_values('total',ascending=False)
# del(x['total'])
# x=x.head(20)
# cm = sns.light_palette("gray", as_cmap=True)
# x = x.style.background_gradient(cmap=cm,axis=1)
#
#
# """Performance of Item item_nbr across stores of different type"""
# top_items_by_type=store_items.groupby(['type','item_nbr'])['item_sales'].sum()
# top_items_by_type=top_items_by_type.reset_index().sort_values(['type','item_sales'],ascending=[True,False])
# top_items_by_type=top_items_by_type.groupby(['item_nbr']).head(20)
# # print(top_items_by_type)
# x = top_items_by_type.pivot(index='item_nbr',columns='type')
# x['total']=x.sum(axis=1)
# x = x.sort_values('total',ascending=False)
# del(x['total'])
# x=x.head(30)
# cm = sns.light_palette("green", as_cmap=True)
# x = x.style.background_gradient(cmap=cm,axis=1)


"""
方法二：
"""
# """加载数据文件"""
# items = pd.read_csv("items.csv")
# holiday_events = pd.read_csv("holidays_events.csv")
# stores = pd.read_csv("stores.csv")
# oil = pd.read_csv("oil.csv")
# transactions = pd.read_csv("transactions.csv", parse_dates=['date'])
# # train 数据集较多，此处取前600w个样本，即6000000X6"
# train = pd.read_csv("train.csv", nrows=6000000, parse_dates=['date'])
#
#
# """
# 检查数据集各个属性是否有空值，打印格式如下：
# Nulls in holiday_events columns: ['date' 'type' 'locale' 'locale_name' 'description' 'transferred'] => [False False False False False False]
# 可以发现：仅有Oil数据集dcoilwtico属性有空值
# """
# print("Nulls in Oil columns: {0} => {1}".format(oil.columns.values,oil.isnull().any().values))
# print("Nulls in holiday_events columns: {0} => {1}".format(holiday_events.columns.values,holiday_events.isnull().any().values))
# print("Nulls in stores columns: {0} => {1}".format(stores.columns.values,stores.isnull().any().values))
# print("Nulls in transactions columns: {0} => {1}".format(transactions.columns.values,transactions.isnull().any().values))



"""
方法三：
"""
"""
This is an upgraded version of Ceshine's LGBM starter script, simply adding more
average features and weekly average features on it.
"""
from datetime import date, timedelta
import gc
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

# """
# 加载数据文件
# usecols:需要加载的column列
# dtype：每列及其指定的数据类型
# converters：列转换函数的字典
# skiprows:跳过的样本行数
# """
# df_train = pd.read_csv('train.csv', usecols=[1, 2, 3, 4, 5], dtype={'onpromotion': bool},
# 	converters={'unit_sales': lambda u: np.log1p(float(u)) if float(u) > 0 else 0},
#     parse_dates=["date"],
#     skiprows=range(1, 66458909)  # 2016-01-01   2017年及以后的样本数据
# )
#
# df_test = pd.read_csv(
#     "test.csv", usecols=[0, 1, 2, 3, 4], dtype={'onpromotion': bool}, parse_dates=["date"]).set_index(['store_nbr', 'item_nbr', 'date'])
#
# items = pd.read_csv(
#     "items.csv",
# ).set_index("item_nbr")
#
# stores = pd.read_csv(
#     "stores.csv",
# ).set_index("store_nbr")
#
#
# """
# one-hot编码
# """
# le = LabelEncoder()
# items['family'] = le.fit_transform(items['family'].values)
# stores['city'] = le.fit_transform(stores['city'].values)
# stores['state'] = le.fit_transform(stores['state'].values)
# stores['type'] = le.fit_transform(stores['type'].values)
#
# df_2017 = df_train.loc[df_train.date >= pd.datetime(2017,1,1)]
# del df_train
#
#
# """空值处理"""
# promo_2017_train = df_2017.set_index(["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(level=-1).fillna(False)
# promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
# promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
# promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
# promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
# promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
# del promo_2017_test, promo_2017_train
# print(promo_2017.head())
# """
# date                2017-01-01  2017-01-02  ...  2017-08-30  2017-08-31
# store_nbr item_nbr                          ...
# 1         96995          False       False  ...       False       False
#           99197          False       False  ...       False       False
#           103520         False       False  ...       False       False
#           103665         False       False  ...       False       False
#           105574         False       False  ...       False       False
#
# [5 rows x 243 columns]
# """
#
#
# """更新列名"""
# df_2017 = df_2017.set_index(["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(level=-1).fillna(0)
# df_2017.columns = df_2017.columns.get_level_values(1)   # 获取二级column名称
# items = items.reindex(df_2017.index.get_level_values(1))
# stores = stores.reindex(df_2017.index.get_level_values(0))
# print(df_2017.head())
# """
# date                2017-01-01  2017-01-02  ...  2017-08-14  2017-08-15
# store_nbr item_nbr                          ...
# 1         96995            0.0    0.000000  ...    0.000000    0.000000
#           99197            0.0    0.000000  ...    0.000000    0.000000
#           103520           0.0    0.693147  ...    0.000000    0.000000
#           103665           0.0    0.000000  ...    0.693147    0.693147
#           105574           0.0    0.000000  ...    1.386294    1.609438
#
# [5 rows x 227 columns]
# """
#
#
# """根据item_nbr分组"""
# df_2017_item = df_2017.groupby('item_nbr')[df_2017.columns].sum()
# promo_2017_item = promo_2017.groupby('item_nbr')[promo_2017.columns].sum()
# print(df_2017_item.head())
# """
# date      2017-01-01  2017-01-02  ...  2017-08-14  2017-08-15
# item_nbr                          ...
# 96995       0.000000    0.000000  ...    7.167038    7.742402
# 99197       0.693147   17.422746  ...    0.000000    0.000000
# 103501      0.000000   55.868320  ...   34.773173   35.512841
# 103520      0.000000   38.875486  ...   33.798042   40.030669
# 103665      2.079442   56.225402  ...   34.262348   35.741351
#
# [5 rows x 227 columns]
# """
#
#
# print(promo_2017_item.head())
# """
# date      2017-01-01  2017-01-02  ...  2017-08-30  2017-08-31
# item_nbr                          ...
# 96995            0.0         0.0  ...         1.0         0.0
# 99197            0.0         0.0  ...         0.0         0.0
# 103501           0.0         0.0  ...         3.0         1.0
# 103520           0.0         0.0  ...         1.0         1.0
# 103665           0.0         0.0  ...         1.0         0.0
#
# [5 rows x 243 columns]
# """
#
#
# """根据class、store_nbr聚合，每个商店每种类型商品情况"""
# df_2017_store_class = df_2017.reset_index()
# df_2017_store_class['class'] = items['class'].values
# df_2017_store_class_index = df_2017_store_class[['class', 'store_nbr']]
# df_2017_store_class = df_2017_store_class.groupby(['class', 'store_nbr'])[df_2017.columns].sum()
#
# df_2017_promo_store_class = promo_2017.reset_index()
# df_2017_promo_store_class['class'] = items['class'].values
# df_2017_promo_store_class_index = df_2017_promo_store_class[['class', 'store_nbr']]
# df_2017_promo_store_class = df_2017_promo_store_class.groupby(['class', 'store_nbr'])[promo_2017.columns].sum()


"""
方法四：
"""
"""建模"""
import subprocess
from subprocess import check_output
import gc
from datetime import date, timedelta
from sklearn.metrics import mean_squared_error
import xgboost as xgb


"""
加载文件
usecols:保留的列序号
dtype：指定属性类型
converters：属性进行一定的操作
parse_dates：指定需要转为datatime的属性
skiprows：跳过的样本行
"""
df_train = pd.read_csv(
    'train.csv', usecols=[1, 2, 3, 4, 5],
    dtype={'onpromotion': bool},
    converters={'unit_sales': lambda u: np.log1p(
        float(u)) if float(u) > 0 else 0},
    parse_dates=["date"],
    skiprows=range(1, 66458909)  # 2016-01-01
)
"""
        date  store_nbr  item_nbr  unit_sales  onpromotion
0 2016-01-01         25    105574    2.564949        False
1 2016-01-01         25    105575    2.302585        False
2 2016-01-01         25    105857    1.386294        False
3 2016-01-01         25    108634    1.386294        False
4 2016-01-01         25    108701    1.098612         True
"""


df_test = pd.read_csv(
    "test.csv", usecols=[0, 1, 2, 3, 4],
    dtype={'onpromotion': bool},
    parse_dates=["date"]
).set_index(['store_nbr', 'item_nbr', 'date'])   # 将三个属性columns置为索引index
"""
                                      id  onpromotion
store_nbr item_nbr date                              
1         96995    2017-08-16  125497040        False
          99197    2017-08-16  125497041        False
          103501   2017-08-16  125497042        False
          103520   2017-08-16  125497043        False
          103665   2017-08-16  125497044        False
"""


items = pd.read_csv(
    "items.csv",
).set_index("item_nbr")


# 筛选2017年后的数据df_2017
df_2017 = df_train.loc[df_train.date >= pd.datetime(2017, 1, 1)]
del df_train


promo_2017_train = df_2017.set_index(["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(level=-1).fillna(False)  # 将onpromotion元素变为columns
"""
                    onpromotion             ...                      
date                2017-01-01 2017-01-02  ... 2017-08-14 2017-08-15
store_nbr item_nbr                         ...                      
1         96995          False      False  ...      False      False
          99197          False      False  ...      False      False
          103520         False      False  ...      False      False
          103665         False      False  ...      False      False
          105574         False      False  ...      False      False
"""


promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
"""
DatetimeIndex(['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04',
               '2017-01-05', '2017-01-06', '2017-01-07', '2017-01-08',
               '2017-01-09', '2017-01-10',
               ...
               '2017-08-06', '2017-08-07', '2017-08-08', '2017-08-09',
               '2017-08-10', '2017-08-11', '2017-08-12', '2017-08-13',
               '2017-08-14', '2017-08-15'],
              dtype='datetime64[ns]', name='date', length=227, freq=None)
"""


promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
"""
                    onpromotion             ...                      
date                2017-08-16 2017-08-17  ... 2017-08-30 2017-08-31
store_nbr item_nbr                         ...                      
1         96995          False      False  ...      False      False
          99197          False      False  ...      False      False
          103501         False      False  ...      False      False
          103520         False      False  ...      False      False
          103665         False      False  ...      False      False
"""


promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
"""
DatetimeIndex(['2017-08-16', '2017-08-17', '2017-08-18', '2017-08-19',
               '2017-08-20', '2017-08-21', '2017-08-22', '2017-08-23',
               '2017-08-24', '2017-08-25', '2017-08-26', '2017-08-27',
               '2017-08-28', '2017-08-29', '2017-08-30', '2017-08-31'],
              dtype='datetime64[ns]', name='date', freq=None)
"""


promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
"""
date                2017-08-16  2017-08-17  ...  2017-08-30  2017-08-31
store_nbr item_nbr                          ...                        
1         96995          False       False  ...       False       False
          99197          False       False  ...       False       False
          103520         False       False  ...       False       False
          103665         False       False  ...       False       False
          105574         False       False  ...       False       False
"""


promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
del promo_2017_test, promo_2017_train


df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
    level=-1).fillna(0)
"""
shape(167515, 227)
                    unit_sales             ...                      
date               2017-01-01 2017-01-02  ... 2017-08-14 2017-08-15
store_nbr item_nbr                        ...                      
1         96995           0.0   0.000000  ...   0.000000   0.000000
          99197           0.0   0.000000  ...   0.000000   0.000000
          103520          0.0   0.693147  ...   0.000000   0.000000
          103665          0.0   0.000000  ...   0.693147   0.693147
          105574          0.0   0.000000  ...   1.386294   1.609438
"""


df_2017.columns = df_2017.columns.get_level_values(1)
"""
DatetimeIndex(['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04',
               '2017-01-05', '2017-01-06', '2017-01-07', '2017-01-08',
               '2017-01-09', '2017-01-10',
               ...
               '2017-08-06', '2017-08-07', '2017-08-08', '2017-08-09',
               '2017-08-10', '2017-08-11', '2017-08-12', '2017-08-13',
               '2017-08-14', '2017-08-15'],
              dtype='datetime64[ns]', name='date', length=227, freq=None)
"""


items = items.reindex(df_2017.index.get_level_values(1))
"""
                 family  class  perishable
item_nbr                                 
96995        GROCERY I   1093           0
99197        GROCERY I   1067           0
103520       GROCERY I   1028           0
103665    BREAD/BAKERY   2712           1
105574       GROCERY I   1045           0
"""


def get_timespan(df, dt, minus, periods, freq='D'):
    """
    筛选出数据集df中满足一定时间规律的时间字段（返回数据集样本数量不变，列变少），例如：
    get_timespan(df_2017, t2017, 3, 3)，筛选出满足pd.date_range(t2017 - 3, periods=3, freq=3)时间列
    """
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]


def prepare_dataset(t2017, is_train=True):
    """
    基于df_2017、promo_2017数据集，统计各个时间间隔（1、3、7、等）样本下的统计值（除了mean，还有std、median等），例如以下字段，作为训练集train（行m为各个商品id）
    """
    X = pd.DataFrame({
        "day_1_2017": get_timespan(df_2017, t2017, 1, 1).values.ravel(),
        "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
        "mean_7_2017": get_timespan(df_2017, t2017, 7, 7).mean(axis=1).values,
        "mean_14_2017": get_timespan(df_2017, t2017, 14, 14).mean(axis=1).values,
        "mean_30_2017": get_timespan(df_2017, t2017, 30, 30).mean(axis=1).values,
        "mean_60_2017": get_timespan(df_2017, t2017, 60, 60).mean(axis=1).values,
        "mean_140_2017": get_timespan(df_2017, t2017, 140, 140).mean(axis=1).values,
        "promo_14_2017": get_timespan(promo_2017, t2017, 14, 14).sum(axis=1).values,
        "promo_60_2017": get_timespan(promo_2017, t2017, 60, 60).sum(axis=1).values,
        "promo_140_2017": get_timespan(promo_2017, t2017, 140, 140).sum(axis=1).values
    })
    for i in range(7):
        X['mean_4_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 28 - i, 4, freq='7D').mean(axis=1).values
        X['mean_20_dow{}_2017'.format(i)] = get_timespan(df_2017, t2017, 140 - i, 20, freq='7D').mean(axis=1).values
    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[
            t2017 + timedelta(days=i)].values.astype(np.uint8)
    if is_train:
        y = df_2017[
            pd.date_range(t2017, periods=16)
        ].values
        return X, y
    return X


print("Preparing dataset...")
t2017 = date(2017, 5, 31)
X_l, y_l = [], []
for i in range(6):
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(t2017 + delta)
    X_l.append(X_tmp)
    y_l.append(y_tmp)
X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l
X_val, y_val = prepare_dataset(date(2017, 7, 26))
X_test = prepare_dataset(date(2017, 8, 16), is_train=False)

print("Training and predicting models...")

param = {}
param['objective'] = 'reg:linear'
param['eta'] = 0.5
param['max_depth'] = 3
param['silent'] = 1
param['eval_metric'] = 'rmse'
param['min_child_weight'] = 5
param['subsample'] = 0.8
param['colsample_bytree'] = 0.7
param['seed'] = 137
num_rounds = 157

plst = list(param.items())

MAX_ROUNDS = 157
val_pred = []
test_pred = []
cate_vars = []

dtest = xgb.DMatrix(X_test)   # 测试集dtest
for i in range(16):
    print("=" * 50)
    print("Step %d" % (i + 1))
    print("=" * 50)
    # 训练集dtrain，包含x、y
    dtrain = xgb.DMatrix(
        X_train, label=y_train[:, i],
        weight=pd.concat([items["perishable"]] * 6) * 0.25 + 1
    )
    # 验证集dval，包含x、y
    dval = xgb.DMatrix(
        X_val, label=y_val[:, i],
        weight=items["perishable"] * 0.25 + 1)

    watchlist = [(dtrain, 'train'), (dval, 'val')]
    model = xgb.train(plst, dtrain, num_rounds, watchlist, early_stopping_rounds=50, verbose_eval=50)

    val_pred.append(model.predict(dval))
    test_pred.append(model.predict(dtest))

print("Validation mse:", mean_squared_error(
    y_val, np.array(val_pred).transpose()))

print("Making submission...")
y_test = np.array(test_pred).transpose()
df_preds = pd.DataFrame(
    y_test, index=df_2017.index,
    columns=pd.date_range("2017-08-16", periods=16)
).stack().to_frame("unit_sales")
df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
submission.to_csv('xgb.csv', float_format='%.4f', index=None)


# """
# favorita-grocery-sales-forecasting
# """
# """读取数据集"""
# """
# 处理后的sales数据集包含date、store_nbr、transactions、city、state、type、cluster、year、week、day、dayoff属性
# """
# sales = pd.read_csv('../input/transactions.csv')
# stores = pd.read_csv('../input/stores.csv')
# stores.type = stores.type.astype('category')  # 指定type属性的数据类型
# sales = pd.merge(sales, stores, how='left')
# holidays = pd.read_csv('../input/holidays_events.csv')
# """对transactions、holidays_events数据集的date属性进行格式化"""
# sales['date'] = sales.date.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
# holidays['date'] = holidays.date.apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
# """对holidays数据集依据type属性分离为俩部分(type值是否为Event)"""
# events = holidays.loc[holidays.type == 'Event']
# holidays = holidays.loc[holidays.type != 'Event']
# """对sales数据集由date属性分离出 year, week and day"""
# sales['year'], sales['week'], sales['day'] = list(zip(*sales.date.apply(lambda x: x.isocalendar())))
# """对sales数据集新建布尔属性dayoff，标记样本日期是否为周末"""
# sales['dayoff'] = [x in [6, 7] for x in sales.day]
# """对dayoff属性进行修正 Adjuusting this variable to show all holidays"""
# for (d, t, l, n) in zip(holidays.date, holidays.type, holidays.locale, holidays.locale_name):
#     if t != 'Work Day':
#         if l == 'National':
#             sales.loc[sales.date == d, 'dayoff'] = True
#         elif l == 'Regional':
#             sales.loc[(sales.date == d) & (sales.state == n), 'dayoff'] = True
#         else:
#             sales.loc[(sales.date == d) & (sales.city == n), 'dayoff'] = True
#     else:
#         sales.loc[(sales.date == d), 'dayoff'] = False
#
# """store_nbr为47的商店，每日date交易量transactions，作图"""
# ts = sales.loc[sales['store_nbr'] == 47, ['date', 'transactions']].set_index('date')
# ts = ts.transactions.astype('float')
# plt.figure(figsize=(12, 12))
# plt.title('Daily transactions in store #47')
# plt.xlabel('time')
# plt.ylabel('Number of transactions')
# plt.plot(ts)
#
# """时间周期为30天的交易量transactions的平均值、标准差，随时间date的变化曲线"""
# plt.figure(figsize=(12, 12))
# plt.plot(ts.rolling(window=30, center=False).mean(), label='Rolling Mean')
# plt.plot(ts.rolling(window=30, center=False).std(), label='Rolling sd')
# plt.legend()
#
#
# from pandas.tools.plotting import autocorrelation_plot
# import statsmodels.tsa.stattools.adfuller
# import statsmodels.api as sm
#
#
# """
# 相关性一般是指两个变量之间的统计关联性，自相关性则是指一个时间序列的两个不同时间点的变量是否相关联；
# 时间序列具有自相关性是我们能够进行分析的前提，若时间序列的自相关性为0，也就是说各个时点的变量不相互关联，那么未来与现在和过去就没有联系；
# 时间序列的自相关性一般用时间序列的自协方差函数、自相关系数函数ACF和偏自相关系数函数PACF等统计量来衡量(即两个时点变量的协方差)
# 时间序列的平稳性，简单理解是时间序列的基本特性维持不变，换句话说，平稳性就是要求由样本时间序列所得到的曲线在未来的一段时期内仍能沿着现有的形态持续下去。单位跟检验衡量。
# 常见的单位根检验方法有DF检验（Dickey-Fuller Test）、ADF检验（AuGMENTED Dickey-Fuller Test）和PP检验（Phillips-Perron Test）
# ADF检验：
# 当一个自回归过程中：y_{t} = by_{t-1} + a + epsilon_{t} ，如果滞后项系数b为1，就称为单位根。当单位根存在时，自变量和因变量之间的关系具有欺骗性，因为残差序列的任何误差都不会随着样本量（即时期数）增大而衰减，也就是说模型中的残差的影响是永久的。这种回归又称作伪回归。如果单位根存在，这个过程就是一个随机漫步（random walk）。
# ADF检验就是判断序列是否存在单位根：如果序列平稳，就不存在单位根；否则，就会存在单位根。
# ADF检验的H0假设就是存在单位根，如果得到的显著性检验统计量小于三个置信度（10%，5%，1%），则对应有（90%，95，99%）的把握来拒绝原假设
# 如果时间序列不满足平稳性，需要对序列进行差分(np.diff(),后元素与前元素的差)、取对数log等处理，再进行平稳性检验
# """
#
#
# def test_stationarity(timeseries):
#     """
#     timeseries为数据集，含date、transactions俩属性
#     """
#     # Perform Dickey-Fuller test:
#     print('Results of Dickey-Fuller Test:')
#     # adfuller(timeseries)实现对timeseries的adf检验，输出6个值(其中第5个值为字典形式)；dftest[0]小于dftest[4],且dftest[1]接近0(至少要小于0.1)，则认为timeseries平稳
#     dftest = adfuller(timeseries, autolag='AIC')
#     dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
#     for key, value in dftest[4].items():
#         dfoutput['Critical Value (%s)' % key] = value
#     print(dfoutput)
#
#
# test_stationarity(ts)
#
#
# """
# 时间序列自相关性作图，y轴为时间序列的自相关性值
# """
# plt.figure(figsize=(12, 6))
# autocorrelation_plot(ts)
#
# plt.figure(figsize=(12, 6))
# autocorrelation_plot(ts)
# plt.xlim(xmax=100)           # 改变x轴范围
#
# plt.figure(figsize=(12, 6))
# autocorrelation_plot(ts)
# plt.xlim(xmax=10)
#
#
# """
# ARIMA模型即自回归移动平均模型，也记作ARIMA(p,d,q)，是统计模型(statistic model)中最常见的一种用来进行时间序列预测的模型。
# p--代表预测模型中采用的时序数据本身的滞后数(lags) ,也叫做AR/Auto-Regressive项
# d--代表时序数据需要进行几阶差分化，才是稳定的，也叫Integrated项。
# q--代表预测模型中采用的预测误差的滞后数(lags)，也叫做MA/Moving Average项
# 定参准则：
# 目前选择模型常用如下准则： （其中L为似然函数，k为参数数量，n为观察数）
# AIC = -2 ln(L) + 2 k 中文名字：赤池信息量 akaike information criterion
# BIC = -2 ln(L) + ln(n)*k 中文名字：贝叶斯信息量 bayesian information criterion
# HQ = -2 ln(L) + ln(ln(n))*k hannan-quinn criterion
# 我们常用的是AIC准则
# """
#
#
# """确定p、q参数"""
# result = sm.tsa.arma_order_select_ic(ts, max_ar=10, max_ma=10, ic='aic', trend='c', fit_kw=dict(method='css', maxiter=500))
# print('The aic prescribes these (p,q) parameters : {}'.format(result.aic_min_order))
# plt.figure(figsize=(12, 6))
# seaborn.heatmap(result.aic)
#
#
# """模型拟合"""
# pdq = (5, 0, 5)
# model = ARIMA(ts, order=pdq, freq='W')
# model_fit = model.fit(disp=False, method='css', maxiter=100)
# # plot residual errors
# residuals = pd.DataFrame(model_fit.resid)   # 残差,作图
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
# residuals.plot(ax=axes[0])
# residuals.plot(kind='kde', ax=axes[1])
#
#
# """预测与实际值作图"""
# plt.figure(figsize=(12, 6))
# plt.plot(ts)
# plt.plot(model_fit.fittedvalues,alpha=.7)
#
#
# """预测"""
# forecast_len = 30
# size = int(len(ts) - forecast_len)
# train, test = ts[0:size], ts[size:len(ts)]
# history = [x for x in train]
# predictions = list()
#
# print('Starting the ARIMA predictions...')
# print('\n')
# for t in range(len(test)):
#     model = ARIMA(history, order=pdq, freq='W')
#     model_fit = model.fit(disp=0)
#     output = model_fit.forecast()
#     yhat = output[0]
#     predictions.append(float(yhat))
#     obs = test[t]
#     history.append(obs)
# print('Predictions finished.\n')
#
# predictions_series = pd.Series(predictions, index=test.index)
#
#
# plt.figure(figsize=(12,12))
# plt.title('Store 47 : Transactions')
# plt.xlabel('Date')
# plt.ylabel('Transactions')
# plt.plot(ts[-2*forecast_len:], 'o', label='observed')
# plt.plot(predictions_series, '-o', label='rolling one-step out-of-sample forecast')
# plt.legend(loc='upper right')
#
#
# plt.figure(figsize=(12,12))
# x = abs(ts[-forecast_len:]-predictions_series)
# seaborn.distplot(x, norm_hist=False, rug=True, kde=False)
