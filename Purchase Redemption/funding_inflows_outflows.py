"""
资金流入流出预测-挑战Baseline：依据用户申购赎回数据，预测未来一个月之中，每天的所有用户申购赎回总量，评估标准为各天申购赎回总量误差之和.
data：
用户信息表 user_profile_table，含user_id、Sex、City、constellation字段
用户申购赎回数据表 user_balance_table。含 20130701 至 20140831 申购和赎回信息
余额宝在 14 个月内的收益率表： mfd_day_share_interest
银行间拆借利率表是 14 个月期间银行之间的拆借利率（皆为年化利率）： mfd_bank_shibor
余额宝收益方式：存入与收益计算时间
"""
"""
时序预测分析的STL分解法、Prophet法研究
分解法：
乘法模型：Y=T*S*C*I
1.用 MA=T×C 分析长期趋势和循环变动（MA统计周期为4个季度）；
2.用 X/MA=S*I 分析季节性
3.对季节取平均，来消除I
4.分析长期趋势T，然后根据 MA/T=C 来分析周期
改进之处：
1.MA居中放置
2.对每月进行30天修正，淘汰数据中的异常值等
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



"""读取数据"""
user_balance = pd.read_csv('user_balance_table.csv', usecols=[0, 1, 2, 3, 4, 8], parse_dates=["report_date"])
# interest = pd.read_csv('mfd_day_share_interest.csv', parse_dates=["mfd_date"])


"""选取的'user_id' 'report_date' 'tBalance' 'yBalance' 'total_purchase_amt' 'total_redeem_amt'字段均无空值"""
# print("Nulls in user_balance columns: {0} => {1}".format(user_balance.columns.values, user_balance.isnull().any().values))


"""绘制 mfd_7daily_yield、mfd_daily_yield 随时间变化散点图"""
# # 绘制mfd_daily_yield散点图ax1
# ax1 = sns.scatterplot(x='mfd_date', y='mfd_daily_yield', data=interest)
# ax1.set_xlim('2013-07-01', '2014-08-31')
# # 绘制mfd_7daily_yield散点图ax2，与ax1共用x坐标
# ax2 = ax1.twinx()
# ax2 = sns.scatterplot(x='mfd_date', y='mfd_7daily_yield', data=interest)
# plt.show()


"""************按日期聚合，查看每个聚合日期的total_purchase_amt量、total_redeem_amt量的统计值*********************"""
"""
按日期聚合，查看每个聚合日期的total_purchase_amt量、total_redeem_amt量的统计值，形式如下：
            total_purchase_amt       total_redeem_amt      
                           sum count              sum count
report_date                                                
2013-07-01            32488348   441          5525022   441
2013-07-02            29037390   480          2554548   480
2013-07-03            27270770   499          5953867   499
2013-07-04            18321185   523          6410729   523
2013-07-05            11648749   544          2763587   544
"""
report_date_level = user_balance.groupby(['report_date'], as_index=False)['total_purchase_amt', 'total_redeem_amt'].agg(['sum', 'count'])
"""
将分级字段合并,保存在col_names列表中，原始索引report_date_level.columns如下：
MultiIndex([('total_purchase_amt',   'sum'),
            ('total_purchase_amt', 'count'),
            (  'total_redeem_amt',   'sum'),
            (  'total_redeem_amt', 'count')],
           )
"""
col_names = []
for i in report_date_level.columns:
    i = list(i)
    col_names.append(''.join(i))
"""
重置columns，索引report_date由index转为column：
"""
# 依据col_names重命名columns
report_date_level.columns = col_names
# 将索引（report_date）转为column，并删除原索引
report_date_level.reset_index(inplace=True)


"""绘图，每个日期的purchase_sum、redeem_sum随时间变化曲线，
可以发现2014-2及以前的数据波动较大，2014-3及以后的数据周期平稳波动，且买卖数据趋势跟随，故选取2014-3——2014-8共六个月的数据"""
# # 首先调整图像大小尺寸
# plt.figure(figsize=(25, 4))
# report_date_level['total_purchase_amtsum'].plot()
# report_date_level['total_redeem_amtsum'].plot()
# plt.show()


"""数据预处理：截取2014-03-01之后的数据分析，并3、5、7、8这四个月份，将30、31日数据取平均赋值给30日，删除31日数据"""
# 截取数据，并删除重置顺序索引
report_date_level_3to8 = report_date_level.loc[report_date_level['report_date'] >= '2014-03-01'].reset_index(drop=True)
# 将30、31日数据取平均赋值给30日，删除31日数据
process_month = [3, 5, 7, 8]
for month in process_month:
    # 筛选该月最后两天的数据，select_days
    select_days = report_date_level_3to8.loc[(report_date_level_3to8['report_date'].dt.day >= 30) & (report_date_level_3to8['report_date'].dt.month == month), :]
    # 最两天数据按列求平均值，tow_days_mean
    tow_days_mean = select_days.apply(lambda x: x.mean(), axis=0)
    # 将tow_days_mean的除’report_date‘字段以外的值赋值给select_days数据集的第一个样本
    select_days.iloc[:1, 1:] = tow_days_mean.tolist()[1:]
    # 将select_days数据集的第一个样本赋值给report_date_level_3to8，更新report_date_level_3to8数据集
    report_date_level_3to8.loc[(report_date_level_3to8['report_date'].dt.day == 30) & (report_date_level_3to8['report_date'].dt.month == month), :] = select_days.iloc[:1, :]
    # 删除该月最后一天的样本
    report_date_level_3to8 = report_date_level_3to8.drop(report_date_level_3to8[(report_date_level_3to8['report_date'].dt.day == 31) & (report_date_level_3to8['report_date'].dt.month == month)].index)
# 删除重置顺序索引,数据持久化
report_date_level_3to8 = report_date_level_3to8.reset_index(drop=True)
# report_date_level_3to8.to_csv("report_date_level_3to8.csv")


"""
依据 tBalance 属性，对用户进行分类（穷人富人）
绘图，每个日期的purchase_sum、redeem_sum随时间变化曲线,发现穷人的变化更规律，买卖量的变化一致性更好:不分开建模
"""
# 按用户 user_id 聚合，查看每个用户的tBalance量平均值
user_avg_Balance_level = user_balance.groupby(['user_id'], as_index=False)['tBalance'].agg(['mean'])
# drop index and rename
user_avg_Balance_level = user_avg_Balance_level.reset_index().rename(columns={'mean':'user_mean_balance'})
# 80%的财富掌握在20%的人手中，以平均余额值mean_balance前20% 为界，区分富人穷人，大约余额是33W
cut_balance = np.percentile(user_avg_Balance_level['user_mean_balance'], 80)
# 新建'is_rich'字段
user_avg_Balance_level['is_rich'] = user_avg_Balance_level['user_mean_balance'].apply(lambda x: 1 if x > cut_balance else 0)
# 数据持久化
# user_avg_Balance_level.to_csv("user_avg_Balance_level.csv")


"""节假日对收益计算的影响"""
"""月周期、周周期"""
"""具体算法实现，请移步huigui.py文件"""



































