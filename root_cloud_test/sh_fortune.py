"""
“上海收盘“是上交所股票从10年1月到16年8月31日每天的收盘价。我有一个投资策略就是
Step1 每个月1号卖掉当前所有股票；
Step2 用卖掉股票的钱买入上个月所有股票中收益表现最差的那10%的股票。譬如说上个月总共有783只股票，那么我会买入78（10% of 783）只表现最糟的股票。买入方式是等权重买入，将手上的钱等分成78份，每份钱买入一只股票；
Step3 持有一个月到下个月的1号，然后重复step1。
再假设在一开始，也就是10年1月1日，我怀揣100万入市，按照上面的投资策略，编程计算16年9月1日我的财富。（不考虑期间的任何交易费用或者涨跌停买不进卖不出这些外在限制。）

一些说明：
1.一开始是10年1月1日入市，无法评估上个月（09年12月）股票表现，因此实际上是10年2月1日开始进行step2，评估周期为10-1-1至10-1-31；
2.因股票交易涉及节假日休市，因此评估周期1日、31日可能用该月份第一个、最后一个交易日替代；
3.对于月中上市交易的股票，参与次月1日的评估，评估周期为上市首日至当月最后一个交易日；
4.交易与计算收益的周期略有不同，为持有股票该月1日至次月1日的价差，例如持有某只股票的2月收益：3-1价格减去2-1价格
5.全仓，市值即财富。手上的钱每个月1号进行交易都会变动
6.运行环境：python3.7

一些待改进：
1.计算上个月最差10%股票，可以并行计算
2.一些常数变量，可以单独列出来，方便以后更新
3.因缺少2016.9.1的数据，无法利用函数cal_fortune()计算8.1-9.1的收益,实际返回值是2016.8.1日的财富
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gc


def missing_values_table(df):
    """
    本函数用于统计数据集df的各属性空值数量及所占比例，返回mis_val_table_ren_columns
    :param df: 待统计空值的表格
    :return: 输入表格df的各属性的空值数量Missing Values与比率
    """
    mis_val = df.isnull().sum()     # 各个属性的空值数量
    mis_val_percent = 100 * df.isnull().sum() / len(df)  # 各个属性的空值比率
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)  # 创建空值表mis_val_table，包含各属性的空值数量mis_val与比率mis_val_percent
    mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'Missing Values', 1: '% of Total Values'})
    # 筛选出mis_val_table有空值的属性，并依据空值比排序，所有数据保留2位小数
    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values('% of Total Values', ascending=False).round(2)
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"+"There are " + str(mis_val_table_ren_columns.shape[0])+" columns that have missing values.")

    return mis_val_table_ren_columns


def cal_bad_stock(sh_data, cur_year, cur_month):
    """
    本函数计算并返回某个月（1月）收益最差的10%股票，评估周期为当月第一天至最后一天10-1-1至10-1-31
    输入：某个月（1月）所有股票数据集
    输出：前10%的股票名称
    step1：分离出某个月（1月）的所有股票数据集
    step2：删除上市不足2天的股票，并计算该月剩余股票数目，即total_stock_num
    step3：计算股票价差，收益率，排序，保存前10%股票名称
    """
    """筛选出目标月份的数据"""
    cur_month_data = sh_data.loc[(sh_data['YEAR'] == cur_year) & (sh_data['MONTH'] == cur_month), :]    # 筛选出cur_year、cur_month的数据集
    cur_month_data = cur_month_data.reset_index(drop=True)    # 对筛选后的样本索引重置，并删除原来的索引
    sale_day = cur_month_data.shape[0]  # 该月份可进行交易的天数

    """筛选并删除数据集中当月上市不足2天的股票"""
    missing_values_data = missing_values_table(cur_month_data)  # 计算数据集各股票的空值率
    missing_values_data = missing_values_data.reset_index().rename(columns={"index": "stock_code"})
    drop_cloumns = missing_values_data[missing_values_data['Missing Values'] > (sale_day - 2)].stock_code  # 统计该月上市不足2天的股票，不参与该月股票表现评估
    cur_month_data = cur_month_data.drop(columns=drop_cloumns.tolist(), axis=1)  # 将数据集sh_data中的部分stock_code删除
    print('delete these stock_code and do not assess in cur_month:', cur_month, ':month', drop_cloumns.tolist())
    total_stock_num = cur_month_data.shape[1] - 3  # 当月参与评估股票总数
    bad_stock_num = int(int(total_stock_num) / 10)

    """获取股票当月的首个交易日（上市日），最后一个交易日的价格 first_price、end_price"""
    first_valid_indices = cur_month_data.apply(lambda series: series.first_valid_index())  # 每个股票column，当月首个交易日的索引(在cur_month_data中的索引)
    # 创建first_price列表，保存股票的首个交易日价格
    first_price = []  # 保存股票的首个交易日价格
    i = 0
    for stock in first_valid_indices:
        first_price.append(cur_month_data.iloc[stock][list(first_valid_indices.index)[i]])  # series对象的index与value处理，获取股票首个交易日价格
        i += 1
    end_price = cur_month_data.iloc[-1]  # 保存股票最后一个交易日价格

    """依据first_price、end_price来创建一个DataFrame，计算当月股票收益率 percent"""
    # 新建df，包含first_price、end_price、percent 共3个columns，其中percent初始化为0
    percent_df = pd.DataFrame(list(zip(first_price, end_price.tolist(), [0 for i in range(len(first_price))])), index=end_price.index, columns=['first_price', 'end_price', 'percent'])
    percent_df['percent'] = percent_df.apply(lambda x: (x['end_price'] - x['first_price'])/x['first_price'], axis=1)   # 计算收益率，更新percent列
    percent_df = percent_df.drop(index=['YEAR', 'MONTH', 'DAY'])           # 删除三个日期属性，不参与收益率排序
    bad_stock_df = percent_df.sort_values(by="percent", ascending=True).iloc[:int(bad_stock_num)]['percent']    # 对收益率排序，截取前10%股票的收益率

    return bad_stock_df   # series对象


def cal_fortune(sh_data, cur_bad_stock, cur_fortune, cur_year, cur_month):
    """
    本函数计算某个月1日（2月1日），用手上的钱买入某些股票，计算这些股票至下个月1日（2-1到3-1）的收益率，并更新返回手上的钱
    输入：某个月1日（2月1日）手上的钱，某个月（1月）收益最差前10%股票名称
    输出：下个月1日（3-1）手上的钱
    step1：分离出前10%股票的对应某个月1日（2-1）、下个月1日（3-1）数据集
    step2 计算这些股票的收益率
    step3：计算这些股票的当前市值，求和并返回
    """
    """年月代号每逢12一个循环"""
    if cur_month == 12:
        next_month = 1
        next_year = cur_year + 1
    else:
        next_month = cur_month + 1
        next_year = cur_year

    """上个月10%股票的总数 bad_stock_num"""
    bad_stock_num = len(cur_bad_stock)
    print('number of cur_bad_stock is:', bad_stock_num)

    """依据 cur_bad_stock 筛选出这些股票的当月价格数据cur_month_data，下月价格数据next_month_data，当月首个交易日数据cur_month_price，下月首个交易日数据next_month_price"""
    cur_month_data = sh_data.loc[(sh_data['YEAR'] == cur_year) & (sh_data['MONTH'] == cur_month), cur_bad_stock.index]
    cur_month_data = cur_month_data.reset_index(drop=True)  # 对筛选后的样本索引重置，并删除原来的索引
    cur_month_price = cur_month_data.iloc[0]     # 当前月份首个交易日价格

    next_month_data = sh_data.loc[(sh_data['YEAR'] == next_year) & (sh_data['MONTH'] == next_month), cur_bad_stock.index]
    next_month_data = next_month_data.reset_index(drop=True)  # 对筛选后的样本索引重置，并删除原来的索引
    next_month_price = next_month_data.iloc[0]    # 下个月份首个交易日价格

    """依据 cur_month_price、next_month_price 来创建一个DataFrame，计算当月股票收益率 percent"""
    percent_df = pd.DataFrame(list(zip(cur_month_price.tolist(), next_month_price.tolist(), [0 for i in range(bad_stock_num)])), index=cur_month_price.index, columns=['cur_month_price', 'next_month_price', 'percent'])
    # 计算并更新买入的这些股票收益率
    percent_df['percent'] = percent_df.apply(lambda x: (x['next_month_price'] - x['cur_month_price']) / x['cur_month_price'], axis=1)  # 计算收益率
    # 所有交易的股票收益率汇总
    sum = 0
    for every_per in percent_df['percent']:
        sum += (1 / bad_stock_num) * (1 + every_per)
    # 计算并更新财富 new_fortune
    new_fortune = sum * cur_fortune
    print('today is the first day of month, and i have new_fortune is:', next_month, new_fortune)

    return new_fortune


if __name__ == "__main__":
    """读取数据集 上海收盘.csv"""
    sh_data = pd.read_csv("上海收盘.csv")
    print('there are %d days and %d stock.' % (sh_data.shape[0], sh_data.shape[1] - 1))

    """
    对于收盘价统计不足20个交易日，1个月的股票（即空值率在1-1.23%以上），删除不参与计算；
    清理内存占用
    因价格在“分”之后无意义，且节省资源，将数据保留两位小数；
    """
    missing_values_data = missing_values_table(sh_data)    # 计算数据集各股票的空值率
    missing_values_data = missing_values_data.reset_index().rename(columns={"index": "stock_code"})    # 将index转为columns，并重命名为stock_code
    drop_cloumns = missing_values_data[missing_values_data['% of Total Values'] > 98.77].stock_code    # 统计所有空值率在98.77%以上的股票stock_code
    sh_data = sh_data.drop(columns=drop_cloumns.tolist(), axis=1)           # 将数据集sh_data中的部分stock_code删除
    print('delete these stock_code and do not use:', drop_cloumns.tolist())

    del missing_values_data
    gc.collect()

    for col in sh_data.columns:
        if col != 'Date':
            sh_data[col] = round(sh_data[col], 2)

    """ 新建特征YEAR、MONTH、DAY 替代Date列"""
    sh_data['Date'] = pd.to_datetime(sh_data['Date'])
    sh_data['YEAR'] = sh_data['Date'].dt.year
    sh_data['MONTH'] = sh_data['Date'].dt.month
    sh_data['DAY'] = sh_data['Date'].dt.day
    sh_data = sh_data.drop(columns=['Date'], axis=1)
    print('there are %d days and %d stock.' % (sh_data.shape[0], sh_data.shape[1] - 3))

    # 初始2010.2.1时候的财富
    fortune_list = [1000000, ]

    for year in [2010, 2011, 2012, 2013, 2014, 2015, 2016]:
        for month in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            # 计算上个月表现最差的10%股票
            cur_bad_stock = cal_bad_stock(sh_data, year, month)
            # 计算下个月的年月，即new_year，new_month
            new_month = month + 1
            new_year = year
            if new_month == 13:
                new_month = 1
                new_year += 1
            # 提前退出循环
            # 说明：month=7时，new_month=8，此时因缺少2016.9.1的数据，无法利用函数cal_fortune()计算8.1-9.1的收益
            # 因此，本代码实际返回的fortune_list[-1]是2016.8.1收盘，时候的财富
            if year == 2016 and month == 7:
                break
            # 计算下个月的财富
            fortune_list.append(cal_fortune(sh_data, cur_bad_stock, fortune_list[-1], new_year, new_month))

    print('fortune of first day in every month:', fortune_list)


""""
# 作图查看财富变化,15年上半年财富增长较快

fortune_list = [1000000, 1091145.972689899, 1111466.9432215851, 1045555.2586849908, 935979.7230415356, 849720.7535573741, 1037055.3667366928, 1067371.2226745407, 1088293.654978709, 1169221.2700703528, 1210355.235514888, 1251629.3293522205, 1184902.5656766144, 1336823.3439303231, 1358764.897163917, 1307294.8177241809, 1211620.0935319872, 1278322.869916539, 1333960.5018545557, 1254713.9639802875, 1102079.5201251663, 1172412.3841820701, 1130110.952430717, 917740.3474998343, 998135.9468941973, 1140412.8362691202, 1117585.3850068937, 1190183.25620815, 1200716.8929802878, 1099389.3511419117, 1014679.820710741, 1065441.0056940103, 1085094.0994395616, 1101727.2847446194, 912395.3514406946, 1143818.0313662172, 1211720.218575048, 1232195.8870015447, 1165743.3088272572, 1123170.185405596, 1297068.0259876444, 1134311.240871204, 1201203.5511731836, 1303169.0534008194, 1387899.1436412316, 1342969.2578968767, 1367940.1431235, 1349906.3212856892, 1304961.4235671547, 1365517.1363975315, 1320171.1255845185, 1283492.7660601488, 1300549.329576549, 1346647.7232649636, 1463580.0253855265, 1587999.4561875537, 1721790.927604484, 1728035.3273370354, 1775631.283292288, 1742631.321749758, 1911985.0031677196, 2074789.6918194047, 2402563.543956564, 2850713.2317285785, 3646702.9394865683, 3324905.463127212, 3036588.993747496, 2656241.379528613, 2950987.003245952, 3304950.6245001582, 3661505.689061112, 3461389.6467318614, 2641285.6452628323, 2732294.738256931, 3186177.698603438, 3237179.5898029394, 3089551.073978055, 3164045.0888618757, 3127978.495318938]
sns.lineplot(x=range(len(fortune_list)), y=fortune_list)
plt.show()
"""
