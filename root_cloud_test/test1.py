"""
Test1 里面有200个csv，每个csv只有一行数据，描述的是一个波形图（数值介于-2到2之间）。每行数据的第一个值表示该行波形图有几类pattern原型，然后接着的几个自然数按顺序表示每类原型的样式（原型加入随机因素）出现的次数。
每个csv文件里的波形图都由电脑生成。生成方式如下：
Step1 生产一种原型，假如原型为(-0.4, 0.3, 0.2, 0.7)。
Step2 在原型上加入随机因素，生成实际样式（-0.4+a, 0.3+b, 0.2+c, 0.7+d），abcd为iid的随机数，将该实际样式写入数据。实际样式和原型模式的唯一区别在于加入的随机干扰。只有加入干扰后的实际样式才会写入数据中，而原型是不会直接写入数据的。
Step3 重复step2，每次的adcd都是重新生成的iid随机数，直到该原型以实际样式的形式出现7次。（因为第一种类型有7个）。
Step4 重复step1，直到4种原型模式及其相应个数的实际样式都写入数据中。
特征强调：
1.	每一种原型的实际样式是连续出现得。如果要出现6次，一定是紧挨着出现6次。
2.	不同的原型包含的数字个数可能是不一样的，也就是，长度可能是不一样的！有的原型含有7个数字，有的只含有3个。但是最多不超过20个数字。且这些数字严格介于-2和2之间。
3.	由同一个原型模式产生的不同实际样式，它们的长度是一样的（含有同样个数的数字），这个很关键！它们之间的不同仅在于受到了不同随机干扰。
这200个csv里的几百个原型都是不一样的，如有雷同，实属巧合。
根据每行数字的波形图（也就是跳过每行开头的几个自然数，而只考虑那些介于-2和2之间的数），编程统计其中有几种原型，每种原型有几个实际样式。

一些说明：
1.200个csv文件的原型类型介于3-5，每个原型出现的次数介于5-9
2.运行环境：python3.7
3.主要有两点：
  数据预处理：一阶差分，尝试移动平均线、移动加权回归效果一般
  第一：对不同原型的数据集进行拆分
  step1：对所有数据点进行自适应聚类，并对簇编号
        对比差分后的数据图、sse差值-K图，确定合适的delta_sse阈值，选择对应k
  step2：对数据集依次统计其所在簇编号
  step3：从前往后统计：前3个数据点对应簇出现至少5次，之后的数据点对应簇在前面出现过；直到新出现一个数据点对应簇在前面没出现过，则在此截断，一个原型结束
  第二：统计拆分后的数据集周期（大致有两种方法，此处用方法二）
  方法一（针对差分后的数据点）：
  step1：获取全部极值点(待确认是否需要删除一定邻域内的其他极值)
  step2：统计计算每个类的值的个数，即周期
  方法二（针对数据点对应簇的编号分析）：
  step1：统计各个簇的数据点个数
  step2：对各个簇依据数据点数目排序，删掉数目最多、最少的1-3个簇（过滤原型首尾的数据点）
  step3：对其他簇的数目求中位数（或者均值），即周期
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import gc
import os
import sys


def Concat_data(dir):
    """
    本函数是将文件夹dir下所有文件的数据进行汇总，生成一个新的.csv文件
    :param dir: 数据集所在文件夹
    :return: 汇总各.csv文件后的一个数据集
    """
    filename_list = []
    frames = []

    for root, dirs, files in os.walk(dir):
        for file in files:
            filename_list.append(file)
            df = pd.read_csv(os.path.join(root, file), header=None)
            frames.append(df)  # 将所有df更新进frame表中
    # 合并所有dataframe
    result = pd.concat(frames, axis=0)
    result.to_csv(dir + 'all.csv', index=False)
    print('~~~~~~~~~~~~~~~~DONE~~~~~~~~~~~~~~~~~~`')


def squeeze_nan(x):
    """
    本函数是将数据集x内的非空值左移，空值移到最右侧，不改变x的shape
    :param x: 数据集
    :return: 处理后的数据集
    """
    original_columns = x.index.tolist()
    squeezed = x.dropna()
    squeezed.index = [original_columns[n] for n in range(squeezed.count())]
    return squeezed.reindex(original_columns, fill_value=np.nan)


"""
数据预处理部分(本段代码只需执行1次，生成test1_filesall.csv、df_x_diff.csv文件)：
生成汇总的数据集文件test1_filesall.csv
数据df_x的分离
数据df_x差分处理，生成df_x_diff，并保存到df_x_diff.csv
"""
# dir = 'test1_files'
# Concat_data(dir)
# df = pd.read_csv('test1_filesall.csv')
#
# df_x = df.loc[:][df.loc[:] % 1 != 0]     # 删除每行数据前面的标签部分
# df_x = df_x.apply(squeeze_nan, axis=1)
#
# df_x_diff = df_x.diff(1, axis=1)         # 数据差分
# df_x_diff = df_x_diff.drop(columns=[str(0)], axis=1)
# df_x_diff.to_csv('df_x_diff.csv')
"""
df_x_diff.csv文件数据集如下，shape为(200, 342)
          1         2         3        4         5  ...  338  339  340  341  342
0 -0.016380  0.843778  0.876212  0.04911 -0.826166  ...  NaN  NaN  NaN  NaN  NaN
1  0.903325  1.085235 -0.799510 -1.16172 -0.063490  ...  NaN  NaN  NaN  NaN  NaN
2 -1.627750  1.850670 -1.207310 -0.00118  1.209120  ...  NaN  NaN  NaN  NaN  NaN
3 -0.128132 -0.123710 -0.117200 -0.10867 -0.098300  ...  NaN  NaN  NaN  NaN  NaN
4 -1.505990 -0.097600  1.561760 -0.79497 -1.107430  ...  NaN  NaN  NaN  NaN  NaN
"""


def distEclud(vecA, vecB):
    return np.sum(np.power(vecA - vecB, 2))  # la.norm(vecA-vecB),计算俩向量的距离


def randCent(dataSet, k):
    """返回k个随机样本,随机样本的属性值在数据集样本属性值内"""
    n = np.shape(dataSet)[1]    # 数据集为2列
    centroids = np.mat(np.zeros((k, n))) #创建K行2列的零矩阵
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = float(max(dataSet[:, j]) - minJ)
        centroids[:, j] = np.mat(minJ + rangeJ * np.random.rand(k, 1))   # np.random.rand(k, 1)返回一个K*1的array，元素服从“0~1”均匀分布
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    """
    随机生成k个簇心，进行更新
    :param dataSet: 样本
    :param k: 簇心个数
    :param distMeas: 计算簇心距离的函数
    :param createCent: 创建初始簇心的函数
    :return: centroids, clusterAssment，k个簇心和m个样本的簇信息
    """
    m = np.shape(dataSet)[0]
    """创建一个m*2矩阵，第一列保存样本所属簇，第二列保存样本到簇心的距离平方"""
    clusterAssment = np.mat(np.zeros((m,2)))
    """生成随机k个样本"""
    centroids = createCent(dataSet, k)
    """定义clusterChanged，遍历m个样本，只要任一改变了所属簇心，则遍历完1轮后，再更新簇心；重新下一次遍历，直到所有样本不改变簇心"""
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        """遍历m个样本"""
        for i in range(m):
            minDist = float("inf")
            minIndex = -1
            """遍历k个簇心，找到最近的"""
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            """如果簇心改变了，clusterChanged改为True，簇信息存入clusterAssment"""
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist**2
        """遍历完整1轮后，更新簇心情况，再根据clusterChanged决定是否遍历下一轮"""
        for cent in range(k):
            """返回簇心索引为当前簇cent的样本，对应索引"""
            dataSet_cent = np.nonzero(clusterAssment[:, 0].A == cent)
            ptsInClust = dataSet[dataSet_cent[0]]  # get all the point in this cluster
            centroids[cent, :] = np.mean(ptsInClust, axis=0) # assign centroid to mean
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    """
    二分法：将所有点看成一个簇，当簇小于k时，
    计算每一个簇的误差平方和ssh，选择ssh最大的簇进行再分，直到簇数达到k
    ps:刚开始可以取较大的k，计算簇数增大的时候，总的ssh是否依次减小，当不再明显减小时，k合适
    :param dataSet: 样本
    :param k: 簇心个数
    :param distMeas: 计算簇心距离的函数
    :return: k个簇心和m个样本的簇信息
    """
    sum_sse_list = [float("inf"), ]
    m = np.shape(dataSet)[0]  # 样本数量
    """创建一个m*2矩阵，第一列保存样本所属簇，第二列保存样本到簇心的距离平方"""
    clusterAssment = np.mat(np.zeros((m, 2)))
    """初始簇心，即k=1时簇心"""
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]  # 将1*2的矩阵转换为list后，取[0]j即矩阵第1行
    centList =[centroid0]
    """计算m个样本与簇心的距离，并保存在clusterAssment"""
    for j in range(m):
        clusterAssment[j, 1] = distMeas(np.mat(centroid0), dataSet[j, :])**2
    """当簇心个数小于K,循环"""
    lowestSSE = float("inf")
    while (len(centList) < k) and lowestSSE > 1e-5:
        for i in range(len(centList)):
            """step1：对列表中的每个簇心i，对簇i内样本尝试一分为二，计算ssh"""
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A==i)[0], :]   # get the data points currently in cluster i
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])  # compare the SSE to the currrent minimum   簇i被划分后的俩子簇ssh之和
            sseNotSplit = sum(clusterAssment[np.nonzero(clusterAssment[:, 0].A!=i)[0],1])  # 此次未被划分的簇（即非簇i）的ssh之和
            # print("sseSplit, and notSplit: ", sseSplit, sseNotSplit)
            """如果总的ssh有改善"""
            if (sseSplit + sseNotSplit) < lowestSSE:
                bestCentToSplit = i                  # 被划分的簇i
                bestNewCents = centroidMat           #簇i分开后的俩子簇心
                bestClustAss = splitClustAss.copy()  #原属簇i的样本的在新子簇下的归属与距离信息
                lowestSSE = sseSplit + sseNotSplit   # 总ssh
                # print('sum_ssh is:', lowestSSE)

        """step2：选择好了被划分的簇后...更新簇心和列表centList"""
        sum_sse_list.append(lowestSSE)  # 长度即簇心个数

        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)  # change 1 to 3,4, or whatever
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        # print('the bestCentToSplit is: ', bestCentToSplit)
        # print('the len of bestClustAss is: ', len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0, :].tolist()[0]  # replace a centroid with two best centroids
        centList.append(bestNewCents[1, :].tolist()[0])
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss  # reassign new clusters, and SSE
    return np.mat(centList), clusterAssment, sum_sse_list


def find_cut_index(core_belong_list):
    """
    给定一个列表，依据 step3 的逻辑查找core_belong_list并返回原型分段处的索引
    """
    # 定义一个字典dic，保存各个簇心编号及其出现次数
    dic = {core_belong_list[0]: 0, core_belong_list[1]: 0, core_belong_list[2]: 0}
    # flag标记，新出现的簇心编号，在之前dic内没出现过，若出现2次，则认为一个原型结束
    flag = 0
    for i, core in zip(range(len(core_belong_list)), core_belong_list):
        if dic[core_belong_list[0]] > 5 and dic[core_belong_list[1]] > 5 and dic[core_belong_list[2]] > 5:
            if core not in dic:
                flag += 1
                if flag == 2:
                    # print('this is a true new model index:', i)
                    return i      # 返回分段处的数据点在core_belong_list的索引值
        dic[core] = 1 + dic.setdefault(core, 0)    # 统计各簇心编号次数，首次出现默认键值为0

    return len(core_belong_list)-1   # 如果没有找到分段点，则认为是整个原型，返回原型结尾处的索引


def core_belong2cut_index(core_belong_mat):
    """
    根据样本数据点的簇编号，进行原型分段，返回原型分段处索引、原型个数
    输入：mX1阶矩阵，某个样本所有数据点的簇编号
    输出：各个分段处的索引cut_list[1:-1]，以及原型个数
    step3：从前往后统计：前3个数据点对应簇出现至少5次，之后的数据点对应簇在前面出现过；直到新出现2个数据点对应簇在前面没出现过，则在此截断，一个原型结束
    """
    # 将矩阵类型转换为list类型
    core_belong_list = [core[0, 0] for core in core_belong_mat]
    # 定义一个cut_list列表，保存各个分段索引
    cut_list = [0]
    while cut_list[-1] < len(core_belong_list)-1:
        i = find_cut_index(core_belong_list[cut_list[-1]:])
        cut_list.append(i+cut_list[-1])

    # print('cut_index is:', cut_list[1:-1])
    return cut_list[1:-1], len(cut_list[1:-1])+1


def YuanXing2Cycle_num(core_list):
    """
    根据某个原型，计算统计该原型的周期
    step1：统计各个簇的数据点个数
    step2：对各个簇依据数据点数目排序，删掉数目最多、最少的1-3个簇（过滤原型首尾的数据点）
    step3：对其他簇的数目求中位数（或者均值），即周期
    输入：原型数据点对于簇心编号
    输出：周期cycle
    """
    # 定义一个字典dic，保存统计各个簇的数据点个数
    dic = {}
    for core in core_list:
        dic[core] = 1 + dic.setdefault(core, 0)  # 统计各簇心编号次数，首次出现默认键值为0
    # 对簇次数进行排序
    value_list = sorted(dic.values())
    half = len(value_list) // 2
    # 获取次数的中位数，并返回
    return (value_list[half] + value_list[~half]) / 2


def list2sublist(par_list, sub_list):
    """将parlist按照sub_list的索引，拆分成多个子列表并返回"""
    res = []
    sub_list.append(len(par_list)-1)
    sub_list.insert(0, 0)
    for i in range(len(sub_list)-1):
        res.append(par_list[sub_list[i]:sub_list[i+1]])

    return res


if __name__ == "__main__":
    """
    导入数据集：
    df_x数据集的加载
    标签的处理，df_y为 200X6 的数据集
    """
    df_x = pd.read_csv('df_x_diff.csv')  # 获取df_x
    df_x = df_x.drop(columns=['Unnamed: 0'], axis=1)    # (200, 342)

    df = pd.read_csv('test1_filesall.csv')
    df_y = df.loc[:][df.loc[:6] % 1 == 0]
    df_y = df_y.drop(columns=[str(i) for i in range(6, 343)], axis=1)
    df_y.fillna(0).astype('int8')

    # 对每个数据集（共200个）转置为 342X1 的矩阵，并删除末位的空值
    for i in range(1):
        # 对某个数据集进行删除空值，转置为mX1格式
        dataMat = df_x.iloc[i]
        dataMat = dataMat.dropna()
        dataMat = np.mat(dataMat.tolist()).reshape(-1, 1)

        # 对数据集作图
        # sns.lineplot(x=range(dataMat.shape[0]), y=[x[0] for x in dataMat.tolist()])
        # plt.show()

        # 簇心个数最多不超过100，根据sum_sse_list变化情况，变动值小于阈值1e-5（待优化）来自动确定k值
        k = 100  # 定义簇心个数(最多每个原型有20个点，5个原型)
        """二分法下的簇情况，因随机聚类，此处多次计算，直到后续计算原型个数相同"""
        changed = 0
        flag = 0
        while changed < 2:
            a, core_belong, sum_sse_list = biKmeans(dataMat, k, distMeas=distEclud)

            # core_belong为各个样本点所属簇心
            # print('本数据集簇数为：', i, 1 + max(core_belong[:, 0]))
            # print('各个样本点所属簇心：\n', core_belong[:, 0])
            # print(type(core_belong[:, 0]))
            # print(core_belong[:, 0].shape)

            # sum_sse_list 为k 不断增加的情况下，sum_sse值
            # print('sse变化情况：\n', sum_sse_list)

            # sum_sse随k值变动情况绘图
            # sns.lineplot(x=range(k-1), y=[ele[0, 0] for ele in sum_sse_list[1:]])
            # plt.show()

            # delta_sum_sse随k值变动情况
            # sse_lst = [ele[0, 0] for ele in sum_sse_list[1:]]
            # sse_df = pd.DataFrame(sse_lst)
            # sse_df_diff = sse_df.diff(-1, axis=0)
            # sns.lineplot(x=range(k - 1), y=sse_df_diff[0].tolist())
            # plt.show()

            # 计算某个样本集的原型个数，以及分段处索引
            cut_list, yuanxin_num = core_belong2cut_index(core_belong[:, 0])
            # 如果连续3次的分段数目一致，则认为分段正确，进入YuanXing2Cycle_num统计各段周期
            if flag == yuanxin_num:
                changed += 1
                if changed == 2:
                    print('最终的原型个数确认为：', yuanxin_num)
                    # print(cut_list)

                    # 某个样本集的簇编号，格式mat转为list
                    core_belong_list = [core[0, 0] for core in core_belong[:, 0]]
                    # 计算每个原型段的周期数目
                    for j in list2sublist(core_belong_list, cut_list):
                        cycle = YuanXing2Cycle_num(j)
                        print('周期数目为：', cycle)
            else:
                changed = 0
            flag = yuanxin_num








