from Approximation import (Approximation, Mask_dir_peak_valley,
                            Except_dir, Mask_status_peak_valley,
                            Relative_values,Segment_stats)

from performance import Strategy_performance
from collections import (defaultdict, namedtuple)
from typing import (List, Tuple, Dict, Union, Callable, Any)


import datetime as dt
import empyrical as ep
import numpy as np
import pandas as pd
import talib
import scipy.stats as st
from IPython.display import display

from sklearn.pipeline import Pipeline

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  #用来正常显示负号

import akshare as ak
# 画图

# 原文章
# https://www.joinquant.com/view/community/detail/f5d05b8233169adbbf44fb7522b2bf53?type=1&page=1

def plot_pivots(peak_valley_df: pd.DataFrame,
                show_dir: Union[str,List,Tuple]='dir',
                show_hl: bool = True,
                show_point:bool = True,
                title: str = '',
                ax=None):

    if ax is None:

        fig, ax = plt.subplots(figsize=(18, 6))

        line = peak_valley_df.plot(y='close', alpha=0.6, title=title, ax=ax)

    else:

        line = peak_valley_df.plot(y='close', alpha=0.6, title=title, ax=ax)

    if show_hl:

        peak_valley_df.plot(ax=line,
                            y='PEAK',
                            marker='o',
                            color='r',
                            mec='black')

        peak_valley_df.plot(ax=line,
                            y='VALLEY',
                            marker='o',
                            color='g',
                            mec='black')

    if show_point:

        peak_valley_df.dropna(subset=['POINT']).plot(ax=line,
                                                     y='POINT',
                                                     color='darkgray',
                                                     ls='--')
    if show_dir:

        peak_valley_df.plot(ax=line,
                            y=show_dir,
                            secondary_y=True,
                            alpha=0.3,
                            ls='--')

    return line

def get_clf_wave(price: pd.DataFrame,


                 rate: float,
                 method: str,
                 except_dir: bool = True,
                 show_tmp: bool = False,
                 dropna: bool = True) -> pd.DataFrame:


    if except_dir:

        # 修正
        perpare_data = Pipeline([('approximation', Approximation(rate, method)),
                ('mask_dir_peak_valley',Mask_status_peak_valley('dir')),
                ('except', Except_dir('dir')),
                ('mask_status_peak_valley', Mask_dir_peak_valley('status'))
                ])
    else:

       # 普通
        perpare_data = Pipeline([('approximation', Approximation(rate, method)),
                ('mask_dir_peak_valley',Mask_dir_peak_valley('dir')),
                ('mask_status_peak_valley', Mask_status_peak_valley('dir'))])

    return perpare_data.fit_transform(price)

quote_hs_300 = ak.stock_zh_index_daily('sh000300')

quote_hs_300['date'] = pd.to_datetime(quote_hs_300['date'])
quote_hs_300.set_index('date',inplace=True)
quote_hs_300 = quote_hs_300.loc[(quote_hs_300.index<='20211022')&(quote_hs_300.index>='20050408')]

begin, end = '2020-02-01','2020-07-20'

# 方式一
flag_frame1: pd.DataFrame = get_clf_wave(quote_hs_300,None,'a',False)
flag_df1 = flag_frame1.loc[begin:end,['close','dir']]
flag_df1 = flag_df1.rename(columns={'dir':'方式1划分上下行'})
line = flag_frame1.loc['2021-01-01':'2021-07-30'].plot(figsize=(18, 6), y='close', color='red',
                    title='沪深300收盘价、DIF线与DEA线(2021-01-04至2021-07-30)')

flag_frame1.loc['2021-01-01':'2021-07-30'].plot(ax=line, y=['dif', 'dea'],
             secondary_y=True, color=['#3D89BE', 'darkgray']);
# 画图
line = flag_df1.plot(y='方式1划分上下行', secondary_y=True, figsize=(
    18, 5), ls='--', color='darkgray', title='沪深300与上下行划分(方式1)')
flag_df1.plot(y='close', ax=line, color='r');


# 方式2:划分上下行
flag_frame2: pd.DataFrame = get_clf_wave(quote_hs_300,0.5,'b',False)
flag_df2 = flag_frame2.loc[begin:end,['close','dir']]
flag_df2 = flag_df2.rename(columns={'dir':'方式2划分上下行'})
# 画图
line = flag_df2.plot(y='方式2划分上下行', secondary_y=True, figsize=(
    18, 5), ls='--', color='darkgray', title='沪深300与上下行划分(方式2,Rate=0.5)')
flag_df2.plot(y='close', ax=line, color='r');


# 方式3:划分上下行 -- 最标准的算法,带修正
flag_frame3: pd.DataFrame = get_clf_wave(quote_hs_300,2,'c',True)
flag_df3 = flag_frame3.loc[begin:end,['close','dir']]
flag_df3 = flag_df3.rename(columns={'dir':'方式3划分上下行'})
# 画图
line = flag_df3.plot(y='方式3划分上下行', secondary_y=True, figsize=(
    18, 5), ls='--', color='darkgray', title='沪深300与上下行划分(方式3,Rate=2)')

flag_df3.plot(y='close', ax=line, color='r');


status_frame: pd.DataFrame = get_clf_wave(quote_hs_300, 2, 'c', True)
dir_frame: pd.DataFrame = get_clf_wave(quote_hs_300, 2, 'c', True)

fig, axes = plt.subplots(2,figsize=(18,12))
# 画图
plot_pivots(dir_frame.iloc[330:450],
            show_dir=['dir'],
            show_hl=True,
            title='沪深300指数波段划分结果展示(方法3,Rate=2)',ax=axes[0])

#
plot_pivots(status_frame.iloc[330:450],
            show_dir=['status'],
            show_hl=True,
            title='沪深300指数波段划分结果展示(方法3-修正,Rate=2)',ax=axes[1]);


fig, axes = plt.subplots(2,figsize=(18,12))
# 画图
plot_pivots(dir_frame.iloc[:50],
            show_dir=['dir'],
            show_hl=True,
            title='沪深300指数波段划分结果展示(方法3,Rate=2)',ax=axes[0])

plot_pivots(status_frame.iloc[:50],
            show_dir=['status'],
            show_hl=True,
            title='沪深300指数波段划分结果展示(方法3-修正,Rate=2)',ax=axes[1]);


fig, axes = plt.subplots(2,figsize=(18,12))
# 画图
plot_pivots(dir_frame.loc['2019-01-01':'2021-07-30'],
            show_dir=['dir'],
            show_hl=True,
            title='沪深300指数波段划分结果展示（2019-01-03 至 2021-07-30）(方法3,Rate=2)',ax=axes[0])

plot_pivots(status_frame.loc['2019-01-01':'2021-07-30'],
            show_dir=['status'],
            show_hl=True,
            title='沪深300指数波段划分结果展示（2019-01-03 至 2021-07-30）(方法3-修正,Rate=2)',ax=axes[1]);

fig, axes = plt.subplots(2,figsize=(18,12))
# 画图
plot_pivots(dir_frame.loc['2021-05-01':'2021-09-15'],
            show_dir=['dir'],
            show_hl=True,
            title='当前点位与它的前两个端点(方法3,Rate=2)',ax=axes[0])

plot_pivots(status_frame.loc['2021-05-01':'2021-09-15'],
            show_dir=['status'],
            show_hl=True,
            title='当前点位与它的前两个端点(方法3-修正,Rate=2)',ax=axes[1]);

'''
上下行划分分析
'''
fig, axes = plt.subplots(2,figsize=(18,12))

plot_pivots(status_frame,
            show_dir=False,
            show_hl=False,
            title='沪深300指数波段划分结果展示(方法3-修正,Rate=2)',ax=axes[0])

plot_pivots(dir_frame,
            show_dir=False,
            show_hl=False,
            title='沪深300指数波段划分结果展示(方法3,Rate=2)',ax=axes[1])


'''
波段的划分
'''
stats_summary = Segment_stats(status_frame,'status')

print('未去极值波段划分情况')
stats_summary.stats_summary()
stats_summary.ttest_segment()


print('去极值波段划分情况')
stats_summary.stats_summary(True)

stats_summary.ttest_segment(True)

fig,axes = plt.subplots(1,2,figsize=(18,6))

stats_summary.plot_segment_ret_hist(title='未去极值',ax=axes[0])
stats_summary.plot_segment_ret_hist(winsorize=True,title='去极值',ax=axes[1]);


# 未修正的波段划分情况

stats_summary = Segment_stats(dir_frame,'dir')

print('未去极值波段划分情况')
stats_summary.stats_summary()
stats_summary.ttest_segment()


print('去极值波段划分情况')
stats_summary.stats_summary(True)

stats_summary.ttest_segment(True)

fig,axes = plt.subplots(1,2,figsize=(18,6))

stats_summary.plot_segment_ret_hist(title='未去极值',ax=axes[0])

stats_summary.plot_segment_ret_hist(winsorize=True,title='去极值',ax=axes[1]);

'''
计算相对效率
'''
rv = Relative_values('dir')
rv_df:pd.DataFrame = rv.fit_transform(dir_frame)

test_rv_df:pd.DataFrame = rv_df[['close','relative_time','relative_price']].copy()

for i in range(1,25):
    test_rv_df[i] = test_rv_df['close'].pct_change(i).shift(-i)

drop_tmp = test_rv_df.dropna(subset=['relative_price'])
drop_tmp[['close', 'relative_price', 'relative_time']].plot(figsize=(18, 12),
                                                            subplots=True);

test_rv_df.loc['2021-09-15',['close','relative_time','relative_price']] # 检验,感觉似乎哪里出了问题先不管


'''
应用部分
'''
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin

X = drop_tmp[['relative_price','relative_time']].values

n_clusters = 3

k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
k_means.fit(X)

k_means_cluster_centers = k_means.cluster_centers_  # 获取聚类核心点

k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)  # 计算一个点和一组点之间的最小距离,默认欧式距离

k_mean_cluster_frame:pd.DataFrame = drop_tmp.copy()

k_mean_cluster_frame['label'] = k_means_labels

from apply_code import *

plot_simple_cluster(k_mean_cluster_frame,k_means_cluster_centers,
                x='relative_price',y='relative_time',hue='label');

mel_df = pd.melt(k_mean_cluster_frame,id_vars=['label'],value_vars=list(range(1,25)),var_name=['day'])
slice_df = mel_df.query('label==0').dropna() 
slice_df['day'] = slice_df['day'].astype(int)
fig,ax = plt.subplots(figsize=(14,9))
sns.kdeplot(data=slice_df, x='day',y='value',cbar=True,cmap="coolwarm");

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib # 用于模型导出

train_df = test_rv_df.loc[:'2018-01-01'].dropna()

test_df = test_rv_df.loc['2018-01-01':]

x_test = test_df[['relative_time','relative_price']]

tscv = TimeSeriesSplit(n_splits=5,max_train_size=180)

nb = GaussianNB()

lr = LogisticRegression()

df = pd.DataFrame()
next_ret = test_rv_df['close'].pct_change().shift(-1)

df['GaussianNB'] = next_ret.loc[test_df.index] * nb.predict(x_test)
df['LogisticRegression'] = next_ret.loc[test_df.index] * lr.predict(x_test)
ep.cum_returns(df).plot(figsize=(18,6))

ep.cum_returns(next_ret.loc[x_test.index]).plot(color='darkgray',label='HS300')
plt.legend();