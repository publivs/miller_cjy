# %matplotlib inline
import pandas as pd
import numpy as np
import scipy.optimize as sco
import scipy.interpolate as sci
import matplotlib.pyplot as plt
# import tushare as ts
import akshare as aks


# 【恒生电子、贵州茅台、海康威视、浦发银行、上海医药、格力电器】
symbols = ['600570','600519','002415','600000','601607', '000651'] 
noa = len(symbols)

# 生成不同资产近一年历史数据的表格
data = pd.DataFrame()
for sym in symbols:
    data[sym] = aks.stock_zh_a_hist(sym)['收盘']


rets = np.log(data / data.shift(1))  # 各资产收益率
arets = rets.mean() * 252  # 各资产年化收益率
ret_max = max(arets)  # 各项资产的年化收益最大值
ret_min = min(arets)  # 各项资产的年化收益最小值
(data / data.iloc[0] * 100).plot(figsize=(8, 5), grid=True)  # 各资产的收益曲线

prets = []
pvols = []
for p in range (2000):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    prets.append(np.sum(rets.mean() * weights) * 252)  # 期望年化收益
    pvols.append(np.sqrt(np.dot(weights.T, 
                        np.dot(rets.cov() * 252, weights))))  # 期望年化波动
prets = np.array(prets)
pvols = np.array(pvols)

plt.figure(figsize=(8, 4))
plt.scatter(pvols, prets, c=prets / pvols, marker='o')
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')

def statistics(weights):
    ''' Return portfolio statistics.

    Parameters
    ==========
    weights : array-like
        weights for different securities in portfolio

    Returns
    =======
    pret : float
        expected portfolio return
    pvol : float
        expected portfolio volatility
    pret / pvol : float
        Sharpe ratio for rf=0
    '''
    weights = np.array(weights)
    portfolio_ret = np.sum(rets.mean() * weights) * 252
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))
    return np.array([portfolio_ret, portfolio_vol, portfolio_ret / portfolio_vol])

