import pandas as pd
import numpy as np
import backtrader as bt
import akshare as ak
import requests

# import matplotlib.pyplot as plt
# import seaborn as sns
# pd.options.display.notebook_repr_html=False  # 表格显示
# plt.rcParams['figure.dpi'] = 75  # 图形分辨率
# sns.set_theme(style='darkgrid')  # 图形主题

'''
第二篇:
https://max.book118.com/html/2021/0917/6110110111004005.shtm
'''



def calculateMACD(closeArray, shortPeriod=12, longPeriod=26, signalPeriod=9):

    def calculateEMA(period, closeArray, emaArray=[]):
        length = len(closeArray)
        nanCounter = np.count_nonzero(np.isnan(closeArray))
        if not emaArray:
            emaArray.extend(np.tile([np.nan], (nanCounter + period - 1)))
            firstema = np.mean(closeArray[nanCounter:nanCounter + period - 1])
            emaArray.append(firstema)
            for i in range(nanCounter + period, length):
                ema = (2 * closeArray[i] + (period - 1) * emaArray[-1]) / (period + 1)
                emaArray.append(ema)
        return np.array(emaArray)

    ema12 = calculateEMA(shortPeriod, closeArray, [])
    ema26 = calculateEMA(longPeriod, closeArray, [])

    diff = ema12 - ema26
    dea = calculateEMA(signalPeriod, diff, [])
    macd = (diff - dea)

    return diff, dea, macd

def get_atr(quote_df,N_input =100):
    def calc_tr(t_high,t_low,pre_close):
        tr = max(t_high -t_low, t_high - pre_close,pre_close - t_low)
        return tr
    tr_lst = []
    atr_lst = [ ]
    for i in range(len(quote_df)):
        se_i = quote_df.iloc[i]
        if i < N_input:
            N = i+1
            tr = calc_tr(se_i['high'],se_i['high'],se_i['pre_close'])
            tr_lst.append(tr)
            atr = sum(tr_lst)/N
            atr_lst.append(atr)
    return atr_lst



def calc_integrated_error(dif,dea):
    # 计算累计误差
    # 考虑同号一致性
    def calc_cum_error(current,cum_error,):
            if i>0 and cum_error >0 :
                cum_error += current
            elif i<0 and cum_error <0:
                cum_error += current
            else:
                cum_error = 0
                cum_error += current
            return cum_error

    dif_dea = pd.Series(dif - dea)
    current = 0
    cum_error = 0
    cum_list = []
    for i in dif_dea:
        if i == 0:
            cum_error = 0
        else:
            cum_error = calc_cum_error(i,cum_error,)
            cum_list.append(i)
    return cum_list


def get_direction(dea_t,dif_t,
                  last_direction,
                  integrate_error,
                  threshold_sigma,
                  method= '3'):
    if method == '1':
        if dif_t - dea_t > 0:
            return 1
        if dif_t - dea_t < 0:
            return -1
    elif method == '2':
        if dif_t - dea_t > threshold_sigma:
            return 1
        if dif_t - dea_t < threshold_sigma:
            return -1
    elif method == '3':
        # 上行状态
        if integrate_error > threshold_sigma:
            return 1
        # 下行状态
        elif integrate_error < -threshold_sigma:
            return -1
        # 未破阈值保持不变
        else:
            return last_direction

def generate_point_high_and_low():
    '''
    判断并生成出高低点
    '''
    pass

# 端点的自动化修正
def auto_process_exception(direction,last_direction,close,last_high,last_low):
    '''
    高低点分化和波段切割,应该是高低相间
    '''
    # 当前时刻检查异常
    if direction == 1 and close <= last_low:
        return -1
    if direction == -1 and close >= last_high:
        return -1
    '''
    1)dir_t-1 != dir_t
    2)dir_t =1,close_t > last_high
    3)dir_t = -1,close < last_low
    '''
    if last_direction != direction:
        return 1
    if direction ==1 and close >= last_high:
        return 1
    if direction == -1 and close <= last_low:
        return 1

def generate_statu(dir_t,exception_t):
    '''

    '''
    status = dir_t * exception_t
    # 当status = 1时,代表了上行没有异常状态;或者下行并出现异常状态
    return status

# 股指
quote_hs_300 = ak.stock_zh_index_daily('sh000300')
quote_hs_300['date'] = pd.to_datetime(quote_hs_300['date'])
quote_hs_300.set_index('date',inplace=True)
quote_hs_300['pre_close'] = quote_hs_300.close.shift(1)



class Calc_high_low:
    def __init__(self,df) -> None:
        self.quote_df = df
        self.dea,self.dif,self.macd = calculateMACD(self.quote_df['close'],)
        self.atr = get_atr(self.quote_df)
        self.rate_simga = 2
        self.sigma_t = self.rate_simga * self.atr
        self.integrate_lst = calc_integrated_error(self.dif,self.dea)


calc_obj = Calc_high_low(quote_hs_300)






'''
https://tvc4.investing.com/8463b893927927345bc436a112d04e51/1668654369/6/6/28/history?symbol=166&resolution=D&from=1455160204&to=1459048201
'''
