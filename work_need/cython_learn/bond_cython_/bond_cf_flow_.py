#导入包
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.offline as of  # 这个为离线模式的导入方法
import datetime
from scipy import interpolate
from scipy import optimize as so
from datetime import timedelta
from dateutil.relativedelta import relativedelta

#数据导入
path = r'C:\Users\kaiyu\Desktop\miller\work_need\cython_learn\bond_cython_\real_ytm.csv'
data_policyfinancialdebt = pd.read_csv(path)

#使用插值法将收益率补齐
duration = sorted(np.concatenate((data_policyfinancialdebt['标准期限(年)'].values,data_policyfinancialdebt['剩余期限(年)'].values)))
duration.pop(0)
duration.pop()
f1=interpolate.interp1d(x=data_policyfinancialdebt['剩余期限(年)'],y=data_policyfinancialdebt['最优报买入收益率(%)'],kind='slinear')
f2=interpolate.interp1d(x=data_policyfinancialdebt['剩余期限(年)'],y=data_policyfinancialdebt['最优报卖出收益率(%)'],kind='slinear')
rates_new_buy = f1(duration)
rates_new_sell = f2(duration)

# 画收益率曲线图
line1 = go.Scatter(y=rates_new_buy, x=duration,mode='lines+markers', name='最优报买入收益率(%)')   # name定义每条线的名称
line2 = go.Scatter(y=rates_new_sell, x=duration,mode='lines+markers', name='最优报卖出收益率(%)')
fig = go.Figure([line1, line2])
fig.update_layout(
    title = '收益率曲线', #定义生成的plot 的标题
    xaxis_title = '期限', #定义x坐标名称
    yaxis_title = '收益率(%)'#定义y坐标名称
)
fig.show()

#计算债券的现金流列表，每一现金流对应的零息利率，每一现金流距离指定时间点间的时间距离
def cal_cashrtime(bar,couponrate,
                 startdate,next_coupon_date,enddate,
                 freq = 1):
    """
   计算债券的现金流列表，每一现金流对应的零息利率，每一现金流距离指定时间点间的时间距离
   Args:
       startdate:  需折现到的日期
       coupon_date: 下一次付息日
       enddate: 债券到期鈤日
       freq: 年付息次数
       duration: 用于插值法的期限list
       rate_list: 用于插值法的利率list
   Returns:
       现金流list，现金流时间距离list,现金流对应零息利率list
    """
    cashflow = []
    time_list = []
    date_temp = next_coupon_date
    while(enddate>=date_temp):
        cashflow.append(bar * couponrate)
        time_list.append((date_temp-startdate)/timedelta(365))
        date_temp = (date_temp + relativedelta(years=1))
    cashflow.append(bar)
    time_list.append((enddate-startdate)/timedelta(365))
    #插值法获取零息利率
    return cashflow,time_list

def calc_rate_list(duration,rate_list,time_list):
    f=interpolate.interp1d(x=duration,y=rate_list,kind='slinear')
    r_list = list(f(time_list))
    return r_list

#债券精确定价函数
def bond_preciseprice(bar,coup_rate,r_list,time_list):
    """
   计算一只债券的精确定价
   Args:
       bar:  债券的票面价值
       coup_rate: 债券的票面利率
       r_list: 每一现金流对应的零息利率
       time_list: 每一现金流离目前的时间点
   Returns:
       返回债券的精确定价
    """
    per_coupon = bar * coup_rate
    discount_coupon = 0
    for r,time in zip(r_list,time_list):
        if(r != r_list[-1]):
            discount_coupon = discount_coupon + per_coupon/(1 + r*0.01)**time
    return (discount_coupon + bar/(1 + r_list[-1]*0.01)**time_list[-1])

#计算麦考利久期
def mcduration(cashflow,time_list,r_list,presentvalue):
    mcduration = 0
    arr_len = len(cashflow)
    for i in range(arr_len):
        cash = cashflow[i]
        time = time_list[i]
        r = r_list[i]
        mcduration = mcduration + time * ( cash / (1 + r * 0.01) ** time) / presentvalue
    return mcduration

#计算ytm
def YTM(presentvalue,cashflow,time_list):
    def ff(y):
        cash_all = []
        for cash,time in zip(cashflow,time_list):
            if(cash != cashflow[-1]):
                cash_all.append(cash/pow(1+y,time))
        return np.sum(cash_all)+cashflow[-1]/pow(1+y,time_list[-1])-presentvalue
    return float(so.fsolve(ff,0.01))

#计算修正久期
def dduration(cashflow,time_list,r_list,presentvalue,ytm):
    a = mcduration(cashflow,time_list,r_list,presentvalue)
    b = (1+ytm)
    c = a/b
    return c

#计算凸性
def Convexity(cashflow,time_list,presentvalue,ytm):
    temp = []
    ytm = ytm
    for cash,time in zip(cashflow,time_list):
        temp.append(cash*(time*time + time)/pow(1+ytm,time))
    return (1/(presentvalue*pow(1+ytm,2))) * np.sum(temp)

#配置22国开05信息配置
bar,couponrate,startdate,next_coupon_date,enddate = 100,0.03,datetime.datetime(2022,5,10),\
                                                datetime.datetime(2023,1,16),datetime.datetime(2032,1,17)

cashflow,time_list = cal_cashrtime(bar,couponrate,startdate,next_coupon_date,enddate)

r_list = calc_rate_list(duration,rates_new_sell,time_list)

price2205 = bond_preciseprice(bar,couponrate,r_list,time_list)

print("22国开05的定价为：",price2205)

mcd2205 =  mcduration(cashflow,time_list,r_list,price2205)

ytm  = YTM(price2205,cashflow,time_list)

dd2205 = dduration(cashflow,time_list,r_list,price2205,ytm)

conv = Convexity(cashflow,time_list,price2205,ytm)

print(1)