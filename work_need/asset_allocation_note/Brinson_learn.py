import pandas as pd
import numpy as np
import backtrader as bt
import akshare as ak

# import matplotlib.pyplot as plt
# import seaborn as sns
# pd.options.display.notebook_repr_html=False  # 表格显示
# plt.rcParams['figure.dpi'] = 75  # 图形分辨率
# sns.set_theme(style='darkgrid')  # 图形主题

'''
参考资料:
1、https://blog.csdn.net/qq_43382509/article/details/106029241
2、https://max.book118.com/html/2021/0221/7101200043003056.shtm
'''

def industry_quick_process(target,y_lst):
    industry_target = pd.concat([ak.fund_portfolio_industry_allocation_em(target,date) for date in y_lst ])
    industry_target['截止时间']= industry_target['截止时间'].apply(lambda x:pd.to_datetime(x))
    industry_target.sort_values('截止时间',inplace=True)
    industry_target = industry_target.drop(columns=['序号']).reset_index(drop = True)
    return industry_target

def Brinson_Multiple(p_w, p_r, b_w, b_r):
    sectors = list(b_w.columns)  # 从基准中提取行业名称（申万一级）
    td_dates = list(b_w.index)  # 从基准值提取交易日信息
    ticker = ['R_pp', 'R_pb', 'R_bp', 'R_bb']
    cum_R = pd.DataFrame(0, columns=ticker, index=td_dates).astype('float')
    single_R = pd.DataFrame(0, columns=ticker, index=td_dates).astype('float')

    for d in td_dates[1:]:
        for s in sectors:
            single_R['R_bb'][d] += b_w[s][d] * b_r[s][d]
            single_R['R_bp'][d] += b_w[s][d] * p_r[s][d]
            single_R['R_pb'][d] += p_w[s][d] * b_r[s][d]
            single_R['R_pp'][d] += p_w[s][d] * p_r[s][d]

        for t in ticker:
            for dd in td_dates[0:td_dates.index(d)]:
                cum_R[t][d] += (cum_R[t][dd] + 1) * single_R[t][td_dates[td_dates.index(dd) + 1]]

    Total_Excess_Return = cum_R['R_pp'] - cum_R['R_bb']
    Time_Selection = cum_R['R_pb'] - cum_R['R_bb']
    Stock_Selection = cum_R['R_bp'] - cum_R['R_bb']
    Interactive_Effect = Total_Excess_Return - Time_Selection - Stock_Selection

    Outcome = pd.DataFrame(list(zip(Total_Excess_Return, Time_Selection, Stock_Selection, Interactive_Effect)),
                           columns=['Total_Excess_Return', 'Time_Selection', 'Stock_Selection', 'Interactive_Effect'],
                           index=td_dates)
    return Outcome

y_lst = np.arange(2020,2023)

# 获取沪深300 及其成分
hs_300_weight = ak.index_stock_cons_weight_csindex('000300')

quote_hs_300 = ak.stock_zh_index_daily('sh000300')
quote_hs_300['date'] = pd.to_datetime(quote_hs_300['date'])
quote_hs_300.set_index('date',inplace=True)
quote_hs_300['rev'] = quote_hs_300['close']/quote_hs_300['close'].iloc[0]-1

hs300_ni = quote_hs_300['rev']

#300 ETF的行业持仓
industry_300 = industry_quick_process('159919',y_lst)
industry_300 = industry_300.pivot('截止时间','行业类别','占净值比例')
industry_300 = industry_300.fillna(0)

port_300_weighted_rev = industry_300.copy(deep=True)

for col in industry_300.columns:
    port_300_weighted_rev[col] = industry_300[col] * hs300_ni.loc[hs300_ni.index.isin(industry_300.index)]/100



'''
中信红利价值 = 900011
'''
my_fund_code = '900011'
fund_info = ak.fund_name_em()

target_fund_info  = fund_info.query(''' 基金代码=='900011' ''')

target_fund_quote = ak.fund_financial_fund_info_em('900011')
target_fund_quote['净值日期'] = pd.to_datetime(target_fund_quote['净值日期'])
target_fund_quote.sort_values('净值日期',inplace=True)
target_fund_quote['每万份收益'] = target_fund_quote['每万份收益'].astype('float')
target_fund_quote['7日年化收益率'] = target_fund_quote['7日年化收益率'].astype('float')
target_fund_quote['pct_change'] = target_fund_quote['每万份收益']/target_fund_quote['每万份收益'].iloc[0]-1
target_fund_quote = target_fund_quote.set_index('净值日期')

# 基金持仓--个股

# 获取持仓信息
# ak.fund_portfolio_hold_em('900011')
target_portfolio = pd.concat([ak.fund_portfolio_hold_em('900011',date) for date in y_lst])

#基金持仓--行业
# ak.fund_portfolio_industry_allocation_em('900011')
target_industry = industry_quick_process(my_fund_code,y_lst)
target_industry = target_industry.pivot('截止时间','行业类别','占净值比例')
target_industry = target_industry.fillna(0)

# 获取基金净值序列
target_ni = ak.fund_financial_fund_info_em(my_fund_code)
target_ni['净值日期'] = pd.to_datetime(target_ni['净值日期'])
target_ni.sort_values('净值日期',inplace=True)
target_ni['每万份收益'] = target_ni['每万份收益'].astype('float')
target_ni.set_index('净值日期',inplace=True)

target_ni['rev'] = target_ni['每万份收益']/target_ni['每万份收益'].iloc[0]-1
target_rev = target_ni['rev']

port_target_rev = industry_300.copy(deep=True)

for col in target_industry.columns:
    port_target_rev[col] = target_industry[col] * target_rev.loc[target_rev.index.isin(target_industry.index)]/100

port_300_weighted_rev
outt  = Brinson_Multiple(target_industry/100, port_target_rev, industry_300/100, port_300_weighted_rev)

# 获取申万一级行业
'''
sws_level_1_url = f'https://www.swsresearch.com/institute-sw/api/index_publish/details/timelines/?swindexcode={my_fund_code}'
Vix
https://wallstreetcn.com/c/chart?description=VIX%E6%B3%A2%E5%8A%A8%E7%8E%87%E6%8C%87%E6%95%B0&interval=1D&symbol=VIX.OTC
'''




