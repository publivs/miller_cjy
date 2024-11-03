import seaborn as sns
import pyecharts as chart
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

# from pyecharts import Bar

import json

# def autolabel(rects):
#     """在*rects*中的每个柱状条上方附加一个文本标签，显示其高度"""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3点垂直偏移
#                     textcoords="offset points",
#                     ha='center', va='bottom')

def global_stock_index_chart(params=None):

    def get_chart(df):
        color_dict = {}
        for col in df.columns :
            color_dict[col] ='b'
            if col in ['深圳成指','上证综指']:
                color_dict[col] ='r'

        sns.set_style('ticks',rc={'font.sans-serif':"Microsoft Yahei"})
        # 文字竖排显示
        #
        # df.columns = pd.Series(df.columns).map(lambda x: '\n'.join(x))
        #
        fig = plt.figure(figsize=(9,4))
        chart = sns.barplot(df,palette=color_dict,width=0.4,)
        chart.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
        chart.axhline(0, color="k", clip_on=False)
        chart.tick_params(bottom=False,direction = 'in',width =1)
        chart.tick_params(axis='x')
        plt.xticks(rotation=90)
        sns.despine(bottom = True)

        for col_num in range(len(df.columns)):
            col_name = df.columns[col_num]
            v_i = df[col_name].values[0]
            if v_i >0:
                chart.text(col_num,v_i+0.005,str(np.round(v_i*100,2))+' %',ha='center',fontsize = 10)
            else:
                chart.text(col_num,v_i-0.006,str(np.round(v_i*100,2))+' %',ha='center',fontsize = 10)

        plt.show()

    # params = json.loads(params)
    # df = pd.DataFrame([params['diVal']]).T
    # df.columns = params['name']

    df = pd.DataFrame([[-1,-2,3,4,5,6,7,8,9,10]])/100
    df.columns = ['深圳成指','俄罗斯RUS','上证综指','d','e','f','g','h','i','j']
    get_chart(df)

def get_turnover_rate(params = None):
    def get_chart(df):
        sns.set_style('white',rc={'font.sans-serif':"Microsoft Yahei"})

        # labels = ['第一项', '第二项','第三项']
        # bef_1 = [10000,20000,30000]
        # bef_1 = [10000,20000,30000]
        # bef_3 = [10000,20000,30000]
        # bef_4 = [10000,20000,30000]
        # bef_5 = [10000,20000,30000]
        # bef_6 = [10000,20000,30000]
        # this_week  = [10000,20000,30000]

        labels = df['name'].unique()
        bef_1 = df['befOneWkVal']
        bef_2 = df['befTwoWkVal']
        bef_3 = df['befThreeWkVal']
        bef_4 = df['befFourWkVal']
        bef_5 = df['befFiveWkVal']
        bef_6 = df['befSixWkVal']
        this_week = df['this_weekth']
        labels = df['name']
        x = np.arange(len(labels))  # 标签位置

        width = 0.1  # 柱状图的宽度
        fig, ax = plt.subplots()

        # 进入循环
        rects1 = ax.bar(x - width*2, bef_1, width, label='第一周')
        rects2 = ax.bar(x - width+0.01,bef_2 , width, label='第二周')
        rects3 = ax.bar(x + 0.02, bef_3, width, label='第三周')
        rects4 = ax.bar(x + width+ 0.03, bef_4, width, label='第四周')
        rects5 = ax.bar(x + width*2 + 0.04, bef_5, width, label='第五周')
        rects6 = ax.bar(x + width*2 + 0.15, bef_6, width, label='第六周')
        rects7 = ax.bar(x + width*3 + 0.16,this_week, width, label='本周日均换手率')
        # 循环退出
        # 为y轴、标题和x轴等添加一些文本。
        # ax.set_ylabel('换手率', fontsize=16)
        # ax.set_xlabel('X轴', fontsize=16)
        ax.set_title('换手率簇柱图(%)')
        ax.set_xticks(x)
        # ax.set_yticks()
        ax.yaxis.grid(True, color ="black")
        ax.set_xticklabels(labels)
        sns.despine(left = True)

        fig.tight_layout()
        plt.legend(bbox_to_anchor=(0.5,-0.25),loc=8, borderaxespad=1,ncol = 7,frameon=False,fancybox=True)
        plt.show()
    params = None
    df = pd.DataFrame()
    df['name'] = ['万德全A','中小板综','上证综指']
    df['befOneWkVal'] = [0.03,0.02,0.04]
    df['befTwoWkVal']=[0.03,0.02,0.04]
    df['befThreeWkVal']=[0.03,0.02,0.04]
    df['befFourWkVal']=[0.03,0.02,0.04]
    df['befFiveWkVal']=[0.03,0.02,0.04]
    df['befSixWkVal']=[0.03,0.02,0.04]
    df['this_weekth'] = [0.03,0.02,0.04]
    get_chart(df)

def get_trading_volume(params = None):

    def get_chart(df):
        sns.set_style('white',rc={'font.sans-serif':"Microsoft Yahei"})
        labels = df['name'].unique()
        bef_1 = df['befOneWkVal']
        bef_2 = df['befTwoWkVal']
        bef_3 = df['befThreeWkVal']
        bef_4 = df['befFourWkVal']
        bef_5 = df['befFiveWkVal']
        bef_6 = df['befSixWkVal']
        this_week = df['this_weekth']

        width = 0.1  # 柱状图的宽度
        fig, ax = plt.subplots()
        x = np.arange(len(labels))  # 标签位置

        rects1 = ax.bar(x - width*2, bef_1, width, label='第一周')
        rects2 = ax.bar(x - width+0.01,bef_1 , width, label='第二周')
        rects3 = ax.bar(x + 0.02, bef_3, width, label='第三周')
        rects4 = ax.bar(x + width+ 0.03, bef_4, width, label='第四周')
        rects5 = ax.bar(x + width*2 + 0.04, bef_5, width, label='第五周')
        rects6 = ax.bar(x + width*2 + 0.15, bef_6, width, label='第六周') 
        rects7 = ax.bar(x + width*3 + 0.16,this_week, width, label='本周日均成交额')

        # 为y轴、标题和x轴等添加一些文本。
        # ax.set_ylabel('换手率', fontsize=16)
        # ax.set_xlabel('X轴', fontsize=16)
        ax.set_title('换手成交额图')
        ax.set_xticks(x)
        # ax.set_yticks()
        ax.yaxis.grid(True, color ="black")
        ax.set_xticklabels(labels)
        sns.despine(left = True)

        fig.tight_layout()
        plt.legend(bbox_to_anchor=(0.5,-0.25),loc=8, borderaxespad=1,ncol = 7,frameon=False,fancybox=True)
        plt.show()

    params = None
    df = pd.DataFrame()
    df['name'] = ['万德全A','中小板综','上证综指']
    df['befOneWkVal'] = [10000,20000,30000]
    df['befTwoWkVal']= [10000,20000,30000]
    df['befThreeWkVal']= [10000,20000,30000]
    df['befFourWkVal']= [10000,20000,30000]
    df['befFiveWkVal']= [10000,20000,30000]
    df['befSixWkVal']= [10000,20000,30000]
    df['this_weekth'] = [10000,20000,30000]
    get_chart(df)

def get_finance_chart(params = None):
    def get_chart(mv,finance ,date_lst ):
        sns.set_style('white',rc={'font.sans-serif':"Microsoft Yahei"})
        fig=plt.figure()
        ax1=fig.add_subplot(111)
        ax1.plot(date_lst,mv,label='两融余额占A股流通市值(%)',color='red',lw=3)
        plt.ticklabel_format(style='plain',scilimits=(0,0),axis='y')
        plt.legend(bbox_to_anchor=(0.2,-0.4),loc=8,borderaxespad=1,ncol = 6,frameon=False,fancybox=True,fontsize = 10)
        #横坐标显示的设置一定要在建立双坐标轴之前
        plt.xticks(date_lst,rotation=45)

        # plt.ticklabel_format(style='plain')
        ax1.yaxis.grid(True, color ="black")
        ax2=ax1.twinx()
        ax2.plot(date_lst,finance,label='融资融券余额(沪深两市)',color='blue',alpha = 0.5)
        ax2.get_yaxis().get_major_formatter().set_scientific(False)
        ax2.tick_params(right=False)
        plt.fill_between(x=date_lst, y1=1, y2=finance, facecolor='blue',alpha =0.5)
        sns.despine(left = True)
        plt.legend(bbox_to_anchor=(0.8,-0.4),loc=8,borderaxespad=1,ncol = 6,frameon=False,fancybox=True,fontsize = 10)
        plt.show()

    mv = np.array([600,200,300,100,500,800,400])*1000000
    finance_r = [1,2,3,4,5,6,7]
    date_lst = pd.date_range('20221001','20221115',freq='w').astype('str')
    get_chart(mv,finance_r ,date_lst )


def get_fund_volume(params = None):
    def get_chart(date_lst,combine_fund_vol,stock_fund_vol):
        sns.set_style('white',rc={'font.sans-serif':"Microsoft Yahei"})
        fig=plt.figure(figsize=(14,6))
        ax1 = sns.lineplot(x = date_lst,y=stock_fund_vol,label = '股票基金发行份额',lw = 3.5)
        ax2 = sns.lineplot(x = date_lst,y=combine_fund_vol,label = '混合型基金发行份额',lw= 3.5)
        plt.xticks(date_lst,rotation=90)
        ax1.yaxis.grid(True, color ="black")
        plt.legend(bbox_to_anchor=(0.5,-0.5),loc=8, borderaxespad=1,ncol = 6,frameon=False,fancybox=True)
        sns.despine(left = True)
        plt.show()
    date_lst  = pd.date_range('20190401','20221115',freq='M').to_period('M').astype('str')
    date_lst = [str_[0:4]+'年' +str_[5:7]+'月' for str_ in date_lst]
    combine_fund_vol = np.arange(1,44)
    stock_fund_vol = np.arange(100,143)
    get_chart(date_lst,combine_fund_vol,stock_fund_vol)

global_stock_index_chart()
get_turnover_rate()
get_trading_volume()
get_finance_chart()
get_fund_volume()

