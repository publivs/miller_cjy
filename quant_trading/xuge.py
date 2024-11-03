import sys,os

import requests
import pandas as pd
import numpy as np
import backtrader as bt
import time
import json
import datetime

import backtrader.indicators as btind
'''
url
获取的数据格式:OHLC,change_vol,pct_change
'''
def connect_url(target_url,req_headers):
    con_continnue = True
    while con_continnue:
        try:
            res_ = requests.get(target_url,headers=req_headers)
            if res_ is not None:
                con_continnue = False
            else:
                time.sleep(5)
                res_ = requests.get(target_url,headers=req_headers)
        except Exception as e:
            print("链接,出异常了！")
    return res_


def get_data():

    req_headers = {
                'USER_AGENT':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en',
                }

    start_time = 1533038400

    next_time_delta = 9705600

    end_time = int(pd.to_datetime(pd.Timestamp(time.time(),unit='s').date()).timestamp())

    timestamp_lst = range(start_time,end_time+next_time_delta,next_time_delta)

    res_lst = []
    for time_i in  timestamp_lst:
        df_i = get_df_data(req_headers,time_i)
        np.random.randint(0,3)
        res_lst.append(df_i)
    res_df = pd.concat(res_lst)
    res_df.tick_at = res_df.tick_at.astype('M8[s]')
    res_df = res_df.set_index('tick_at')

def get_df_data(req_headers,timestamp):

    target_url = f'''https://api-ddc-wscn.awtmt.com/market/kline?prod_code=XAUUSD.OTC&timestamp={timestamp}&tick_count=499&period_type=14400&fields=tick_at%2Copen_px%2Cclose_px%2Chigh_px%2Clow_px%2Cturnover_volume%2Cturnover_value%2Caverage_px%2Cpx_change%2Cpx_change_rate%2Cavg_px'''

    res_ = connect_url(target_url,req_headers)
    res = json.loads(res_.text)
    df_i  = pd.DataFrame(res['data']['candle']['XAUUSD.OTC']['lines'],columns=res['data']['fields'])
    # df_i['tick_at'] = df_i.tick_at.apply(lambda x:pd.Timestamp(x,unit = 's'))
    return df_i


class Mean_revert(bt.Strategy):
        params = (
                ('roll_num',55),
                ('first_price_input',95),
                ('add_position_price',40),
                ('stake',0.1),
                ('max_hold_day',120)
                )
        def log(self, txt, dt=None, doprint=True):
            ''' 日志函数，用于统一输出日志格式 '''
            if doprint:
                dt = dt or self.datas[0].datetime.date(0)
                print('%s, %s' % (dt.isoformat(), txt))

        def __init__(self) -> None:
            self.ma_55 = bt.talib.SMA(self.data.close,timeperiod=self.params.roll_num)
            self.sell_size = 0
            self.hold_days = 0
            self.first_price = 0
            self.add_position = 0   
            self.first_day = 0

        def notify_order(self,order):
            if order.status in [order.Submitted, order.Accepted]:
                return
            # 如果order为buy/sell executed,报告价格结果
            if order.status in [order.Completed]:
                if order.isbuy():
                    self.log(f'买入:\n价格:{order.executed.price},\
                    数量:{order.executed.value},\
                    手续费:{order.executed.comm}')
                    self.buyprice = order.executed.price
                    self.buycomm = order.executed.comm
                else:
                    self.log(f'卖出:\n价格：{order.executed.price},\
                    数量: {order.executed.value},\
                    手续费{order.executed.comm}')
                self.bar_executed = len(self)

            # 如果指令取消/交易失败, 报告结果
            elif order.status in [order.Canceled, order.Margin, order.Rejected]:
                self.log('交易失败')
                print(order.status,order.Margin)
            self.order = None

        def notify_trade(self, trade):

            if not trade.isclosed:
                return

            self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                    (trade.pnl, trade.pnlcomm))  # pnl：盈利  pnlcomm：手续费

        def check_hold_days(self):
            if self.datetime.date(0) == self.first_day:
                pass
            else:
                self.hold_days = (self.first_day - self.datetime.date(0)).days + 1

        def next(self):
            if not self.position : # 无持仓的情况下
                if self.data.close[0] > (self.ma_55[0] + self.params.first_price_input):
                    self.order = self.sell(size = 0.1)
                    self.first_price = self.data.close[0] # 记录一下进场价格
                    self.hold_days = 1 # 初始化持仓日期
                    self.first_day = self.datetime.date(0) # 记录第一笔单子进场


            else:
                self.hold_days = self.check_hold_days()
                # 有持仓的情况下
                # 加仓逻辑1
                if self.data.close[0]>(self.first_price + 40)>(self.ma_55[0]+100):
                    self.order = self.sell(size = 0.1)

                    self.add_position = 1
                # 加仓逻辑2
                elif (
                    (self.data.close[0]> (self.first_price + 100)) and (self.data.close[0]> (self.ma_55[0] + 100))
                    ):
                    self.order = self.sell(size = 0.2)
                    self.add_position = 1
                # 清仓逻辑
                if self.add_position != 0:
                    if self.data.close[0]< self.ma_55[0] - 40:
                        self.log("现价<均线-40美金,退出...")
                        self.close()

                # 单仓设定止盈600
                elif self.add_position == 0:
                    if self.broker.orders[0].executed.pnl >= 600:
                        self.close()
                        self.log("单笔盈利超过600,退出止盈...")

                elif self.hold_days>120:
                        self.log("持仓周期120天止盈...")
                        self.close()
                else:
                    pass


def strategy_main():
    cerebro = bt.Cerebro()
    mod_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    data_path = mod_path+'\\xauusd_4h_from_20180731.csv'

    df = pd.read_csv(data_path,parse_dates=['tick_at'])

    params = dict(
                fromdate = datetime.datetime(2010,1,4),
                todate = datetime.datetime(2023,2,20),
                timeframe = bt.TimeFrame.Days,
                compression = 1,
                #dtformat=('%Y-%m-%d %H:%M:%S'),
                # tmformat=('%H:%M:%S'),
                datetime=0,
                high=2,
                low=3,
                open=1,
                close=4,
                volume=5,
                openinterest=6)

    df = pd.read_csv(data_path,encoding='gbk')
    df = df[['tick_at', 'open_px', 'high_px', 'low_px', 'close_px', 'turnover_volume', 'turnover_value',]]
    df.columns = ['datetime','open','high','low','close','volume','openinterest']
    df = df.sort_values("datetime")
    df.index=pd.to_datetime(df['datetime'])
    df=df[['open','high','low','close','volume','openinterest']]
    feed =  bt.feeds.PandasDirectData(dataname=df,**params)
    cerebro.adddata(feed)
    cerebro.addstrategy(Mean_revert)
    cerebro.broker.setcommission(
                                margin=1, # 必须为1 automargin=1000,# 不同货币的保证金可以使用公式计算得出   
                                mult=100.0, # 100是杠杆的倍数，1000固定
                                ) # 设
    cerebro.broker.setcash(10000)
    cerebro.broker.tradehistory = True
    cerebro.run()
    cerebro.plot(style='candlestick')


strategy_main()
