import MetaTrader5 as mt5
import json
import time
import pytz
import pandas as pd
from datetime import datetime

K1 = 0.2
K2 = 0.2
xlot = 0.1  # 下单手数
slPrices = 50  # 止损
tpPrices = 50  # 止盈
magic = 234000  # magic编号
currencyList = ["EURUSDm", "GBPUSDm", "USDCHFm", "USDJPYm", "USDCADm", "CHFJPYm", "AUDUSDm", "EURGBPm", "CADJPYm",
                "GBPJPYm", "EURJPYm", "AUDJPYm", "NZDUSDm"]
"""
执行下单方法
symbol     #货币对名称
actiontype #订单类型
lot        #下单手数
order_type #订单类型
sl, tp     #止损，止盈
"""


def OrderSend_Task(symbol, actiontype, lot, order_type, price, tp, sl, order_fs):
	deviation = 5  # 滑点
	zhiying = 0
	if (order_fs == 1):  # 多单
		zhiying = price + tp * mt5.symbol_info(symbol).point
	elif (order_fs == 2):  # 空单
		zhiying = price - tp * mt5.symbol_info(symbol).point
	print("止损=", 0, "止盈=", zhiying, "挂单价位=", price)
	request = {
		"action": actiontype,
		"symbol": symbol,
		"volume": lot,
		"type": order_type,
		"price": price,
		# "sl": zhisun,
		"tp": zhiying,
		"deviation": deviation,
		"magic": 234000,
		"comment": "python script open",
		"type_time": mt5.ORDER_TIME_GTC,
		"type_filling": mt5.ORDER_FILLING_RETURN,
	}
	# 发送交易请求
	result = mt5.order_send(request)
	# # 检查执行结果
	print("1. 发送订单:  {} {} 手数 价格 {}  滑点={} points".format(symbol, lot, price, deviation))
	print("2. 订单返回结果=", result.retcode)
	answer = result.retcode
	if (answer == 10006):
		print("拒绝请求!")
	elif (answer == 10007):
		print("交易者取消请求!")
	elif (answer == 10009):
		print("下单完成!")
	elif (answer == 10009):
		print("############{}下单完成###############!", format(symbol))
	elif (answer == 10010):
		print("请求部分完成!")
	elif (answer == 10011):
		print("请求处理错误!")
	elif (answer == 10011):
		print("超时取消请求!")
	elif (answer == 10013):
		print("无效请求!")
	elif (answer == 10014):
		print("请求中无效成交量!")
	elif (answer == 10015):
		print("请求中的无效价格!")
	elif (answer == 10016):
		print("请求中的无效访问!")
	elif (answer == 10026):
		print("服务器无效自动交易!")
	elif (answer == 10027):
		print("客户端无效自动交易!")
	elif (answer == 10030):
		print("无效命令填满字节!")
	return

# 建立MetaTrader 5到指定交易账户的连接
if (mt5.initialize(login=28210, server="Exness-MT5", password="XXXXXX") == False):
   print("连接失败code =", mt5.last_error())
   quit()
else:
   CuttenTime = True
   while CuttenTime:
      localtime = time.localtime(time.time())
      time.sleep(5)
      if (localtime.tm_hour == 8):
         for item in currencyList:
            # 获取5日内的柱状图数据
            rates = mt5.copy_rates_from_pos(item, mt5.TIMEFRAME_H4, 0, 5)
            #次数省略了部分策略代码
            print(item, "多单=", BuyLine)
            print(item, "空单=", SellLine)
            OrderSend_Task(item, mt5.TRADE_ACTION_PENDING, xlot, mt5.ORDER_TYPE_BUY_STOP, BuyLine, tpPrices,
                           slPrices,
                           1)  # 发送多单挂单
            OrderSend_Task(item, mt5.TRADE_ACTION_PENDING, xlot, mt5.ORDER_TYPE_SELL_STOP, SellLine, tpPrices,
                           slPrices,
                           2)  # 发送空单挂单
         CuttenTime = False
         print("时间=", localtime.tm_hour, localtime.tm_sec)