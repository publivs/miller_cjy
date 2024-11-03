import pandas as pd
import numpy as np
import akshare as ak
import time

from hdf_helper import *

# 实时行情数据
# stock_zh_index_spot_df = ak.stock_zh_index_spot()
# print(stock_zh_index_spot_df)

# 股票指数历史行情数据
# stock_zh_index_daily_df = ak.stock_zh_index_daily_tx(symbol="000001")
# print(stock_zh_index_daily_df)

# 获取所有指数信息
index_stock_info_df = ak.index_stock_info()
print(index_stock_info_df)

h5_path = 'indexQuote_daily'
# 历史行情数据-通用
h5_client = h5_helper(h5_path)
h5_client.append_table(index_stock_info_df,'index_info')
fail_lst = []
for index_code in index_stock_info_df.index_code.unique():
    print(f'正在处理指数{index_code}的数据......')
    try:
        index_zh_a_hist_i = ak.index_zh_a_hist(symbol=index_code,period="daily")
    except:
        index_zh_a_hist_i = pd.DataFrame()
        fail_lst.append(index_stock_info_df.query('''index_code == '{index_code}' '''))
    if not index_zh_a_hist_i.empty:
        index_zh_a_hist_i['指数代码'] = index_code
        h5_client.append_table(index_zh_a_hist_i,'quote_data')
        sl_time = np.random.randint(5,7)
        time.sleep(sl_time)
    else:
        pass
fail_df = pd.concat(fail_lst)
fail_df.to_csv('fail_df.csv')


