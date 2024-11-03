import pandas as pd
import numpy as np 
from collections import OrderedDict
import asyncio
import time

'''
在这里用计算流调异步进行计算
'''

cols = ['A','B','C',"D","E","F"]
mat = np.random.uniform(0,10,(1000000,6))
df = pd.DataFrame(mat)
df.columns = cols


dic_list = [
    {"index_id":1,
    'name':'alpha',
    'eq':'''reg_monitor(df['A'],df['B'])'''
    },
        {"index_id":2,
    'name':'beta',
    'eq':'''reg_monitor(df['A'],df['C'])'''
    },

            {"index_id":3,
    'name':'theta',
    'eq':'''reg_monitor(df['A'],df['D'])'''
    },

                {"index_id":4,
    'name':'theta',
    'eq':'''reg_monitor(df['A'],df['E'])'''
    },
    ]


def reg_monitor(A,B):
    moni = np.sqrt((A+B))
    time.sleep(3)
    return moni

# res_lst = []
# async def calc_cactors(df_io,calc_dict):
#     df = df_io.copy(deep=True)
#     res_dict = dict()
#     res_dict[calc_dict['index_id']] = eval(calc_dict['eq'])
#     res_lst.append(res_lst)


# result_df = pd.DataFrame()
# loop = asyncio.get_event_loop()
# task_lst = []
# for calc_dict in dic_list:
#     task_lst.append(
#         asyncio.ensure_future(calc_cactors(df,calc_dict))
#     )

# start = time.time()
# loop.run_until_complete(asyncio.wait(task_lst))
# endtime = time.time()-start
# print(endtime,'异步协程计算时间')
# loop.close()
# print(res_lst)

# result_df = pd.DataFrame()
# start = time.time()
# for calc_dict in dic_list:
#     res = eval(calc_dict['eq'])
#     result_df[calc_dict['index_id']] = res
# endtime = time.time()-start
# print(result_df.tail(5))
# print(endtime,'串行计算时间')

# 计算密集型任务除非特别耗时的回归运算，简单的shift,pct_change这种完全没有并行必


import math
import datetime
import multiprocessing as mp


def calc_index(df,calc_dict,result_df):
    index_id = calc_dict['index_id']
    res = eval(calc_dict['eq'])
    result_df[index_id] = res
    return result_df

def to_result_df(res):

    result_df

if __name__ =='__main__':

    result_df = pd.DataFrame()
    start_t = datetime.datetime.now()

    num_cores = int(mp.cpu_count())
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(3)

    pool_list = []
    results = []
    for  calc_dict in dic_list:
        eq = calc_dict['eq']
        # 这里get会阻塞进程
        pool_list.append(pool.apply_async(calc_index, args=(df.copy(),calc_dict,result_df)))

    pool.close()
    pool.join()
    result_list = [xx.get() for xx in pool_list]
    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("多进程计算 共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")
    print(pd.concat(result_list,axis=1))


'''
基本完事，但是后续是一个二级账户一个计算流，所以用多进程搞

'''