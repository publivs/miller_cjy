import redis
import pandas as pd 
import numpy as np
import json
import pickle

ip = '127.0.0.1'
PID = '5632'
password = ''

r1=redis.Redis(host=ip,
                # password= password,
                port=6379,db=0,
                encoding='utf-8',
                decode_responses=False)#redis默认连接db0

# decode_responses 为False 这里为False会有啥效果

print(r1.get('name'))

# 利用管道,请求式的进行数据通信

# ----------------------------- pickle_部分 -------------------------------- #

a = 1
# 保存
data =1
with open('data.pickle', 'wb') as f:
  pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
# 读取
with open('data.pickle', 'rb') as f:
  b = pickle.load(f)

# ------------------------------------------------------------------------- #

# 输入列表
df_dict ={
        'se':{'col_1':[1,2,3,4],
                'col_1':[1,2,3,4]},
        'pivot':{'df_1':pd.DataFrame(),
                'df_2':pd.DataFrame()}
        }

columns =['A','B','C']
df_1 = pd.DataFrame(None,index=[1,2,3,4,5,6,7,8,9,10],columns=['A','B','C']).fillna(0)
df_2  = df_1.copy(deep=True)
df_dict ={
        'se':{'col_1':[1,2,3,4],
                'col_1':[1,2,3,4]},
        'pivot':{'df_1':df_1,
                'df_2':df_2}
        }

# 对于列表中的dict
res_dict_list = [df_1,df_2]
# 列表是不可迭代,还是老老实实转成这样的

all_mid_args_dict = {}
all_mid_args_dict['res_dict_list'] = res_dict_list

picklst_str = pickle.dumps(all_mid_args_dict)
r1.set('picklst_str',picklst_str)
r1.get('picklst_str')


class redis_helper:

        def __init__(self,cfg_dict,):
                self.server_ip = cfg_dict['server_ip']
                self.port = cfg_dict['server_ip']
                self.db = cfg_dict['db'] if cfg_dict['db'] is not None else 0
                self.encoding = cfg_dict['encoding'] if cfg_dict['encoding'] is not None else 'utf-8'
                self.pwd = cfg_dict['password'] if cfg_dict['password'] is not None else None
                self.decode_responses = cfg_dict['decode_responses']  if cfg_dict['decode_responses'] is not None else False
                self.redis_con = redis.Redis(   host= self.server_ip,
                                                password= self.pwd,
                                                port=self.port,
                                                db= self.db,
                                                encoding= self.encoding,
                                                decode_responses=self.decode_responses
                                                )

        # 利用事务进行提交
        def save(self,name,value,pipe_msg = 1):
                value_2_redis  = pickle.loads(value)
                self.redis_con.set(name,value_2_redis)
                if pipe_msg == 1:
                        pipe = self.redis_con.pipeline(transaction=True)
                        pipe.set(name,value_2_redis)
                        res = pipe.execute()

        # 利用事务进行提交
        def read(self,name,encoding,pipe_msg = 1):
                if encoding is None:
                        encoding = self.encoding
                self.redis_con.get(name)
                if pipe_msg == 1:
                        pipe = self.redis_con.pipeline(transaction=True)
                        pipe.get(name)
                        res = pipe.execute()

