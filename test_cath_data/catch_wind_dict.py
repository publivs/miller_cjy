from numpy import empty
import pandas as pd
import numpy as np
import re
# import bs4
import json
import requests
import time
import sys,os
# catch_handler
# import cry

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(r"C:\Users\kaiyu\Desktop\miller")
from chenqian_tools.hdf_helper import *

def connect_tree_url(id):
     param = f'''%5B%22%5C%22{id}%5C%22%22%2C%22%5C%229490%5C%22%22%2C%22%5C%22%E6%8A%95%E5%86%B3%E9%A1%B9%E7%9B%AE%E7%BB%842%5C%22%22%2C%221%22%2C%22true%22%5D'''
     paylaod = {'way':'1','action':'2','interfaceId':'101','param':param,}
     res = requests.post(f'''{handler}''',data=paylaod,headers=req_headers)
     res = json.loads(res.text)
     return res

def connect_table_info_url(table_name):
     param = f'''%5B%22%5C%22{table_name}%5C%22%22%2C%22%5C%229490%5C%22%22%5D'''
     paylaod = {'way':'1','action':'2','interfaceId':'103','param':param,}
     res = requests.post(f'''{handler}''',data=paylaod,headers=req_headers)
     res = json.loads(res.text)
     return res

sleep_time = 0.1
cookie = '''
c600ae004e004e00a60076002e00ca0086003600a600=0c00;
 4a0022006200aa00ce00a6004e0072008600b600a6007200a600ee00=8c000c00ec001c00ec00;
 8200c600c600f600ae0076002e0092002200=8c000c00ec001c00ec00; aa00ce00a6004e0072008600b600a600=a946cd8a9e19776e237e2c00;
 8200c600c600f600ae0076002e00920022002a009e000e00a600=ea002200ca00; 4a0022006200320086007600e600ae008600e600a600=c2007600;
 ea002200ca00ca00a600ce00ce009600f600760092002200=6600ec00a600ac00ac006c00c600cc00b40026006c00cc004600b4002c001c00ec001c00b4001c0086004c00ec00b40046000c00460066006600a6006c0086000c00ac008c008c00'''
cookie = cookie.replace(' ','').replace('\n','')
# cookie = cookie.replace('\n','')
req_headers = {
                'USER_AGENT':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
                'Cookie':cookie,
                'Connection':'keep-alive',
                'Content-Type':'application/x-www-form-urlencoded; charset=UTF-8',
                }

# 详细数据
handler = '''https://wds.wind.com.cn/rdf/Home/ServiceHandler'''

pay_load_pp = '''MN0Zugwk9+EynjaUhkY7zv9Eq9brY4Ttw5VDir5uvlZX7acRUZ26lYDv98m98FT857WHL6YPoDTO55w5x2y+43cju3zqesW3tsS6htVm7BFhQyZYLM7MALA1TJcGwB43v9O5Ug1otpTpBFEOlDJiNrYPvetc8UIlYdlQ/hnn3mE='''
pay_load_query = {'way':'1','action' :'2','interfaceId':'103','params':pay_load_pp}
# res = requests.post(f'''{handler}''',data=pay_load_query,headers=req_headers)
# res = json.loads(res.text)


# 获取基础的目录树
menu_paylaod_pp = '''EWEN2czyzRIPv64PNRsKlf1zRrmm3k4iBU5UVXON43f46Y57wjrHgPDdIAiZWQy1Fz+GQH3Oud7vqn9eDqHeGy/XBLLec2G3fs5notBT1ga47/7vEDBnC9prQ1LMpQl6wKfybm4QRjtuNqL5fcmnR58ygOsvcThE7vWOHG9qQogCrGyayToQ6bj7qMTO5Mfg+5K6SicorJ1tyG5qFF+QWLKlPza5IzHJbZBsoSZ/hMZ/qG8spi5a3qxtIN/Ik3K6GH2HGbGukcdpxvlpXUc75V4VEoOVKo4/7rrHqgaNVfX2llhmdX1+WF8mZNjaJOp1toS3kvuUe5XiXYa6oToNN553H48fznaIjixwVYpHzVCmNNUUyGWmUC3T7kVn/BTCXOpuweFZaRcg3a3QroHXmTvFU/R289qwCgriTQsd41C1vzHo8zF5Ks83HgRSBpY3dLW1Z3DKkVTuDUmwmxYtFEzbLiLvP6xbENbeCjhu3vDiGvubJOHqF5LeTPP+wVjE'''
menu_paylaod = {'way':'1','action':'2','interfaceId':'101','param':menu_paylaod_pp,}
menu_res = requests.post(f'''{handler}''',data=menu_paylaod,headers=req_headers)
res = json.loads(menu_res.text)

def initial_path(tree_nodes,res_path ='wind_datafile',):

     if isinstance(tree_nodes,list):
          for tree_node in tree_nodes:
               initial_path(tree_node,res_path)

     if isinstance(tree_nodes,dict):
          if 'Content' in tree_nodes.keys():
               content = tree_nodes['Content']
               if 'Data' in content.keys():
                    data = content['Data']
                    initial_path(data,res_path)

          if 'DataType' in tree_nodes.keys():
               if tree_nodes['DataType'] == 1:
                    get_table_info_main(tree_nodes,res_path)
                    time.sleep(sleep_time)

               if tree_nodes['DataType'] == 0:
                    id = tree_nodes['ID']
                    data = connect_tree_url(id)
                    append_path  = '\\'+ tree_nodes['Text']
                    append_path = append_path.replace('/','_')
                    res_path = res_path + append_path
                    if  not os.path.exists(res_path):
                         os.mkdir(res_path)
                    initial_path(data,res_path)

def get_table_info_main(tree_nodes,res_path):

          def update_qa_info(QA,res_path):
               str_q = '问题:'+ QA['Name'] + '\n'
               str_a = '回答:' + QA['ContentNoHtml']  +'\n'
               str_qa_info = str_q+str_a + ' \n \n \n'
               # # # 写入部分 # # #
               with open(f'''{res_path}_QA.txt''','a',encoding='utf-8') as f:
                    f.write(str_qa_info)
               return str_qa_info

          ID_ = tree_nodes['ID']
          next_nodes = connect_table_info_url(ID_)
          table_data = next_nodes['Content']['Data']

          filed_df = pd.DataFrame(table_data['FieldList'])

          other_data = {k:v for k,v in table_data.items() if k not in ('FieldList','TopicList','SampleData')}
          other_df = pd.DataFrame.from_dict(other_data,orient = 'index').T
          other_df = other_df.astype('str')

          if table_data['SampleData'].__len__() > 0 :
               sample_data = json.loads(table_data['SampleData'])
               sample_df = pd.DataFrame(sample_data['rows'],columns=sample_data['fieldName'])
          else:
               sample_df = pd.DataFrame()

          append_path = '\\'+ tree_nodes['Text']
          append_path = append_path.replace('/','_')

          h5_path = res_path + append_path

          qa_data = table_data['TopicList']

          if qa_data is not None:
               for qa in qa_data:
                    update_qa_info(qa,h5_path)

          h5_group_path = h5_path +'.h5'
          if not os.path.exists(h5_group_path):
               h5_client = h5_helper(h5_group_path)
               h5_client.append_table(filed_df,'filed_df')
               h5_client.append_table(other_df,'other_df')
               h5_client.append_table(sample_df,'sample_df')

          print(h5_path)

all_data_set_path = 'wind_datafile'

if  not os.path.exists(all_data_set_path):
        os.mkdir(all_data_set_path)

nodes = res['Content']['Data']
initial_path(nodes,res_path =all_data_set_path,)
