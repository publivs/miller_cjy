from pydoc import describe
from numpy import empty
import pandas as pd
import numpy as np
import re
# import bs4
import json
import requests
import time
import sys,os

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(r"C:\Users\kaiyu\Desktop\miller")
from chenqian_tools.hdf_helper import *

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

def initial_file_path(select_df,all_df,res_path = 'all_data_set_path'):
    need_sub = '0'
    if select_df.__len__()>1:
        need_sub = '1'
    # if (select_df.sequence == 1).any():
    #     need_sub = '1'
    if need_sub != '0':
        for i in  range(select_df.__len__()):
            select_df_i = select_df.iloc[i].to_frame().T
            res_path,all_df = initial_file_path(select_df_i,all_df,res_path)

    else:
    # 不等於0
        write_mark = '0'
        if (select_df.display_type >0).any():
            write_mark = '1'
        if write_mark == '0':
            father_id = select_df['guid'].iloc[0]
            append_path  = '\\'+ select_df['menu_name'].iloc[0]
            append_path = append_path.replace('/','_')
            res_path = res_path + append_path
            # print(res_path)
            if not os.path.exists(res_path):
                os.mkdir(res_path)
            select_df = all_df.loc[all_df.father_guid == father_id]

            res_path,all_df = initial_file_path(select_df,all_df,res_path)
            # 针对一个节点下之后一个的，回退一次路径
            if select_df.__len__() == 1:
                    last_path = '\\'+res_path.split('\\')[-1:][0]
                    res_path = res_path.replace(last_path,'')
            return res_path,all_df
        else:
            guid = select_df['guid'].iloc[0]
            table_cn_name = select_df['menu_name'].iloc[0]
            if table_cn_name is None:
                pass

            h5_path = f'''{res_path}\\{table_cn_name}'''
            print(h5_path)
            if not os.path.exists(h5_path+'_'+'table'+'.h5'):
                    sleep_time = time.sleep(np.random.randint(3,4))
                    append_table_main(guid,h5_path)
            return res_path,all_df

    last_path = '\\'+res_path.split('\\')[-1:][0]
    res_path = res_path.replace(last_path,'')
    return res_path,all_df



# 
def append_table_main(guid,h5_group_path_rela):

    def process_field_describe(need_append_describe):
        data_lst = []
        for guid_id in need_append_describe:
            field_describe_url = f'''http://gogoaldata.go-goal.cn/api/v1/dd_data/get_field_describle?guid={guid_id}'''
            res_ = connect_url(field_describe_url,req_headers)
            if res_ is not None:
                res = json.loads(res_.text)
                df = pd.DataFrame.from_dict(res['data'],orient = 'index').T
                data_lst.append(df)
        if len(data_lst) > 0:
            return pd.concat(data_lst,ignore_index= True)
        else:
            return pd.DataFrame()

    target_url = f'''{table_info_url}?table_type=0&guid={guid}'''
    table_info = connect_url(target_url,req_headers)
    res = json.loads(table_info.text)

    table_field_df = pd.DataFrame(res['data']['table_field'])
    describe_df = pd.DataFrame.from_dict(res['data']['table_describle'],orient = 'index').T

    need_append_describe = table_field_df.loc[table_field_df.display_description == 1]['guid']
    fields_describe_df = process_field_describe(need_append_describe)
    fields_describe_df  = fields_describe_df.astype('str')

    h5_client = h5_helper(h5_group_path_rela+'_'+'table'+'.h5')

    table_en_name = res['data']['table_describle']['table_name']
    example_url = f'''http://gogoaldata.go-goal.cn/api/v1/dd_data/get_table_content?table_name={table_en_name}'''
    exmaple_text = connect_url(example_url,req_headers)
    exmaple_text = json.loads(exmaple_text.text)
    if exmaple_text['code'] == 500:
        exmaple_df = pd.DataFrame()
    else:
        exmaple_df = pd.DataFrame(exmaple_text['data'])
        exmaple_df.columns = [str__.strip('[]') for str__ in exmaple_df.columns]

    # 枚举描述
    h5_client.append_table(fields_describe_df,'fields_describe_df')

    # 表字段信息
    h5_client.append_table(table_field_df,'table_field_df')

    # 表描述
    describe_df = describe_df.astype('str')
    h5_client.append_table(describe_df,'describe_df')

    # 表样例信息
    exmaple_df = exmaple_df.astype('str')
    h5_client.append_table(exmaple_df,'exmaple_df')

    sleep_time = time.sleep(np.random.randint(3,4))



base_url = 'http://gogoaldata.go-goal.cn/'

cookie = '''acw_tc=0bca294216575900140796892e017889409b7ab583aa05c3ae92913f6c768d; session=764981c035524399bccc98ae49fcc5651657590025074; web=764981c035524399bccc98ae49fcc5651657590025074; tk=764981c035524399bccc98ae49fcc565'''

# Override the default request headers:
req_headers = {
                'USER_AGENT':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
                'Cookie':cookie,
                'Connection':'keep-alive',
                'Content-Type':'application/x-www-form-urlencoded; charset=UTF-8',
                }

# 全分支的节点树
all_tree_url = f'http://gogoaldata.go-goal.cn/api/v1/dd_data/get_header'

#
pay_load_query = '?table_type=0&org_id=45513&function_code=103&user_id=100468768'

# pay_load_query = {
#                     'org_id':"45513",
#                     'function_code':"103",
#                     'user_id':"100468768",
#                     }

# pay_load_query = json.dumps(pay_load_query)

# 获取主节点的信息
res_all_tree = requests.get(f'''{all_tree_url}{pay_load_query}''',headers=req_headers)
res_all_tree = json.loads(res_all_tree.text)

mother_tree = res_all_tree['data']

all_data_set_path = 'zhaoyang_datafile'
    # os.remove(all_data_set_path)
if  not os.path.exists(all_data_set_path):
        os.mkdir(all_data_set_path)

table_info_url = 'http://gogoaldata.go-goal.cn/api/v1/dd_data/get_table_struct'

# 里面文档的顺序依赖father_guid和guid
mother_df = pd.DataFrame(mother_tree)
mother_df_1 = mother_df.loc[mother_df.level == 1]
for i in range(mother_df_1.__len__()):
    select_df = mother_df_1.iloc[i].to_frame().T
    initial_file_path(select_df,mother_df,res_path = all_data_set_path)



    # 根据 level_id判断path6
    # print(label['menu_name'])
    # print(label['level'])
    # print(label['group_id']) #
    # print(label['guid']) #  请求的核心
