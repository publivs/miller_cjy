
from numpy import empty
import pandas as pd
import numpy as np
import re
# import bs4
import json
import requests
import time
from hdf_helper import *
import copy

def initial_file_path(tree_nodes,
                    res_path = 'Juyuan_datafile',
                    next_level_key = 'nodes',
                    path_key = 'groupName',
                    table_name_key  = 'tableName',
                    table_id_key = 'id'):

    #
    initial_path = res_path
    next_level_key = next_level_key
    path_key = path_key
    table_name_key = table_name_key
    table_id_key = table_id_key

    # 传进来是列表对象就继续遍历子节点
    if isinstance(tree_nodes,list):
            for i in  range(tree_nodes.__len__()):
                    sub_node_i = tree_nodes[i]
                    res_path = initial_file_path(sub_node_i,res_path,next_level_key,path_key,table_name_key)

    # 不是列表继续搜索,搜到列表位置
    else:
        write_mark = '0'
        if tree_nodes[next_level_key] is None:
            write_mark = '1'
        else:
            if (tree_nodes[next_level_key].__len__() > 0):
                pass
            else:
                write_mark = '1'

        if write_mark == '0':
                if res_path is None:
                    res_path = initial_path
                append_path  = '\\'+ tree_nodes[path_key]
                append_path = append_path.replace('/','_')
                res_path = res_path + append_path
                # print(res_path)
                sub_nodes = tree_nodes[next_level_key]
                if not os.path.exists(res_path):
                    os.mkdir(res_path)
                res_path = initial_file_path(sub_nodes,res_path,next_level_key,path_key,table_name_key)
                return res_path

        # nodes一般是大于0的，如果为0那么基本是遍历到基层的子节点了,在这里实例化HDFS对象存数据
        else:
            # print('准备实例化_hdf5!!!')
            table_id = tree_nodes['MODEL_ID']
            obj_id = tree_nodes['OBJ_ID']

            table_name = tree_nodes['OBJ_NAME']
            table_cn_name = tree_nodes['OBJ_C_NAME']
            if table_cn_name is None:
                pass
            else:
                table_cn_name = table_cn_name.replace('/','_').replace('\\','_')
            h5_path = f'''{res_path}\\{table_cn_name}_{obj_id}_{table_name}'''
            print(h5_path)
            # if obj_id  :
            #     print(obj_id)

            if not os.path.exists(h5_path+'_'+'table'+'.h5'):
                    # sleep_time = time.sleep(np.random.randint(0,1))

                    try:
                        catch_caihui_main(tree_nodes,req_headers,h5_path)
                    except:
                        print(f'{table_cn_name}_{obj_id}_{table_name},数据有问题...')
            return res_path

    last_path = '\\'+res_path.split('\\')[-1:][0]
    res_path = res_path.replace(last_path,'')
    return res_path

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

def catch_caihui_main(table_info,req_headers,h5_group_path_rela):

    def generate_enum_df(fields_df):
        # COL对应的枚举详情
        enum_df_lst = []
        for index,df_i in fields_df.iterrows():
            if df_i['ENUMS'].__len__() > 0:

                enum_df_i = pd.DataFrame(df_i['ENUMS'])
                enum_df_i['table'] = df_i['C_OBJNAME']
                enum_df_i['col_id'] = df_i['COL_ID']
                enum_df_i['col_name'] = df_i['COLNAME']
                enum_df_i['obj_id'] = df_i['OBJ_ID']
                enum_df_i['model_id'] = df_i['MODEL_ID']
                enum_df_lst.append(enum_df_i)
        if enum_df_lst.__len__() > 0 :
            enum_df = pd.concat(enum_df_lst,ignore_index=True)
        else:
            enum_df = pd.DataFrame()
        return enum_df

    table_id = table_info['MODEL_ID']
    obj_id = table_info['OBJ_ID']
    base_url  = f'''https://datadict.finchina.com/api/DataStru/DataObject?modelID={table_id}&objectID={obj_id}'''
    res_ = connect_url(base_url,req_headers)
    res = json.loads(res_.text)

    if res is not None:
        if 'code' in res.keys():
            if res['code'] == 401:
                print(table_info['MODELNAME'],''' 有数据无法拉取...''')
                print(res['msg'])
                return
        # 数据的DF
        table_info_df = pd.DataFrame.from_dict({k:v for  k,v in res.items() if k not in ['TABLEDATA','FIELDS']},orient = 'index').T
        fields_df = pd.DataFrame(res['FIELDS'])
        fields_df_enum = fields_df.loc[:,~fields_df.columns.str.contains('ENUMS')]
        enum_df = generate_enum_df(fields_df)

        try:
            example_url =f'''https://datadict.finchina.com/api/DataStru/DataObject?modelID={table_id}&objectID={obj_id}&datatype=1'''
            res_ = connect_url(base_url,req_headers)
            res_example = json.loads(res_.text)
        except:
            res_example = {}

        method = 'xls'
        output = 1

        if method == 'h5':
            h5_client = h5_helper(h5_group_path_rela+'_'+'table'+'.h5')
            if output != 0:
                if res_example['TABLEDATA'] is not None:
                    if str(res_example['TABLEDATA']).__len__() > 5:
                        example_table_df = pd.DataFrame(json.loads(res_example['TABLEDATA']))
                        if not example_table_df.empty:
                            h5_client.append_table(example_table_df,'example_table_df')

                if not table_info_df.empty:
                    table_info_df = table_info_df.astype('str')
                    h5_client.append_table(table_info_df,'table_info_df')

                if not fields_df.empty:
                    h5_client.append_table(fields_df_enum,'fields_df')

                if not enum_df.empty:
                    h5_client.append_table(enum_df,'enum_df')
            else:
                with open('out_put.txt', 'a') as f:
                    f.write(f'{h5_group_path_rela}\n')

        if method == 'xls':

            df_data_save = fields_df
            all_df_lst.append(df_data_save)


    else:
        print(table_id,obj_id,'''该表出问题''')


global all_df
all_df_lst  = []

tree_url = 'https://datadict.finchina.com/api/DataStru/Tree?IsDeep=true'

base_url = ''

cookie = '''Authorization=eyJVc2VySUQiOjI3Mn0=.NEMtMzctQkItNkMtODItNkEtOUUtOTUtMUItQTgtOUItQTQtOTUtNzQtQTQtRUQ='''

# Override the default request headers:
req_headers = {
                'USER_AGENT':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en',
                'Cookie':cookie,
                }

# 数据筛选层
res_ = requests.get(tree_url,headers=req_headers)
res_all_tree = json.loads(res_.text)
all_dataset_name = res_all_tree

## ---------------------------------------------------------- ##
all_data_set_path = 'CaiHui_datafile'
    # os.remove(all_data_set_path)
if not os.path.exists(all_data_set_path):
        os.mkdir(all_data_set_path)
# else:
#     os.remove(all_data_set_path)

next_level_key = 'CHILDREN'
path_key = 'MODEL_NAME'

initial_file_path(all_dataset_name['CHILDREN'][0],
                    res_path = all_data_set_path,
                    next_level_key = 'CHILDREN',
                    path_key = 'MODELNAME')


file_path = 'target_sheet_.xlsx'
f = all_df_lst
result = pd.concat(f, axis=0,ignore_index=True)  # 将两个文件concat，也就是合并
if not os.path.exists(file_path):
    result.to_excel(file_path, index=False,encoding ='gbk',)
# else:
#     with pd.ExcelWriter(file_path, engine='openpyxl', mode='a',if_sheet_exists='replace') as writer:
#         df1 = pd.read_excel(file_path) # 由于没有找到好的方法，所以我们读出之前文件的内容

#         result.to_excel(writer,index=False) # 保存 注意：index_label必须要和上面的index_col相同，不然下次读文件的时候会出index_col不存在的错误
# table_info = {'MODEL_ID':207,'OBJ_ID':203,}

# catch_caihui_main(table_info,req_headers,'h5_group_path_rela')