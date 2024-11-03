from numpy import empty
import pandas as pd
import numpy as np
import re
# import bs4
import json
import requests
import time
from sqlalchemy.dialects.oracle import BFILE, BLOB, CHAR, CLOB, DATE, DOUBLE_PRECISION, FLOAT, INTERVAL, LONG, NCLOB, NUMBER, NVARCHAR, NVARCHAR2, RAW, TIMESTAMP, VARCHAR, VARCHAR2
from sqlalchemy import create_engine

from sqlalchemy import types
from sqlalchemy import null
from hdf_helper import *
import cx_Oracle as cx

def init_engine():
    engine = create_engine(
    '''oracle+cx_oracle://{username}:{password}@{host}:{port}/?service_name={service}'''.format(
        username="idb_gil",
        password="70edf78d8f31cadc",
        host="192.168.0.18",
        port=1521,
        service="ORCL",
        encoding='utf-8',
        pool_size=4,
        # max_overflow=15,
        # pool_timeout=30,
        echo=True,
    # pool_pre_ping=True
    )
    )
    return engine



# pd.read_sql("SELECT * FROM IDB_GIL.BOND_ABSBASICINFO ",engine)
def get_url_data(table_id,req_headers,base_url,table_url,data_struc_type='0',dict_sub_key = ''):

        con_continnue = True
        while con_continnue:
            # try:
            #     res_ = requests.get(base_url+table_url,headers=req_headers)
            try:
                res_ = requests.get(base_url+table_url,headers=req_headers)
                if res_ is not None:
                    con_continnue = False
                else:
                    time.sleep(5)
                    res_ = requests.get(base_url+table_url,headers=req_headers)
            except Exception as e:
                print("链接,出异常了！")

        res_.encoding = 'utf-8'
        if not res_.text.__len__() == 0:
        # 标准的DF类型
            if data_struc_type == '0':
                res = json.loads(res_.text)
                res_df = pd.DataFrame(res)

        # 标准的DataFrame类型
            if data_struc_type == '1':
                res = json.loads(res_.text)
                if not dict_sub_key:
                    res_df = pd.DataFrame.from_dict(res,orient = 'index')
                else:
                    res_df = pd.DataFrame.from_dict(res[dict_sub_key],orient = 'index')

        # 返回的html静态类型
            if data_struc_type == '2':
                try:
                    res_df = pd.read_html(res_.text)[0]
                except:
                    print(base_url+table_url,'html出错......')
                    res_df = pd.DataFrame()
                    return res_df

            return res_df
        else:
            return pd.DataFrame()

def get_all_table_info(table_id,header,base_url):

    table_info_url = f'/api/table/{table_id}'
    column_url = f'/api/column/{table_id}'
    slave_column_url = f'/api/slaveColumn/{table_id}'
    table_index_unique_url = f'/api/tableIndexByUnique/{table_id}'
    table_change_desc_url = f'/api/tableChangeDesc/query/{table_id}'
    table_modify_date_url = f'/api/getTableModifyDate/{table_id}'

    example_info_url =  f'/api/exampleData/readExampleHtml/{table_id}'

    Q_A_info_url = f'/api/qa/query/{table_id}'

    table_info_df = get_url_data(table_id,header,base_url,table_info_url,'1','data').T
    column_info_df = get_url_data(table_id,header,base_url,column_url)
    slave_column_info_df = get_url_data(table_id,header,base_url,slave_column_url)
    table_index_unique_df = get_url_data(table_id,header,base_url,table_index_unique_url,'1').T
    table_change_desc_df = get_url_data(table_id,header,base_url,table_change_desc_url)

    table_modify_date_df = get_url_data(table_id,header,base_url,table_modify_date_url,'1').T

    example_info_df = get_url_data(table_id,header,base_url,example_info_url,'2')
    Q_A_info_df = get_url_data(table_id,header,base_url,Q_A_info_url)
    # 涉及的QA部分的数据
    return table_info_df,column_info_df,slave_column_info_df,table_index_unique_df,table_change_desc_df,table_modify_date_df,example_info_df,Q_A_info_df

def save_db_data(df, db, table_name, schema, chunksize=2000):
    _dtype = {c: types.VARCHAR(df[c].str.len().max()) for c in
              df.columns[df.dtypes == 'object'].tolist()}
    with db.engine.begin() as conn:
        df.to_sql(table_name,
                  conn,
                  if_exists="append",
                  chunksize=chunksize,
                  index=False,
                  schema=schema,
                  dtype=_dtype)
    print("%s落库成功"%table_name)

def mapping_df_types(data):  # 这里要做数据类型转换的，不然插入数据库操作会报错
    dtypedict = {}
    for i, j in zip(data.columns, data.dtypes):
        if "object" in str(j):
            dtypedict.update({i: VARCHAR(256)})
        if "float" in str(j):
            dtypedict.update({i: NUMBER(19, 8)})
        if "int" in str(j):
            dtypedict.update({i: VARCHAR(19)})
        if "int64" in str(j):
            dtypedict.update({i: VARCHAR(19)})
    return dtypedict




def catch_data_main(table_info,req_headers,base_url,h5_group_path_rela):
    table_id = table_info['id']
    table_name = table_info['tableName']

    table_info_df,\
    column_info_df,\
    slave_column_info_df,\
    table_index_unique_df,\
    table_change_desc_df,\
    table_modify_date_df,example_info_df,Q_A_info_df = get_all_table_info(table_id,req_headers,base_url)

    method = 'sql'

    if method == 'xls':
        df_data_save = column_info_df
        df_data_save['tableCnName'] = table_info_df['tableChiName'].values[0]
        df_data_save['tableName'] = table_info_df['tableName'].values[0]
        all_df_lst.append(df_data_save)
        # 写入文件
        with open('out_put.txt', 'a') as f:
                    f.write(f'{h5_group_path_rela}\n')

    if method == 'sql':
        table_name = table_info_df['tableName'].iloc[0]


        # dtypedict = mapping_df_types(example_info_df)
        # def process_text_col(df):
        clob_col = column_info_df.query('columnType == "clob" ')['columnName']
        clob_col = clob_col.apply(lambda x:x.upper())
            # 2
            # col_info = column_info_df[['columnName','columnType']].dropna()
            # dtypedict = dict(zip(col_info['columnName'].apply(lambda x:x.lower()).values,
            #                          col_info['columnType'].apply(lambda x:eval(x.upper())).values)
            #                          )
            # 3
        example_info_df.columns = [i.upper() for i in example_info_df.columns]
        example_info_df = example_info_df.loc[:,~example_info_df.columns.isin(clob_col)]
        example_info_df.loc[:,example_info_df.columns.str.contains('DATE')] = example_info_df.loc[:,example_info_df.columns.str.contains('DATE')].apply(lambda x:pd.to_datetime(x),axis=0)
        try:
            save_db_data(example_info_df, conn, table_name, schema='IDB_GIL', chunksize=2000)
        except Exception as e:
            print(f'{table_name},这个表没整进去,后面再整...')
            with open(r'C:\Users\kaiyu\Desktop\miller\work_need\db_tasks\error_logs.txt','a') as f:
                f.write('-'*70+"start"'-'*70)
                f.write(str(e)+'\t')
                f.write('-'*50+'end'+'-'*70)
        # a = pd.read_sql(f'select * from IDB_GIL.{table_name}',conn)

    elif method == 'h5':
        h5_client = h5_helper(h5_group_path_rela+'_'+'table'+'.h5')

        if not table_info_df.empty:
            table_info_df = table_info_df.astype('str')
            h5_client.append_table(table_info_df,'table_info_df')

        if not column_info_df.empty:
            h5_client.append_table(column_info_df,'column_info_df')

        if not slave_column_info_df.empty:
            h5_client.append_table(slave_column_info_df,'slave_column_info_df')

        if not table_index_unique_df.empty:
            table_index_unique_df = table_index_unique_df.astype('str')
            h5_client.append_table(table_index_unique_df,'table_index_unique_df')

        if not table_change_desc_df.empty:
            h5_client.append_table(table_change_desc_df,'table_change_desc_df')

        if not table_modify_date_df.empty:
            h5_client.append_table(table_modify_date_df,'table_modify_date_df')

        if not example_info_df.empty:
            h5_client.append_table(example_info_df,'example_info_df')

        # ---------------------------------------------------------------------------------- #
        def update_qa_info(table_id,table_name,qa_table_name,h5_group_path_rela,Q_A_info_df):
            str_q = '问题:'+ Q_A_info_df.qaQuestion + '\n'
            qa_table_name = '该问题涉及的表单:' + qa_table_name +'\n'
            str_a = '回答:' + Q_A_info_df.qaAnswer +'\n'
            str_time = '最后修改的时间:'+ Q_A_info_df.lastModifiedDate
            str_qa_info = str_q+qa_table_name+str_a+str_time + ' \n \n \n'

            # # # 写入部分 # # #
            with open(f'''{h5_group_path_rela}_QA.txt''','a',encoding='utf-8') as f:
                f.write(str_qa_info)
            return str_qa_info

        if Q_A_info_df.__len__() != 0:
            # Q_A_info_df = Q_A_info_df.loc[:,~Q_A_info_df.columns.str.contains('tables')]
            for no_ in range(Q_A_info_df.__len__()):
                em_str = ''
                qa_table_name = [em_str + i['tableName'] + '' for i in Q_A_info_df.iloc[0]['tables']].__str__()[1:-1]
                update_qa_info(table_id,table_name,qa_table_name,h5_group_path_rela,Q_A_info_df.iloc[no_])


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
            table_id = tree_nodes['id']
            # obj_id = tree_nodes['OBJ_ID']
            table_name = tree_nodes['tableName']
            table_cn_name = tree_nodes['groupName']
            if table_cn_name is None:
                pass
            h5_path = f'''{res_path}\\{table_name}_{table_cn_name}'''
            print(h5_path)
            sleep_time = time.sleep(np.random.randint(1,3))
            if not os.path.exists(h5_path+'_'+'table'+'.h5'):
        # sleep_time = time.sleep(np.random.randint(0,1))
                try:
                    catch_data_main(tree_nodes,req_headers,base_url,h5_path)
                except:
                    print(f'{table_cn_name}_{table_id}_{table_name},数据有问题...')
            return res_path

    last_path = '\\'+res_path.split('\\')[-1:][0]
    res_path = res_path.replace(last_path,'')
    return res_path





# -------------------------------------------------------------------------------------- #

base_url = 'https://dd.gildata.com/'

cookie = '''rememberMeFlag=true; sSerial=TVRBNllXNWlZVzVuYzJwck1tZHBiR1JoZEdGQU1USXo=; SESSION=700741e8-1fd3-4887-bc8a-1d0410aaa8eb'''

# Override the default request headers:
req_headers = {
                'USER_AGENT':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en',
                'Cookie':cookie,
                }

# 全分支的节点树
all_tree_url = f'/api/productGroupTreeWithTables/8/802/ALL_TREE'

# 获取主节点的信息
res_all_tree = requests.get(base_url+all_tree_url,headers=req_headers)
res_all_tree = json.loads(res_all_tree.text) 

conn = init_engine()
# 声明空列表方便存储
all_df_lst = []
# 获取所有data_set的信息
all_dataset_name = res_all_tree[0]
# 数据集的path
all_data_set_path = 'Juyuan_datafile'
# os.remove(all_data_set_path)
if not os.path.exists(all_data_set_path):
        os.mkdir(all_data_set_path)



for i in range (len(all_dataset_name['nodes']) ):
         initial_file_path(all_dataset_name['nodes'][i])

# 使用xlsx的时候可以用下面的代码 #
# file_path = 'target_sheet_.xlsx'6
# f = all_df_lst
# result = pd.concat(f, axis=0,ignore_index=True)  # 将两个文件concat，也就是合并
# if not os.path.exists(file_path):
#     result.to_excel(file_path, index=False,encoding ='gbk',)








# -------------------------------------- 分割线一定要长 ------------------------------------------- #

