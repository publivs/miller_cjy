
from sqlalchemy import create_engine
from sqlalchemy import types
import os

import sys,os
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # 两层路径之外
sys.path.append(base_path)
from chenqian_tools.hdf_helper import *

def init_engine():
    engine = create_engine(
    '''oracle+cx_oracle://{username}:{password}@{host}:{port}/?service_name={service}'''.format(
        username="dm_wind",
        password="78eeb83d8ef690dc",
        host="192.168.0.18",
        port=1521,
        service="ORCL",
        encoding='utf-8',
        # pool_size=4,
        # max_overflow=15,
        # pool_timeout=30,
        # echo=True,
        # pool_pre_ping=True
    )
    )
    return engine

def save_db_data(df, db, table_name, schema, chunksize=10000):

    dtyp = {}
    for c in df.columns[df.dtypes == 'object'].tolist():
        if df[c].str.len().max() <= 0:
            value = types.VARCHAR(df[c].str.len().max())
        else:
             value = types.VARCHAR(8)
        dtyp[c] = value

    with db.engine.begin() as conn:
        df.to_sql(table_name,
                  conn,
                  if_exists="append",
                  chunksize=chunksize,
                  index=False,
                  schema=schema,
                  dtype = dtyp
                  )
    print("%s落库成功"%table_name)


doc_path = r'D:\wind_data'
file_paths  = os.listdir(doc_path)
error_table_list = [
    # 'AShareCashFlow',
                    'AshareIncome',
                    'ChangeWindCode',
                    'CHINAMUTUALFUNDASSETPORTFOLIO',
                    'CHINAMUTUALFUNDDESCRIPTION',
                    'CHINAMUTUALFUNDBONDPORTFOLIO',
                    'globalworkingDay'
                    ]
error_table_list = [i.upper()+'.H5' for i in error_table_list]

for file_path in file_paths:
    if file_path.upper() in error_table_list:
        try:
            print(f"开始处理{file_path}...")
            # file_path = "IndexContrastSector.h5"
            h5_client = h5_helper(doc_path+'\\'+file_path)
            df = h5_client.select_table('data')
            conn = init_engine()
            save_db_data(df,conn,f'''{file_path[:-3].upper()}''','''DM_WIND''')
        except Exception as e:
            print(f'{file_path},这个h5文件没整进去,后面再整...')
            with open(r'C:\Users\kaiyu\Desktop\miller\work_need\db_tasks\error_logs.txt','a') as f:
                f.write('-'*50)
                f.write(str(e)+'\t')
                f.write('-'*50)





