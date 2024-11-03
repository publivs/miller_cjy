import os
import sys
import pandas as pd
import numpy as np
import pickle
import tables as ts
import time


'''
详情请参考这个
pd的源码:
    https://pandas.pydata.org/pandas-docs/stable/reference/io.html#hdfstore-pyts-hdf5
知乎高质量回答:
    https://zhuanlan.zhihu.com/p/352539790
'''

class h5_helper():
    def __init__(self,store_path,is_print = '0'):
        if os.path.exists(store_path):
            print('目标路径存在文件,可以启动HDFS......')
        else:
            # 写入模式创新的数据集
            f = ts.open_file(store_path, mode="w")
            # 关闭指针
            f.close()
            print(f'目标路径不存在文件,已创建新的HDFS:{store_path}...')

        self.store_path = store_path
        self.print_time = is_print
    def get_table(self,table_name):
        '''
        读全表的时候用这个方法,拿数据的都是get_node
        '''
        store_client = pd.HDFStore(self.store_path,mode='r+')
        # 这里还可以再包一些其他参数,后续会补上
        t0 = time.time()
        df = store_client.get(table_name)
        t1 = time.time()
        if self.print_time == '1':
            print(f'数据读取完毕...,耗时:{t1-t0}')

        store_client.close()
        return df

    def select_table(self,table_name,where_str = None):
        '''
        Instr:获取table中的数据
        Args:
            table_name:store中node的名字
            where_str:有时您的查询操作可能会涉及要创建一个行列表来进行选择。通常可以使用索引操作返回结果的索引来进行选择。
                        下面是有效的表达式:
        tips:
            where_str的写法举例:
                        'index >= date'
                        "columns = ['A', 'D']"
                        "columns in ['A', 'D']"
                        'columns = A'
                        'columns == A'
                        "~(columns = ['A', 'B'])"
                        'index > df.index[3] & string = "bar"'
                        '(index > df.index[3] & index <= df.index[6]) | string = "bar"'
                        "ts >= Timestamp('2012-02-01')"
                        "major_axis>=20130101"
                        "c[pd.DatetimeIndex(c).month == 5].index"

        [有稍微高级一点的写法,可以参考read_single_col...]
        '''
        # 这里还可以再包一些其他参数,读取尽量只用只读
        store_client = pd.HDFStore(self.store_path,mode='r+')
        df = store_client.select(table_name,where=where_str)
        store_client.close()
        return df

    def append_table(self,df,table_name,complevel = 9):
        '''
        根据以前的数据向里面追加额外数据,插入数据类型要跟原HDF对齐就有点难受
        tips:
            1、data_columns是必须填的参数而且必须是py的原生对象,一般不支持其他类(如pandas中的一些特殊类)
            2、complevel会影响写入和读取速度,个人仍然推荐高压缩比[7,8,9];
            3、complib: 选择blosc:lz4读写速度都很快(也试了下其他格式,zlib啥的拉夸的一批)
        '''
        if not df.empty:

            if self.print_time == '1':
                print('数据写入中...')
            t0 = time.time()

            # store_client = pd.HDFStore(self.store_path,mode='a')

            # 这里可能还需要检查,因为新增部分的列名可能跟之前的不同,得先检查表的属性对上列名让后再追加数据

            df.to_hdf(self.store_path,table_name,
                    format='table',
                    mode='a',
                    data_columns = df.columns.to_list(),
                    append=True,
                    index=False,  # 索引最好写完之后自己统一加,针对全DF加索引没有任何意义还浪费速度浪费空间
                    complevel=complevel, complib="blosc:lz4")
            t1 = time.time()

            # store_client.close()
            if self.print_time == '1':
                print(f'数据写入完毕...,耗时:{t1-t0}')
        else:
            print('数据为空,请检查数据......')

    def set_table_index(self,table_name,col_name =[]):
        '''
        Instru:
            table_name: store下对应table的名字
            col_name: 指定索引列,
        tips:
            索引的设置:
                1、先更新数据,再考虑索引。
                2、尽量少设置设置,索引多了等于没索引。
                3、贴近业务需求,按照区分度的优先原则进行设置。
        '''
        store_client = pd.HDFStore(self.store_path)

        try:
            # 单索引
            if isinstance(col_name,str):
                store_client.create_table_index(table_name, columns=[col_name], optlevel=9, kind="full")

            # 多索引
            if isinstance(col_name,str):
                store_client.create_table_index(table_name, columns=col_name, optlevel=9, kind="full")

        except Exception as e:
            store_client.close()
            raise(e)

        store_client.close()

    def get_single_col(self,table_name,col_name):
        '''
        Instru:复刻原生的read_columns,不支持布尔运算
        tips:
            这个功能存在的意义:
                1、h5的table文件,提前设置好了索引
                2、你先选好了索引列,并且获取了pandas的DF对象
                3、获取了指定列(索引)的对象之后呢,对这个列进行pandas逻辑运算,筛选出数据
        E.g:
            c = store.get_single_col(table_name, "index")
            where = c[pd.DatetimeIndex(c).month == 5].index
            df = h5_client.select("df_mask", where_str = where)
        '''
        store_client = pd.HDFStore(self.store_path)
        df = store_client.select_column(table_name,col_name)
        store_client.close()
        return df

    def remove_table(self,table_name):
        '''
        删除node
        '''
        store_client = pd.HDFStore(self.store_path,mode='w')
        store_client.remove(table_name)
        store_client.close()

def degrade_incuracy(df,degrage_level = 'low'):
    '''
    降低精度,方便读取和存储
    1、把objective全部切换成字符串
    2、float类型转为float16
    3、所有的obj全部强转为STR
    4、所有datetime和时间序列类型转成datetime64
    '''
    def quick_numeric(se,degrage_level):
        dtype_str = se.dtype.__str__()

        # 调整float
        if 'float' in dtype_str:
            # se = pd.to_numeric(se,downcast='float')
            if degrage_level  == 'low':
                se = se.astype('float16')
            else:
                se = se.astype('float32')
        # 调整int
        if 'int' in dtype_str:
            if degrage_level  == 'low':
                se = pd.to_numeric(se,downcast='integer')
            else:
                se = se.astype('int32')
        if 'datetime' in dtype_str:
            se = se.astype('datetime64[ns]')
        return se

    # 针对OBJ切换成string
    obj_columns = df.loc[:,df.dtypes == 'O'].columns
    df[obj_columns] = df[obj_columns].astype('str')

    # 调节精度优化读写
    df.loc[:,~df.columns.isin(obj_columns)] = df.loc[:,~df.columns.isin(obj_columns)].apply(lambda se:quick_numeric(se,degrage_level))
    return df

