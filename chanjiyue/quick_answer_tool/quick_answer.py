import pandas as pd
import numpy as np
import jieba
import re

sheet_list = ['多选','单选','判断']
file_path = [r'C:\Users\Administrator\Desktop\chanjiyue\quick_answer_tool\data\bank_law.xlsx']
question_col = '题目：'

rename_dict = {'正确':'right','错误':'wrong'}

res_df_lst = []
for df_path in file_path:
    for sheet_name  in sheet_list:
        df = pd.read_excel(df_path,sheet_name=sheet_name)
        if sheet_name == '判断':
            df['正确答案:']=  df['正确答案:'].map(rename_dict)
        res_df_lst.append(df)
res_df = pd.concat(res_df_lst,ignore_index=True)


for col in res_df.columns:
    res_df[col] = res_df[col].fillna('_').apply(lambda x:str(x))

combination = pd.Series(None,index = res_df.index).fillna('_')
for col in res_df.columns:
    a = res_df[col]
    combination = a +combination



def quick_reverse(x):
    cut_for_search_1 = [i for i in jieba.cut_for_search(x)]
    return cut_for_search_1

combination_cut_for_search = combination.apply(lambda x:quick_reverse(x))
'''
按照关键字对题目进行搜索
'''



def process_input(match_sentence):
    match_lst = pd.Series([i for i in jieba.cut_for_search(match_sentence)])
    return match_lst

input_str = 'start'
while input_str != 'stop':
    input_str = input('请输入查找内容:')
    
    match_lst = process_input(input_str)
    match_lent  = match_lst.__len__()
    for i in range(len(combination_cut_for_search)):
            index  = i
            word_cut_for_search = combination_cut_for_search.iloc[i]
            if match_lst.isin(word_cut_for_search).all():
                print(res_df.iloc[index])






