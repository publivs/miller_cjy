import pandas as pd
import numpy as np

import time

class CreditSpreadCalculation:
    def __init__(self) -> None:
        self.df_dict = {}

    def check_options(self):
        '''

        '''
        pass

    def process_valution(self):
        '''

        '''
        pass
    def calc_valuation_df(self,bond_valuation_lst,remaining_term_lst,NDB_valuation_dict):
        """
        计算债券估值差异。
        :param bond_valuation_lst: 债券估值列表。
        :param remaining_term_lst: 剩余期限列表。
        :param ndb_valuation_dict: 国开债估值字典。
        :return: 债券估值差异列表。
        """
        def ndb_adj_valuation(num,lst,map_dict):
            """
            在有序列表list中找到num所在的区间
            """
            n = len(lst)
            if num < lst[0] or num > lst[-1]:
                return (np.nan)
            left, right = 0, n-2
            while left <= right:
                mid = (left + right) // 2
                if lst[mid] <= num < lst[mid+1]:
                    left,right = lst[mid], lst[mid+1]
                    break
                elif lst[mid+1] <= num:
                    left = mid + 1
                else:
                    right = mid - 1

            res = ((num - left)/(right - left)) *(map_dict.get(right) - map_dict.get(left)) + map_dict.get(left)
            return res

        NDB_valuation_lst = list(NDB_valuation_dict.keys())
        target_list = [bond_valuation_lst[i] - ndb_adj_valuation(remaining_term_lst[i], NDB_valuation_lst, NDB_valuation_dict)
                   for i in range(len(remaining_term_lst))]

        return target_list
        # 国开债这里用插值计算,国开债到期收益率用线性差值计算：(T-T1)/(T2-T1)*(Y2-Y1)+Y1）
        # target_list = []
        # NDB_remain_list = list(NDB_valuation_dict.keys())
        # for i in range(remaining_term_lst.len()):
        #     res = bond_valuation_lst[i] - ndb_adj_valuation(remaining_term_lst[i],NDB_remain_list,NDB_valuation_dict)
        #     target_list.append(res)
    def get_moni_data(self):

        mat = pd.DataFrame()
        mat['NDB_valuation'] = np.arange(0,10)
        mat['NDB_valuation'] = mat['NDB_valuation'] + np.random.random(10)
        # mat = mat.T
        # 单债券的债券信息
        single_ytm = pd.DataFrame(columns=['bond_code','bond_valuation','remaining_term'])

        single_ytm['bond_code'] = ['a','b','c','d','e']
        single_ytm['bond_valuation'] = [101,102,103,104,105]
        single_ytm['remaining_term'] = np.arange(5,10) + np.random.random(5)

        # self.df_dict['test_df'] = 'a'
        a = time.time()
        single_ytm['valuation_diff'] = self.calc_valuation_df(
                                                            single_ytm['bond_valuation'].values,
                                                            single_ytm['remaining_term'].values,
                                                            dict(mat['NDB_valuation']))
        print(time.time() -a )
        print(1)

    def calc_city_investment_bond(self):
        pass

# if __name__ == "main":
a = CreditSpreadCalculation()

b = a.get_moni_data()