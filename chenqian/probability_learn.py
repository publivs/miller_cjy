
import pandas as pd
import numpy as np
import scipy as sp


from scipy.stats import binom

# # 两个的投篮结果相互独立
# 不需要考虑条件概率 直接求交即可

# 8.1
k_arr = np.arange(0,4)

p1 = 0.6
p2 = 0.7
total_shot = 3

total_value = 0
for  i in k_arr:
    count_value = binom.pmf(i,total_shot,p1).sum() * binom.pmf(i,total_shot,p2).sum()
    total_value = total_value+ count_value
print(total_value)

# 8.2
total_value = 0
for  i in k_arr:
    if i >= 1:
        count_value = binom.pmf(i,total_shot,p1).sum() * binom.pmf(i-1,total_shot,p2).sum()
        total_value = total_value+ count_value
print(total_value)

# 9.0
p = 0.1
n = 10
k1 = np.arange(1)
value_answer_1 =  binom.pmf(k1,n,p).sum()

# 9.2
k2 = np.arange(1,3)
value_answer_2 = binom.pmf(k2,n,p).sum()

# 9.3
values_9_3 = binom.pmf(0,5,p).sum()

# 9.4
value_9_4 = value_answer_2 * binom.pmf(0,5,p).sum()

# 9.5
k5_1_negative  = np.arange(3,n)
k5_1_positive = np.arange(1,3)
k5_2 = np.arange(1,5)
value_9_5  = binom.pmf(k5_1_negative,n,p).sum() + binom.pmf(k5_1_positive,n,p).sum() * binom.pmf(k5_2,5,p).sum()

# 实验成功
n = 8
p10  = 1/2
k_10 = np.arange(0,5)
value_10_1 = binom.pmf(4,n,p10)






