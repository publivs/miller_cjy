import pandas as pd 
import numpy as np 
from scipy import stats
import sympy as sy



from scipy.stats import binom

from scipy.stats import norm

def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

array_list = [
    -2.456,
    -3.388,
    -6.816,
    1.531,
    1.737,
    -1.254,
    -1.164,
    1.532,
    2.550,
    0.296,
    -0.979,
    -4.259,
    2.810,
    -1.608,
    -0.575
]

array_list = np.array(array_list)

answer_2 = standardization(array_list)

skew_3 = stats.skew(array_list)

B=pd.Series(array_list)

A= pd.Series(answer_2)
A = A.apply(lambda x:x**4)



def cdf_normalized(value,mean,var):
    return (value - mean)/np.sqrt(var)

#
answer_39_a = norm(0,1).cdf(0) - norm(0,1).cdf(-1.5)
answer_39_b = norm(0,1).cdf(-1.5)
answer_39_c = norm(0,1).cdf(-1/np.sqrt(2))-norm(0,1).cdf((-1.5-1)/np.sqrt(2))
answer_39_d =  1 - norm(0,1).cdf(cdf_normalized(2,1,2))
answer_39_e = 1 - norm(0,1).cdf(cdf_normalized(12,3,9))

# 
answer_310_a = norm(0,1).interval(0.95)
answer_310_c  = norm(0,1).interval(0.75)

# 
answer_310_d_1 = norm(0,1).interval(0.125)
a = answer_310_d_1[0]*2+2



rev_ = 0.08/252
var_ = (0.2)**2/252
answer_310_d_1 = norm(0,1).interval(0.001)
answer_310_d_1  =  answer_310_d_1[0] * np.sqrt(var_)+rev_

#
answer_312_a = norm(0,1).cdf(cdf_normalized(-10,20,300))
answer_312_a =  norm.interval(0.999)[0]*np.sqrt(300)+20

# 
answer_313 = 5,6

from scipy.stats import expon

y = sy.symbols('y')
b = sy.symbols('b')
func = (1/b)*sy.exp(-y/b)
answer_314_1 = sy.integrate(func,(y,0,5))

large_firm = [-50,0,10,100]
small_firm = [-1,0,2,4]
value_martix = np.matrix([
                        [1.97,3.9,0.8,0.1],
                        [3.93,23.5,12.6,2.99],
                        [0.8,12.7,14.2,6.68],
                        [0,3.09,6.58,6.16]
                        ])
A= pd.DataFrame(value_martix,index =large_firm,columns= small_firm)/100

# E(X1X2)
def calc_expect(df):
    if isinstance(df,pd.DataFrame):
        res_lst = []
        for col in df.columns:
            se_calc = pd.Series.multiply( df[col] , np.array(df.index),axis=0) * col
            res_lst.append(se_calc)
        EE_df = pd.concat(res_lst,axis=1)
    if isinstance(df,pd.Series):
        EE_df = pd.Series.multiply( df , np.array(df.index),axis=0)
    EE  = EE_df.values.flatten().sum()
    return EE

EE=  calc_expect(A)
#EX1
EX1 = (A.sum(axis=1) * [tt for tt in A.index]).sum()

#EX2
EX2 = (A.sum(axis=0) * [tt for tt in A.columns]).sum()

cov = EE - EX1*EX2


EX1_2 = (A.sum(axis=1) * [tt**2 for tt in A.index]).sum()

EX2_2 = (A.sum(axis=0) * [tt**2 for tt in A.columns]).sum()

std_1 =  np.sqrt(EX1_2 - EX1**2)

std_2 =  np.sqrt(EX2_2 - EX2**2)
corr = cov/(std_1*std_2) 

# 4.13
minus = 0.2
EX_13 = EX1*minus + EX2*(1-minus)
EX_13 = minus**2  *(std_1)**2 + (minus **2)*(std_2**2) + minus*2 * (minus *2)*cov

# 4.14
B = A[[-1,0]]
B_condition_proba = B / B.sum().sum()
B_sum_ = B_condition_proba.sum(axis=1)
EX_B = calc_expect(B_sum_)

#415
sp_500_lst = np.array([-10,0,10])
sp_500_proba = np.array([25,50,25])/100
nikkei_lst = np.array([-5,0,5])
nikkei_proba  = np.array([20,60,20])/100
joint_probability  = pd.DataFrame(np.dot(sp_500_proba.reshape(3,1),
                                    nikkei_proba.reshape(1,3)),
                                    index = nikkei_lst,columns=sp_500_lst)



# 第五章计算题
answer_515 = [
0.0,0.07,0.13,0.13,0.2,0.23,0.25,0.27,0.34,0.41,0.6,0.66,0.76,0.77,0.96]
s_515 =pd.Series(answer_515)
s_515.index = [s_515.index+1]
unbias_var  = s_515.var() * (s_515.__len__() /(s_515.__len__() -1) )
b =  2*s_515.mean()

# 用方差估参数
b_bias = np.sqrt(0.086*12)
b_unbias = np.sqrt(0.092*12)

answer_516= [0.38,0.28,0.27,0.99,0.26,0.43]


#
answer_611 = norm.interval(0.9)
answer_612 = norm.interval(0.8)
answer_613 = norm.interval(0.99)
answer_614 = norm.interval(0.9995)

#
answer_613_a = 2*(1- norm.cdf(1.45))

answer_613_b = 1*(1- norm.cdf(1.45))

answer_613_c = norm.cdf(1.45)

answer_613_d = 1-norm.cdf(-2.3)

answer_613_e = 2*(1- norm.cdf(2.7))



a = norm.interval(0.9)

# quantitive 第七章课后习题
'''
7.1
1、有一定线性上的表达关系
2、误差是可加的
3、所有解释变量都是可以被观察的
'''

# 计算题
from scipy.stats import linregress
x = np.arange(0,10)
y = [-1.46,0.35,6.46,4.09,7.34,6.18,14.97,14.28,20.20,21.24]

slope, intercept, r_value, p_value, std_err = linregress(x, y)
std_error_variance  = ((y - (intercept +slope*x ))**2)/(y.__len__()-2)

#
asw_7_13 = np.sqrt(20.30*(0.71**2+19.82)/(20*12*19.82))
asw_7_13_b_se = np.sqrt(20.3/19.82/240)
#
from scipy.stats import norm
fi  = norm.interval(0.99)[1]
confi_interval = 1.37 + fi*asw_7_13_b_se

fi = norm.interval(0.9)[1]

#

y=[-5.76,0.03,-0.25,-2.72,-3.08,-7.1,-4.1,0.14,-6.13,0.74]
x_1=[-3.48,-0.02,-0.5,-0.18,-0.82,-2.08,-1.06,0.02,-1.66,0.68]
x_2=[-1.37,-0.62,-1.07,-1.01,0.39,1.39,0.75,-0.63,1.31,-1.15]

from scipy.stats import linregress
df = pd.DataFrame([x_1,x_2,y]).T
# /y = np.array(y).reshape(10,1)
# res = linregress(df.iloc[:,:2].T.values,df.iloc[:,2].T.values)

import statsmodels.api as sm
mod=sm.OLS(y,sm.add_constant(x_1,x_2)) # 需要用sm.add_constant 手动添加截距项
res=mod.fit()
res.summary()

y =[
    -2.353,
    -0.114,
    -1.665,
    -0.364,
    -0.081,
    -0.735,
    -2.507,
    -1.144,
    -2.419,
    -3.151,
    -2.085,
    -2.972,
    -0.633,
    -2.678,
    -7.095
]

x1 =[
    -0.409,
    0.397,
    -0.856,
    1.682,
    0.455,
    -1.39,
    0.954,
    1.021,
    -0.156,
    1.382,
    -0.562,
    -1.554,
    -1.123,
    -0.124,
    0.284
]

x2 = [-0.008,-1.216,-0.911,0.366,-0.639,-1.086,0.67,0.238,-0.055,1.148,-0.135,-0.299,-1.027,0.331,2.622]
mod=sm.OLS(y,sm.add_constant(x1))
res=mod.fit()
res.summary()

mod=sm.OLS(y,sm.add_constant(x2))
res=mod.fit()
res.summary()

mat = pd.DataFrame([x1,x2],index=['a','b']).T

COV_arr = (mat['a']-mat['a'].mean())*(mat['b']-mat['b'].mean())
COV = COV_arr.mean()
x1 =  np.array(x1)
x2 =  np.array(x2)
v1 = x1.var()
v2 = x2.var()
d1 = COV/v1
d2 = COV/v2

# aic_bic
def calc_aic(T,k,sigma_squared):
    res = T*np.log(sigma_squared) + 2*k
    return res

def calc_bic(T,k,sigma_squared):
    res = T*np.log(sigma_squared) + k*np.log(T)
    return res