import numpy as np
cimport numpy as np
import pandas as pd
from scipy import interpolate
from libc.math cimport sqrt as csqrt
from libc.math cimport pow as cpow
from libc.stdio cimport  sscanf
from libc.time cimport tm,mktime,time_t,strptime,strftime,localtime
from libc.stdio cimport sizeof
from scipy import optimize as so 
DTYPE = np.intc

cpdef calc_cashflow_cy(double bar,
                        double couponrate,
                        double start_date,
                        double next_coupon_rate,
                        double enddate,
                        freq = 1):
    cdef:
        list cashflow = list()
        list time_list = list()
        double date_temp
        double timedelata_365_s

    date_temp = next_coupon_rate
    timedelata_365_s = 365*24*60*60
    while enddate >= date_temp:
        cashflow.append(bar * couponrate)
        time_list.append((date_temp-start_date)/timedelata_365_s)
        date_temp =(date_temp +timedelata_365_s)
    cashflow.append(bar)
    return cashflow,time_list

from scipy import interpolate
cpdef calc_rate_list_cy(double duration,
                        list rate_list,
                        list time_list):
    f=interpolate.interp1d(x=duration,y=rate_list,kind='slinear')
    r_list = list(f(time_list))
    return r_list

# 债券精确定价函数
%%cython
import numpy as np
cimport numpy as np
cpdef calc_precisePrice_cy(double bar,
                            double couponrate,
                            double[:] r_list,
                            double[:] time_list):
    per_coupon = bar * couponrate
    discount_coupon = 0
    cdef int arr_len = r_list.shape[0]
    for i in range(arr_len):
        r = r_list[i]
        time = time_list[i]
        if(r != r_list[-1]):
            discount_coupon = discount_coupon + per_coupon/(1 + r*0.01)**time
    return (discount_coupon + bar/(1 + r_list[-1]*0.01)**time_list[-1])

cpdef mac_duration_cy(double[:] cashflow,
                    double[:] time_list,
                    double[:] r_list,
                    double presentvalue):
    cdef :
        double mcduration = 0.0
        int arr_len = time_list.shape[0]
    for i in range(arr_len):
        cash = cashflow[i]
        time = time_list[i]
        r = r_list[i]
        mcduration = mcduration + time * ( cash / (1 + r * 0.01) ** time) / presentvalue
    return mcduration

#计算ytm --可以优化
%%cython
from scipy import optimize as so
import numpy as np
cimport numpy as np
def YTM_cy(double presentvalue,
            double[:] cashflow,
            double[:] time_list):
    def ff(y):
        cash_all = []
        for cash,time in zip(cashflow,time_list):
            if(cash != cashflow[-1]):
                cash_all.append(cash/pow(1+y,time))
        return np.sum(cash_all)+cashflow[-1]/pow(1+y,time_list[-1])-presentvalue
    return float(so.fsolve(ff,0.01))

def dduration(double mac_dura,
            double presentvalue,
            double ytm):
    a = mac_dura
    b = (1+ytm)
    c = a/b
    return c


%%cython
import numpy as np

cimport numpy as np
cpdef Convexity_cy(list cashflow,
                list time_list,
                double presentvalue,
                double ytm):

    cdef list temp = list()
    cdef int arr_len = len(cashflow)
    for i in range(arr_len):
        cash = cashflow[i]
        time = time_list[i]
        temp.append(cash*(time*time + time)/pow(1+ytm,time))
    return (1/(presentvalue*pow(1+ytm,2))) * np.sum(temp)


%%cython

%timeit calc_precisePrice_cy(bar,couponrate,r_list,time_list)

%timeit mac_duration_cy(cashflow,time_list,r_list,price2205)

%timeit ytm = YTM_cy(price2205,cashflow,time_list)

%timeit dduration(cashflow,time_list,r_list,price2205,ytm)

%timeit Convexity_cy(list cashflow,list time_list,double presentvalue,double ytm)



