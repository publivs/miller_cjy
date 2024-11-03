import numpy as np
cimport numpy as np

from scipy.stats import norm
from libcpp.string cimport string
from libc.math cimport sqrt as csqrt
from libc.math cimport exp as cexp
from libc.math cimport log as clog
from libc.math cimport pow as cpow
from libc.math cimport erfc

cdef norm_cdf_cy(double x,):
    return 1/2 * erfc(-x/csqrt(2))

cpdef vanilla_option_cy(
                        double S,
                        double K,
                        double T,
                        double r,
                        double sigma,
                        string option = b'call'):
    cdef :
        double d1
        double d2
        double p

    d1 = (clog(S/K) + (r + 0.5* cpow(sigma,2) )*T)/(sigma*csqrt(T))
    d2 = (clog(S/K) + (r - 0.5* cpow(sigma,2) )*T)/(sigma *csqrt(T))
    if (option ==  b'call'):
        p = (S*norm_cdf_cy(d1) - K*cexp(-r*T)*norm_cdf_cy(d2))
    elif (option == b'put'):
        p = (K*cexp(-r*T)*norm_cdf_cy(-d2) - S*norm_cdf_cy(-d1))
    else:
        return None
    return p

vanilla_option_cy(50, 100, 1, 0.05, 0.25, option=b'call')

