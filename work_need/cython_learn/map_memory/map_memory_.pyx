%%cython -cpp
#cython: language=c++
#cython: language_level=3


'''
测试数组分配性能的DEMO
'''

from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
from cython cimport view
from cpython.array cimport array,clone
from libcpp.vector cimport vector
from libcpp.map cimport map
import time

np.import_array()


cdef int loops=10000
cdef int BLOCK_SIZE=sizeof(double)
cdef list counts=[1,10,100,1000,10000,100000,1000000]

ctypedef void (*FuncPtr)(int)


cdef inline double avg(vector[double] arr):
    cdef int k
    cdef int s
    cdef double rs
    for k in range(arr.size()):
        rs+=arr[k]
    #for
    s=arr.size()
    return rs/s
#for


cdef void perform_test(name:str,FuncPtr func):
    cdef int cnt,i
    cdef int memSize
    cdef vector[double] ov

    print("执行函数",name)
    for cnt in counts:
        for i in range(loops):
            s=time.time_ns()
            func(cnt)
            e=time.time_ns()
            ov.push_back(e-s)
        #for
        memSize=cnt*sizeof(double)
        print(name+"执行"+str(i+1)+"次的"+"分配内存空间"
              +str(memSize)+"平均耗时"+str(avg(ov))+"ns")
    #for
#def


cdef inline void memview_malloc(int N):
    cdef double * m = <double *>malloc(N * sizeof(int))
    cdef double[::1] b = <double[:N]>m
    free(<void *>m)
#def

cdef inline void memview_ndarray(int N):
    cdef double[::1] b = np.empty(N, dtype=np.double)
#def

cdef inline void memview_np_arange(int N):
    cdef double[::1] b = np.arange(N, dtype=np.double)
#def

cdef inline void memview_cyarray(int N):
    cdef double[::1] b = view.array(shape=(N,), itemsize=sizeof(double), format="d")
#def

cdef inline void memview_cpython_array(int N):
    cdef int i
    cdef double[::1] arr
    cdef array template = array('d')
    arr = clone(template, N, False)
#def

cdef inline void memview_array_buffer(int N):
    cdef int i
    cdef double[:] arr
    cdef array tmp = array('d')
    arr = clone(tmp, N, False)
    # Prevents dead code elimination
    arr[:]=0.0
#def

def test_all():
    perform_test("memview_cpython_array",memview_cpython_array)

    perform_test("memview_malloc",memview_malloc)

    perform_test("memview_ndarray",memview_ndarray)

    perform_test("memview_array_buffer",memview_array_buffer)

    perform_test("memview_cyarray",memview_cyarray)

    perform_test("memview_np_arange",memview_np_arange)

test_all() #

# 这个方案是最快的[单纯指生成]
cpdef inline void memview_cpython_array(int N):
    cdef int i
    cdef double[::1] arr
    cdef array template = array('d')
    arr = clone(template, N, False)
    return arr
#def

# Malloc版本的C 不太熟，玩不转这玩意