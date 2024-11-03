# %%cython -cpp
cimport cython
cimport openmp
from libc.math cimport log
from cython.parallel cimport prange
from cython.parallel cimport parallel
from libcpp.vector cimport vector

import time

ctypedef void FuncPrt(double[:],double[:],double[:])

cdef void serial_loop(double[:] A ,double[:] B ,double[:] C):
    cdef int N = A.shape[0]
    cdef int i
    for i in range(N):
        C[i] = log(A[i])*log(B[i])
    #for
#def

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void parallel_loop(double[:] A ,double[:] B,double[:] C):
    cdef int N = A.shape[0]
    cdef int i

    with nogil:
        for i in prange(N,schedule ='static',num_threads = 12):
            C[i] = log(A[i]+log(B[i]))
        #for
    #with
#def

cdef double avg(vector[double] arr):
    cdef double s = 0.0
    cdef int length = arr.size()
    cdef int i

    for i in range(length):
        s += arr[i]
    return s/length

cdef void perform_test(name:str,FuncPrt func,double[:] a ,double[:] b ,double[:] c ,):
    cdef int i
    cdef vector[double] ov

    for i in range(10):
        s = time.time_ns()
        func(a,b,c)
        e = time.time_ns()
        ov.push_back(e-s)
        print(f"执行函数{name}第{i}次耗时{avg(ov)}ns")

    print("执行函数",name,"平均耗时",avg(ov),"ns")


def test_serial(double[:] x1,double[:] x2,double[:] y):
    perform_test("serial_loop",serial_loop,x1,x2,y)

def test_parallel(double[:] x1,double[:] x2,double[:] y):
    perform_test("parallel_loop",parallel_loop,x1,x2,y)



