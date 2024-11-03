from cpython.array cimport array,clone
from libc.string cimport memset
from libcpp.vector cimport vector

'''
Cpython的Array

'''


def test_1():

    cdef array tmp = array('i',[1,2,3,4,5])
    cdef int[:] arr = tmp

    cdef k,j
    cdef int L = arr.shape[0]

    for i in range(L):
        j = tmp.data.as_ints[k] # 针对一维数组，直接用这个方法实现C级别的访问


def test_2():
    cdef vector[int] vc =[1,2,3,4,5,6,7]

    cdef int[:] mv = <int [:vc.size()]>vc.data()

    cdef int k
    for k in range(mv.shape[0]):
        print(mv[k])



