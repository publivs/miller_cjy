# distutils: language=c++
#cython:language_level=3

from currency cimport MoneyFormator
from libcpp.string cimport string

cpdef string money_format(str localName,double n):
    '''重堆中为MoneyFormator类分配内存'''
    cdef MoneyFormator* mon

    try:
        if localName=='' or localName==None:
            mon=new MoneyFormator()
        else:
            mon=new MoneyFormator(localName[0].encode('utf-8'))
        return mon.str(n)
    except Exception as e:
        print(e)
    finally:
        del mon