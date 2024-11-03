#cython:language_level=3
#
cdef extern from "currency.cpp":
    pass

from libcpp.string cimport string

cdef extern from "currency.hh" namespace "ynutil":
    cdef cppclass MoneyFormator:

        MoneyFormator() except +
        MoneyFormator(const char*) except+

        string str(double)