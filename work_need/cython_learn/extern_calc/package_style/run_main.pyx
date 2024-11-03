# distutils: language=c++
#cython:language_level=3
cdef extern from "use_func.h":
    cdef double c_func(int n)

import time

def func(int n):
    return c_func(n)

def main():
    start = time.time()
    res = func(30000000)
    print(f"res = {res}, use time {time.time() - start:.5}")

