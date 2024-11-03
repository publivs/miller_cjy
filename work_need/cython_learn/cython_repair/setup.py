from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import os
os.environ["CXX"] = "g++"

ext_modules = [
    Extension(
        "hello",
        sources=["hello.pyx"],
        language='c++',
    )
]
setup(
    name='Hello world app',
    ext_modules=cythonize(ext_modules),
)

# python setup.py build_ext --inplace

'''
第二种编译方法
'''
