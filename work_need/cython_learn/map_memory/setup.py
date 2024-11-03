from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

import os,sys
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = False

os.environ["CC"] = "gcc-8.1.0"

# check_open_mp
def on_openmp():
    if sys.platform == 'win32':
        return '/openmp'
    elif sys.platform == 'linux':
        return "-fopenmp"

def set_optimize_option(arg:str) -> str:
    if sys.platform == 'win32':
        return f'/O{arg}'
    elif sys.platform == 'linux':
        return f'-O{arg}'

# ext_modules
ext_mod = cythonize(
    Extension(
    'test_multithread',
    ['cython_map_openmp.pyx'],
    language='c++',
    extra_compile_args=[
        set_optimize_option(2), # 加这个参数是为什么?
        on_openmp()
    ], #
    extra_link_args=[on_openmp()]),
    language_level ='3'
    )

setup(
    name = 'time_lib',
    ext_modules=ext_mod ,
    cmdclass= {'build_ext':build_ext},
    # include_dirs=[np.get_include(),scipy.get_include()]
    )
# python setup.py build_ext --inplace
