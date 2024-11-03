from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext

import os
import numpy as np

os.environ["CC"] = "gcc-8.1.0"

ext_mod = cythonize(["calc_credit_spread.pyx"],
                    language='c++',
                    language_level = '3'
                    )

setup(
    name = 'calc_credit_spread',
    ext_modules=ext_mod ,
    cmdclass= {'build_ext':build_ext},
    # include_dirs=[np.get_include()]
    )

# python setup.py build_ext --inplace