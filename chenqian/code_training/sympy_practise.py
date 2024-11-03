
import numpy as np 
import pandas as pd

import sympy as sp

from IPython.display import display, Latex
x, y, z = sp.symbols('x y z')
sp.init_printing(use_unicode=True)

f1 = sp.cos(x)
f2 = sp.exp(x**2)
f3 = (x**3-2)/(-2*x**2)
display(Latex(f"$$d{sp.latex(f1)}={ sp.latex(sp.diff(f1,x))}$$"))
display(Latex(fr"$$d^2\left({sp.latex(f2)}\right)={ sp.latex(sp.diff(f2,x,2))}$$"))
display(Latex(fr"$$d^3\left({sp.latex(f3)}\right)={ sp.latex(sp.diff(f3,x,3))}$$"))

x = sp.spmbols('x')
expr = sp.exp(x)*sp.sin(x)+sp.exp(x)*sp.cos(x)
integ = sp.integrate(expr)
print(integ)

