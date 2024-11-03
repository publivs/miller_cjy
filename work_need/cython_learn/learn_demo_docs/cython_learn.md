#  Cython笔记本

# Cython学习笔记

## 参考资料索引:

用户指南 - 在 Cython 中使用 C ++ - 《Cython 3.0 中文文档》 - 书栈网 · BookStack](https://www.bookstack.cn/read/cython-doc-zh/docs-31.md)

[
    Cython/PyPy编程技术 - 知乎 (zhihu.com)](https://www.zhihu.com/column/c_1284193666899787776)

## Cython的基本使用

### 编译:

#### 1)Setup-tool方法 --推荐(简单，方便)

创建一个文件: setup.py

```
from distutils.core import setup

from Cython.Build import cythonize

import os
import numpy as np
import scipy

os.environ["CC"] = "gcc-8.1.0" # 这里要在CMD里面 gcc -v看一下自己对应的配置版本安装没

ext_mod = cythonize(["time_libs.pyx"], 对具体的pyx文件进行Cythonize化
                    language='c++', # 指定语言
                    language_level = '3')

setup(
      name="Example Cython", # Example_Cython:给新包起名字
      ext_modules=ext_mod,
      cmdclass= {'build_ext':build_ext},
      include_dirs=[np.get_include(),#  指定附载的包,个人建议默认把numpy载进去
      				scipy.get_include()]
      )

```

3、让后在同目录根下运行

    python setup.py build_ext --inplace

4、让后等待编译完毕之后，会在built文件目录下生成一个文件:

​	Windows: examples_cy.cp39-win_amd64.pyd，如果是Mac/Linux系统,文件后缀就是So



#### 2)Cythonize方法

在 Cmd中输入**Cythonize --h** 即可查看所有对应的附加指令。

**--include-dir**:指定编译时包含的C/C++头文件或其他*.py或*.pyx文件

**--output-file**:指定解析后的C/C++源代码文件的所在路径

**--working**:指定cython解析器的工作目录

**-2或-3**:-2告知cython解析器以python2的方式理解python源代码,-3告知cython解析器以python3的方式去理解python源代码

代码示例:

```python
	cythonize -a -i example_code.pyx -3
```

example_code是示例的pyx代码

-a	开关还会生成带注释的源代码html文件,可以查看和Cpython解释器调用多少次



#### 3) 命令行Cython方法

生成c/cpp文件

```bash
cython test_cython_code.pyx -o ./test_cython_code.c -3 --cplus --gdb
```

指定编译:

```bash
gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing C:\Programs\Python\Python37 -o ./test_cython_code.pyd  ./test_cython_code.c
```



### Cython:Jupyter直接交互使用



1、保证运行的环境安装了Cython

2、安装了Ipython已经其依赖的kernel



```python
%load_ext cython

%%cython

cpdef func_test(a,b):

    return a+b

test_value = func_test(1,2)
print(test_value)

```

可以使用后台帮忙编译加载 Cython计算函数,针对写好的Cython函数可以直接验算交互。

ipython-Cython有非常多的扩展功能



## Cython中的字符

在Cython中处理字符串类型的问题，如果没有需要对特定类型做明确的处理，最好也最快的方法是根本不用给他们指定类型，这意味着Cython中的字符串对象会被视为通用PyObject*。

​	举例

```cython
cdef str a='Hello....'
cdef str b='World....'
cdef str c=a+b
```

​	**str仍然会驱动Cpy的内存管理器**，做Py级别的字符操作,事实上**起不了多大用**。

对上面代码块优化的办法就是**声明Cpp的内置字**符类型

```cython
from libcpp.string cimport string
cdef string a='Hello....'
cdef string b='World....' 
cdef string c=a+b
```

这里推荐一下**Cpp的libcpp.string的String类型**,不建议使用**C的char***指针，因为这需要额外手动实现这些字符串垃圾回收的繁琐逻辑，Cpp自带的足够高效，而且用法接近Cpy(**Cpython的str就是借鉴的cpp**),cpp和cpython的都有自己的内存回收逻辑。

​	这里声明了cpp的string没有调用Cpython的堆栈，如果有大量的字符拼接想要Cpp优化,做类似于上面代码块的操作就行。

​	字符串的操作，在返回之前（注意这个描述）以C/C++原生的数据类型去执行，效率相对高效，这时才需要在Cython代码的上下文静态指定C数据类型char\*、std::string。然而执行到return语句之后，由于Python字符串的PyObject属于非常糟糕的过度封装。会导致之前Cython所做的性能优化大打折扣。简而言之，Cython对字符串输出的优化的空间其实非常有限。

​	事实上，Cpython的原生字符性能足够，而且字符操作优化在Cython优化中中上限比较明显，这里不应该是学习 Cython的重点。

​                                         

## Cython中的数组类型和循环

Python循环灵活，简单，语法和缩进写的标准的话读起来接近自然语言。Cython中的循环也借鉴了CPython的循环，基本不需要修改正常使用。但是，要注意数据的类型，保留一些指针，保证使用的时候尽量不去调度Cpython过度封装的方法或者实例化对象。

举个例子,用ipython来实现Cython代码

```cython
%load_ext cython
%%cython  -a

N=100
cdef long long res=0
cdef list a  = list(range(N))
for i in a:
    res+=i
print(res)
```



Cython里面的不同数组类型

1、Cpython:Pyobject-List

​		顾名思义，这个list就是cpython原生的数组对象。Cython的版本list实质上是静态版本的PyListObject这个C级别的实现。

​	上面的代码块就是用python原生对象(list)进行递归的循环。

​	比较简单的优化思路就是**针对代码中的动态类型指定参数**，减少不必要的数据类型判断反复的去调度PyObj的类型检查。

```cython
%%cython  -a
cdef long N=100
cdef long i
cdef long long res=0
cdef list a  = list(range(N))
for i in a:
    res+=i
print(res)
```

2、Cython;np.ndarray

Cython对numpy支持非常友好(因为numpy的底层数组基本也是C/Cpp，所以调度起来只要绕开类型检查效率仍然非常快)

```python
%%Cython

import numpy as np

cimport numpy  as np

cdef:

		long int = total

		int k;

		double t1,t2;

		np.ndarray arr;

arr = np.arange(10000)

for k in arr:

		total = total + k

print('Total = ',total)
```

注意，声明数组的时候可以声明的更加具体

**cdef np.ndarray[np.int_t,ndim=1]**

ndim指定的是数组的维数，一般的数组维数就为1，矩阵型为2，张量或者更高阶的需要自己指定维度。

代码就变成了这样子

```cython
%%cython
import time
import numpy as np 
cimport numpy as np 
ctypedef numpy.int_t DTYPE_t

cdef：
 int n
 long total
 int k
 double t1,t2
 np.ndarray[DTYPE_t,ndim=1] arr
    
arr = np.arange(100000)

for k in arr:
    total=total+k
print("total=",total)
```



3、cython-cpp:vector

cpp的vector在Cython基本支持，在Cython中是比较推荐和使用的类型，使用合理是可以在Cython中达到Cpp的效率的。实测应该是比Cython的原生List对象要快一点的

使用方法Cpy的list有点不同，而且要注意Vector对象和方法和list的名称不同,下面是一个简单的Demo

```cython
%%cython --cplus 
# list
from libcpp.vector cimport vector
# 创建一个list对象
list_python = [i for i in range(10)]
cdef vector[int] vec_cpp = list_python
# 输出看看
for i in vec_cpp:
    print(i)
vec_cpp.push_back(11) # 添加元素，类似于append
print(vec_cpp)
```



## Cython中的函数类型

- 由def关键字定义的函数，我们称为**原生的Python函数**
- 由cdef关键字定义的函数，我们称为**C函数**或**Cython函数**
- 由cpdef关键字定义的函数，我们称为**混合函数**
- 由def关键子定义的函数，函数体内出现关键字定义的C类型的参数或局部变量，这样的函数是**混合函数**的特殊形式
- 可以作为参数传递给其他函数

C函数具有最低的调用开销，并且比Python函数快好几十个数量级，但它具有一些特点局限性

### C函数的限制

- 不能在另一个函数中定义
- 具有不可修改的静态分配名称
- 仅接受位置参数
- 不支持参数的默认值









