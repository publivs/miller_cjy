import numpy as np
from numpy.lib import stride_tricks
import pandas as pd

print(np.__version__)
print(np.show_config())

Z = np.zeros(10)
print(Z)

Z = np.zeros((10,10))
print("%d bytes" % (Z.size * Z.itemsize))

# 设置报错
defaults = np.seterr(all='ignore')
Z = np.ones(1)/0

# 获得numpy 的日期
Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')



# 利用迭代对象生成数组
def generate():
    for x  in range(10):
        yield x 
Z = np.fromiter(generate(),dtype=float,count=-1)
print(Z)

#查看公共元素 
Z1 = np.random.randint(0, 10, 10)
Z2 = np.random.randint(0, 10, 10)
print (np.intersect1d(Z1, Z2))

# 47 柯西矩阵
A = np.arange(8)
B = 1+A
# 广播函数直接(Xij-Yij)
C = np.subtract.outer(A,B)
print(C)


# 结构化数组
# 表示位置(x, y)和颜色(r, g, b, a)的结构化数组
Z = np.zeros(10, [('position', [('x', float, 1), 
                                ('y', float, 1)]),
                  ('color',    [('r', float, 1), 
                                ('g', float, 1), 
                                ('b', float, 1)])])
print (Z)

# 构造二维的高斯矩阵

X, Y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(-1, 1, 10))
D = np.sqrt(X**2 + Y**2)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / (2.0*sigma**2) ))
print (G)

# 按照指定列排序
Z = np.random.randint(0,10,(3,3))
print(Z)
print(Z[ Z[:,1].argsort() ])


# 如何判定一个给定的二维数组存在空列
Z = np.random.randint(0,3,(3,10))
# 说实话这种API我还不如我自己for循环呢
print((~Z.any(axis=0)).any())

# 从数组中找出与给定值最接近的值 (★★☆)
Z = np.random.uniform(0,1,10)
z = 0.5
m = Z.flat[np.abs(Z - z).argmin()]
print(m)

# 使用迭代器计算
A = np.arange(3).reshape(3, 1)
B = np.arange(3).reshape(1, 3)
it = np.nditer([A, B, None])

# 创建一个具有name属性的数组类 (★★☆)
class NameArray(np.array):
    def __new__(cls,array,name='no_name'):
        # 这里确定对象的时候很
        obj = np.asarray(array).view(cls)
        obj.name =name
        return obj
        
    def __array_finalize__(self, obj):
            if obj is None: return
            self.info = getattr(obj, 'name', "no name")


# 定一个向量，如何让在第二个向量索引的每个元素加1(注意重复索引)? (★★★)




# 考虑一维向量D，如何使用相同大小的向量S来计算D的子集的均值，其描述子集索引？(★★★)
import pandas as pd
D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
print(pd.Series(D).groupby(S).mean())

# 思考描述10个三角形（共享顶点）的一组10个三元组，找到组成所有三角形的唯一线段集 (★★★)




# 不会的部分
# no 78 思考两组点集P0和P1去描述一组线(二维)
# 和一个点p,如何计算点p到每一条线 i (P0[i],P1[i])的距离？(★★★)
def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1)
    U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U*T - p
    return np.sqrt((D**2).sum(axis=1))

P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p  = np.random.uniform(-10,10,( 1,2))
print(distance(P0, P1, p))

# no 79 

# 79. 考虑两组点集P0和P1去描述一组线(二维)和一组点集P，
# 如何计算每一个点 j(P[j]) 到每一条线 i (P0[i],P1[i])的距离? (★★★)
P0 = np.random.uniform(-10, 10, (10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10, 10, (10,2))
print(np.array([distance(P0,P1,p_i) for p_i in p]))


# 考虑一个数组Z =
#  [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
# 如何生成一个数组R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ...,[11,12,13,14]]? 
# (★★★)
Z = np.arange(1,15)
R = stride_tricks.as_strided(Z,(11,4),(4,4))



# 查看矩阵的秩
Z = np.random.uniform(0,1,(10,10))
# SVD解
U, S, V = np.linalg.svd(Z) # Singular Value Decomposition
rank = np.sum(S > 1e-10)
# 矩阵解
rank_2 = np.linalg.matrix_rank(Z)
print(rank,rank_2)

# 频率抽查
Z = np.random.randint(0,10,50)
print(np.bincount(Z).argmax())

# 创建一个满足 Z[i,j] == Z[j,i]的二维数组子类 (★★★)
class Symetric(np.ndarray):
    def __setitem__(self, index, value):
        i,j = index
        super(Symetric, self).__setitem__((i,j), value)
        super(Symetric, self).__setitem__((j,i), value)

def symetric(Z):
    return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)
S = symetric(np.random.randint(0,10,(5,5)))

# 这是拉是对称阵的资料
# np.triu()是过去上三角

np.triu(X)
X = X + X.T - np.diag(X.diagonal())

# 如何找到一个数组的第N个最大值
Z = np.arange(10000)
np.random.shuffle(Z)
n =5 
top_n_arr = Z[np.argsort(Z)][-n:]

# 矩阵分块求和
Z = np.ones((16,16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0), np.arange(0, Z.shape[1], k), axis=1)


%timeit np.power(x,3)
%timeit x*x*x
%timeit np.einsum('i,i,i->i',x,x,x)

# 289 ms ± 650 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)
# 67.4 ms ± 305 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
# 42.1 ms ± 493 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

'''
NP的内置的api咋能差这么多,
符号运算确实太猛了,向量乘除后面是不是尽量考虑用einsum这种猛人算法，快TM八倍
'''

A =np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))

# 逐行判断,enumerate或者for循环走行遍历都行,用df更方便,不过会影响性能，
# 能不用DF就不用df
A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))

C = (A[..., np.newaxis, np.newaxis] == B)
rows = np.where(C.any((3,1)).all(axis = 1))[0]
print(rows)



# 
Z = np.random.randint(0,5,(10,3))
print(Z)

U = Z[Z.max(axis=1) != Z.min(axis=1),:]
print(U)