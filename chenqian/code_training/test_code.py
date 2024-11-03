import numpy as np 
import pandas as pd
# def cartesian(arrays):
#     arrays = [np.asarray(a) for a in arrays]
#     shape = (len(x) for x in arrays)

#     ix = np.indices(shape, dtype=int)
#     ix = ix.reshape(len(arrays), -1).T

#     for n, arr in enumerate(arrays):
#         ix[:, n] = arrays[n][ix[:, n]]

#     return ix

# print(cartesian(([1, 2, 3], [4, 5], [6, 7])))

values_mat = [[0.2,0.1,0.0],[0.1,0.0,0.3],[0.1,0.1,0.1]]
# values_mat = np.array(values_mat)
index_ = [-1,0,1]
df = pd.DataFrame(values_mat,index = index_,columns = (range(1,4)))

def ex_func(x,y,func):
    return func(x,y)

ex= 0

for y_ in df.index:
    for x_ in df.columns:
        # print(y_)
        ex += df.loc[y_,x_] * ex_func(x_,y_,lambda x,y:y/x)