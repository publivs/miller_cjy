import numpy  as np
cimport numpy as np
import cython
cimport numpy as np
# 创建一个包含键值对的字典

ctypedef np.float64_t DTYPE_t

@cython.boundscheck(False)
def ndb_adj_valuation(double num, double[:] lst, dict map_dict):
    """
    在有序列表list中找到num所在的区间
    """
    cdef int n = lst.shape[0]
    cdef int left, right, mid
    cdef double res, left_val, right_val
    if num < lst[0] or num > lst[-1]:
        return np.nan
    left, right = 0, n-2
    while left <= right:
        mid = (left + right) // 2
        if lst[mid] <= num < lst[mid+1]:
            left_val, right_val = lst[mid], lst[mid+1]
            break
        elif lst[mid+1] <= num:
            left = mid + 1
        else:
            right = mid - 1
    res = ((num - left_val)/(right_val - left_val)) *(map_dict.get(right_val) - map_dict.get(left_val)) + map_dict.get(left_val)
    return res

@cython.boundscheck(False)
@cython.wraparound(False)
def calc_valuation_df_cython(np.ndarray[DTYPE_t, ndim=1] bond_valuation_lst,
                             np.ndarray[DTYPE_t, ndim=1] remaining_term_lst,
                             dict NDB_valuation_dict):

    cdef np.ndarray[DTYPE_t, ndim=1] NDB_valuation_lst = np.asarray(list(NDB_valuation_dict.keys()), dtype=np.float64)
    cdef np.ndarray[DTYPE_t, ndim=1] target_list = np.zeros_like(remaining_term_lst)

    cdef Py_ssize_t i

    for i in range(remaining_term_lst.shape[0]):
        target_list[i] = bond_valuation_lst[i] - ndb_adj_valuation(remaining_term_lst[i], NDB_valuation_lst, NDB_valuation_dict)

    return target_list.tolist()