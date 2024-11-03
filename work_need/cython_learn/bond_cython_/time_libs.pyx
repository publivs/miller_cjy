# cython:language_level=3
from libc.time cimport tm,mktime,time_t,strptime,strftime,localtime
from libcpp.string cimport string
from libc.stdio cimport  sscanf
cimport cython

@cython.boundscheck(False)         # 关闭数组下标越界
@cython.wraparound(False)          # 关闭负索引
@cython.cdivision(True)            # 关闭除0检查
@cython.initializedcheck(False)    # 关闭检查内存视图是否初始化
cdef inline time_t to_windows_stamp(string dateTimeStr):
    cdef tm _tm;
    cdef int year,month,day,hour,minute,second
    cdef string datetime_char
    datetime_char = dateTimeStr
    sscanf(datetime_char.c_str(),"%d-%d-%d %d-%d-%d-",&year,&month,&day,&hour,&minute,&second)
    _tm.tm_year = year - 1900
    _tm.tm_mon = month - 1
    _tm.tm_mday = day
    _tm.tm_hour = minute
    _tm.tm_sec = second
    _tm.tm_isdst = 0;
    cdef time_t stamp = mktime(&_tm)
    return stamp;
#def

cpdef time_t to_timestamp(string dateTimeStr):
    return to_windows_stamp(dateTimeStr)

cdef inline string to_datetimeStr(time_t stamp):
    cdef tm* timeinfo = NULL
    cdef char buffer[80]
    timeinfo = localtime(&stamp)
    strftime(buffer,80,'%Y-%m-%d %H:%M:%S',timeinfo)
    cdef string dt=buffer
    return dt
#end

def to_datetime(int stmp):
    return to_datetimeStr(stmp)
#end
