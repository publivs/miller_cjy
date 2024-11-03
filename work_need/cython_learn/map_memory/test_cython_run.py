import numpy as np
from test_multithread import test_parallel,test_serial

LENGTH = 1024
FACTOR = 100
NUM_TOTAL_ELEMS = FACTOR*LENGTH*LENGTH

x1 = -1+2*np.random.rand(NUM_TOTAL_ELEMS)
x2 =  -1+2*np.random.rand(NUM_TOTAL_ELEMS)
y = np.zeros(x1.shape)

if __name__ == '__main__':
    test_parallel(x1,x2,y)
    test_serial(x1,x2,y)