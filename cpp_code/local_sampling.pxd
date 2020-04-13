# distutils: language=c++

cimport cython
import numpy as np
cimport numpy as np


ctypedef np.uint16_t (*nb_func_type)(const int,const int,const int) nogil;



@cython.boundscheck(False)
cdef inline np.uint16_t neighbors_func_0(int x, int y, int L) nogil:
    return ((x+1)%L+L)%L+L*y

@cython.boundscheck(False)
cdef inline np.uint16_t neighbors_func_1(int x, int y, int L) nogil:
    return ((x-1)%L+L)%L+L*y

@cython.boundscheck(False)
cdef inline np.uint16_t neighbors_func_2(int x, int y, int L) nogil:
    return x+L*(((y+1)%L+L)%L)

@cython.boundscheck(False)
cdef inline np.uint16_t neighbors_func_3(int x, int y, int L) nogil:
    return x+L*(((y-1)%L+L)%L)



@cython.boundscheck(False)
cdef inline np.uint16_t neighbors_func_4(int x, int y, int L) nogil:
    return ((x+1)%L+L)%L+L*(((y+1)%L+L)%L)

@cython.boundscheck(False)
cdef inline np.uint16_t neighbors_func_5(int x, int y, int L) nogil:
    return ((x-1)%L+L)%L+L*(((y+1)%L+L)%L)

@cython.boundscheck(False)
cdef inline np.uint16_t neighbors_func_6(int x, int y, int L) nogil:
    return ((x+1)%L+L)%L+L*(((y-1)%L+L)%L)

@cython.boundscheck(False)
cdef inline np.uint16_t neighbors_func_7(int x, int y, int L) nogil:
    return ((x-1)%L+L)%L+L*(((y-1)%L+L)%L)


