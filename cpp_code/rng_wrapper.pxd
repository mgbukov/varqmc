# distutils: language=c++


# cdef extern from "<stdlib.h>" nogil:
#     int rand_r(unsigned int *seed) nogil;

'''
## cpp mt19937: different on linux and osx 
cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937 nogil:
        mt19937() nogil # we need to define this constructor to stack allocate classes in Cython
        mt19937(unsigned int seed) nogil # not worrying about matching the exact int type for seed

    cdef cppclass uniform_real_distribution[T] nogil:
        uniform_real_distribution() nogil
        uniform_real_distribution(T a, T b) nogil
        T operator()(mt19937 gen) nogil # ignore the possibility of using other classes for "gen"

    cdef cppclass uniform_int_distribution[T] nogil:
        uniform_int_distribution() nogil
        uniform_int_distribution(T a, T b) nogil
        T operator()(mt19937 gen) nogil # ignore the possibility of using other classes for "gen"
'''


cdef extern from "boost/random/mersenne_twister.hpp" namespace "boost::random" nogil:
    cdef cppclass mt19937 nogil:
        mt19937() nogil # we need to define this constructor to stack allocate classes in Cython
        mt19937(unsigned int seed) nogil # not worrying about matching the exact int type for seed

cdef extern from "boost/random/uniform_int_distribution.hpp" namespace "boost::random" nogil:
    cdef cppclass uniform_int_distribution[T] nogil:
        uniform_int_distribution() nogil
        uniform_int_distribution(T a, T b) nogil
        T operator()(mt19937 gen) nogil # ignore the possibility of using other classes for "gen"

cdef extern from "boost/random/uniform_real_distribution.hpp" namespace "boost::random" nogil:
    cdef cppclass uniform_real_distribution[T] nogil:
        uniform_real_distribution() nogil
        uniform_real_distribution(T a, T b) nogil
        T operator()(mt19937 gen) nogil # ignore the possibility of using other classes for "gen"











