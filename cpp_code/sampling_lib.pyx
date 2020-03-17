# distutils: language=c++
# cython: language_level=2
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: profile=False

from jax.config import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import jit, grad, random, device_put, partial
from jax.tree_util import tree_structure, tree_flatten, tree_unflatten


cimport cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libc.stdlib cimport rand, srand, RAND_MAX
from libcpp cimport bool
from libc.math cimport exp #, sin, cos, acos, sqrt, fabs, M_PI, floor, ceil

from cython.parallel cimport prange, threadid, parallel
cimport openmp


from DNN_architectures_cpx import *
from DNN_architectures_real import *

from reshape_class import NN_Tree
from functools import partial   


##############################################
# linear square lattice dimension

DEF _L=4
cdef extern from *:
    """
    #define _L 4
    """
    pass


##############################################



IF _L==4:
    ctypedef np.uint16_t basis_type
    basis_type_py=np.uint16
IF _L==6:
    ctypedef np.uint64_t basis_type
    basis_type_py=np.uint64
IF _L==8:
    ctypedef np.uint64_t basis_type
    basis_type_py=np.uint64


N_sites=_L*_L
ctypedef basis_type (*rep_func_type)(const int,basis_type,np.int8_t*) nogil;
ctypedef void (*func_type)(const int,basis_type,np.int8_t*) nogil;


### enables OMP pragmas in cython
cdef extern from *:
    """
    #define START_OMP_PARALLEL_PRAGMA() _Pragma("omp parallel") {
    #define END_OMP_PRAGMA() }
    #define START_OMP_SINGLE_PRAGMA() _Pragma("omp single") {
    #define START_OMP_CRITICAL_PRAGMA() _Pragma("omp critical") {
    #define OMP_BARRIER_PRAGMA() _Pragma("omp barrier")   
    """
    void START_OMP_PARALLEL_PRAGMA() nogil
    void END_OMP_PRAGMA() nogil
    void START_OMP_SINGLE_PRAGMA() nogil
    void START_OMP_CRITICAL_PRAGMA() nogil
    void OMP_BARRIER_PRAGMA() nogil


cdef extern from "<stdlib.h>" nogil:
    int rand_r(unsigned int *seed) nogil;

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



cdef extern from "sample.h":
        
    int choose_n_k(int, int) nogil
    
    T swap_bits[T](const T, int, int) nogil
    T magnetized_int[T](int, int, T) nogil

    int update_offdiag[I](const int, const char[], const int[], const double, const int, const int, const I[], I[], np.int8_t [], np.int32_t[], double[], I (*)(const int,I,np.int8_t*) ) nogil  
    
    void update_diag[I](const int, const char[], const int[], const double, const int, const I[], double[] ) nogil
    
    void offdiag_sum(int,int[],double[],double[],np.uint32_t[],double[],const double[],const double[],const double[]) nogil

    void int_to_spinstate[T,J](const int,T ,J []) nogil
    void int_to_spinstate_conv[T,J](const int,T ,J []) nogil

    T cyclicity[T](const int,T) nogil

    T rep_int_to_spinstate[T,J](const int,T ,J []) nogil
    T rep_int_to_spinstate_conv[T,J](const int,T ,J []) nogil



@cython.boundscheck(False)
def swap_spins(basis_type s, int i, int j):
    cdef basis_type t;

    with nogil:
        t = swap_bits(s,i,j)

    return t


@cython.boundscheck(False)
def integer_to_spinstate(basis_type[:] states,np.int8_t[::1] out, int N_features, object NN_type='DNN'):
    cdef int i;
    cdef int Ns=states.shape[0]
    cdef int Nsites=N_sites

    cdef func_type spin_config

    if NN_type=='DNN':
        spin_config=<func_type>int_to_spinstate
    elif NN_type=='CNN':
        spin_config=<func_type>int_to_spinstate_conv
    else:
        raise ValueError("unsupported string for variable NN_type.")


    with nogil:
        for i in range (Ns):
            spin_config(Nsites,states[i],&out[i*N_features])


@cython.boundscheck(False)
def integer_cyclicity(basis_type[:] states,np.npy_uint32[::1] cycl):
    cdef int i;
    cdef int Ns=states.shape[0];
    cdef int Nsites=N_sites;
    

    with nogil:
        for i in range (Ns):
            cycl[i]=cyclicity(Nsites,states[i])


@cython.boundscheck(False)
def update_diag_ME(np.ndarray ket,double[::1] M,object opstr,int[::1] indx,double J):
    cdef char[::1] c_opstr = bytearray(opstr,"utf-8")
    cdef int n_op = indx.shape[0]
    cdef int Ns = ket.shape[0]
    cdef void * ket_ptr = np.PyArray_GETPTR1(ket,0)

    with nogil:
        update_diag(n_op,&c_opstr[0],&indx[0],J,Ns,<basis_type*>ket_ptr,&M[0])


@cython.boundscheck(False)
def update_offdiag_ME(np.ndarray ket, basis_type[:] bra, np.int8_t[:,:] spin_bra,np.int32_t[:] ket_indx,double[::1] M,object opstr,int[::1] indx,double J,int N_symm, object NN_type='DNN'):
    cdef char[::1] c_opstr = bytearray(opstr,"utf-8")
    cdef int n_op = indx.shape[0]
    cdef int Ns = ket.shape[0]
    cdef int l
    
    cdef void * ket_ptr = np.PyArray_GETPTR1(ket,0)

    cdef rep_func_type rep_spin_config

    if NN_type=='DNN':
        rep_spin_config=<rep_func_type>rep_int_to_spinstate
    elif NN_type=='CNN':
        rep_spin_config=<rep_func_type>rep_int_to_spinstate_conv
    else:
        raise ValueError("unsupported string for variable NN_type.") 

    with nogil:
        l=update_offdiag(n_op,&c_opstr[0],&indx[0],J,Ns,N_symm,<basis_type*>ket_ptr,&bra[0],&spin_bra[0,0],&ket_indx[0],&M[0],rep_spin_config)

    return l



@cython.boundscheck(False)
def c_offdiag_sum(
                double[::1] Eloc_cos,
                double[::1] Eloc_sin,
                int[::1] n_per_term,
                np.uint32_t[::1] ket_indx,
                double[::1] MEs,
                const double[::1] log_psi_bras,
                const double[::1] phase_psi_bras,
                const double[::1] log_psi_kets
                ):
    
    cdef int Ns = n_per_term.shape[0]
    
    with nogil:
        offdiag_sum(Ns,&n_per_term[0],&Eloc_cos[0],&Eloc_sin[0],&ket_indx[0],&MEs[0],&log_psi_bras[0],&phase_psi_bras[0],&log_psi_kets[0])
        







###########################







@cython.boundscheck(False)
cdef class Neural_Net:

    cdef object NN_Tree, NN_Tree_vec

    cdef object NN_architecture, NN_architecture_dyn
    cdef object apply_layer, apply_layer_dyn
    cdef object apply_fun_args, apply_fun_args_dyn
    
    #cdef object W_real, W_imag
    cdef object params
    cdef object input_shape, reduce_shape, output_shape, out_chan, strides, filter_shape 
    cdef object NN_type, NN_dtype
    cdef object comm

    cdef int MPI_rank, seed
    cdef vector[unsigned int] thread_seeds

    cdef int N_varl_params, N_symm, N_sites, N_features
    cdef object N_varl_params_vec
    cdef int N_MC_chains, N_spinconfigelmts, N_layers

    cdef object evaluate_phase, evaluate_log

    cdef vector[double] log_psi_s, log_psi_t
    
    cdef np.int8_t[::1] spinstate_s, spinstate_t
    cdef object spinstate_s_py, spinstate_t_py
    cdef func_type spin_config
    cdef basis_type[::1] sf_vec, s0_vec

    
    cdef vector[np.uint16_t] sites

    


    cdef uniform_real_distribution[double] random_float
    cdef uniform_int_distribution[int] random_int, rand_int_ordinal
    cdef vector[mt19937] RNGs # hold a C++ instance
        

    def __init__(self,comm,shapes,N_MC_chains,NN_type='DNN',NN_dtype='cpx',seed=0):

        self.N_sites=_L*_L

        self.comm=comm
        self.MPI_rank=self.comm.Get_rank()
 
        # fix seed
        self.seed=seed
        np.random.seed(self.seed+self.MPI_rank)
        np.random.RandomState(self.seed+self.MPI_rank)
        srand(self.seed+self.MPI_rank)
        rng = random.PRNGKey(self.seed) # same seed for all MPI processes to keep NN params the same
        
        # order important
        self._init_NN(rng,shapes,NN_type,NN_dtype)
        self._init_evaluate()
        self._init_variables(N_MC_chains)
        self._init_MC_data()


    def _init_NN(self,rng,shapes,NN_type,NN_dtype):

        self.NN_type=NN_type
        self.NN_dtype=NN_dtype

        if NN_type=='DNN':
           
            self.N_symm=_L*_L*2*2*2 # no Z symmetry
          
            # define DNN
            

            if self.NN_dtype=='cpx':

                shape_last_layer=shapes['layer_1']


                self.NN_architecture = {
                                        'layer_1': GeneralDense_cpx(shapes['layer_1'], ignore_b=True, init_value_W=1E-2,init_value_b=1E-2,), 
                                        'nonlin_1': elementwise(poly_cpx),
                                        #'layer_2': GeneralDense_cpx_nonholo(shapes['layer_2'], ignore_b=False, init_value_W=1E-1,init_value_b=1E-1,),
                                        #'nonlin_2': Poly_cpx,    
                                        'reg': Regularization_cpx((shape_last_layer[1],)),
                                    
                                        #'norm_1': Norm_real(),
                                    #    'batch_norm_1': BatchNorm_cpx(axis=(0,)), # Normalize_cpx,
                                    
                                    #    'layer_2': GeneralDense_cpx(shapes['layer_2'], ignore_b=False),
                                    
                                    #    'batch_norm_2': BatchNorm_cpx(axis=(0,)), # Normalize_cpx,
                                    #    'layer_3': GeneralDense_cpx_nonholo(shapes['layer_3'], ignore_b=False), 
                                    }

                # determine shape variables
                input_shape=(1,self.N_sites)
                reduce_shape = (-1,self.N_symm,shape_last_layer[1]) 
                output_shape = (-1,shape_last_layer[1],)

                # create a copy of the NN architecture to update the batch norm mean and variance dynamically
                self.params, self.apply_layer, self.apply_fun_args, self.apply_layer_dyn, self.apply_fun_args_dyn = self._compute_layers(rng, self.NN_architecture, input_shape, output_shape, reduce_shape)


                self.NN_Tree = NN_Tree(self.params)
                self.NN_Tree_vec=[NN_Tree(self.params), ]

                self.N_varl_params=self.NN_Tree.N_varl_params
                self.N_varl_params_vec=[self.N_varl_params,] 


            elif self.NN_dtype=='real-decoupled':

                shape_last_layer_log  =shapes[0]['layer_1']
                shape_last_layer_phase=shapes[1]['layer_2']


                NN_arch_log = {
                                        'layer_1': GeneralDense_cpx(shapes[0]['layer_1'], ignore_b=True, init_value_W=1E-2),  # 5E-2
                                        'nonlin_1': elementwise(poly_real),
                                        #
                                    #    'layer_2': GeneralDense_cpx(shapes[0]['layer_2'], ignore_b=False, init_value_W=1E-1,init_value_b=1E-0,),
                                    #    'nonlin_2': elementwise(poly_real),
                                        #
                                        'reg': Regularization((shape_last_layer_log[1],)),
                            }


                NN_arch_phase = {
                                #        'layer_1': GeneralDense_cpx(shapes[1]['layer_1'], ignore_b=True, init_value_W=1E-1,init_value_b=1E-1,), 
                                #        'nonlin_1': Poly_cpx,
                                #         'reg': Regularization_cpx2((shape_last_layer_phase[1],)),

                                        'layer_1': GeneralDense(shapes[1]['layer_1'], ignore_b=True, init_value_W=1E-1, ), #3E-1
                                        'nonlin_1': elementwise(logcosh),
                                        'layer_2': GeneralDense(shapes[1]['layer_2'], ignore_b=False, init_value_W=5E-2, init_value_b=5E-2), #init_value_W=5E-1, init_value_b=5E-1
                                        'nonlin_2': elementwise(logcosh),
                                        # 'layer_3': GeneralDense(shapes[1]['layer_3'], ignore_b=False, init_value_W=1E-1, init_value_b=1E-1), #4.3E-2
                                        # 'nonlin_3': elementwise(logcosh),
                                        'reg': Phase_arg((shape_last_layer_phase[1],)),
                                    }

                self.NN_architecture=(NN_arch_log, NN_arch_phase)


                # determine shape variables
                input_shape=(1,self.N_sites)

                reduce_shape_log = (-1,self.N_symm,shape_last_layer_log[1])
                reduce_shape_phase = (-1,self.N_symm,shape_last_layer_phase[1]) 

                output_shape_log = (-1,shape_last_layer_log[1],)
                output_shape_phase = (-1,shape_last_layer_phase[1],)


                # create a copy of the NN architecture to update the batch norm mean and variance dynamically
                params_log,   apply_layer_log,   apply_fun_args_log,   apply_layer_dyn_log,   apply_fun_args_dyn_log  = self._compute_layers(rng, NN_arch_log  , input_shape, output_shape_log, reduce_shape_log)
                params_phase, apply_layer_phase, apply_fun_args_phase, apply_layer_dyn_phase, apply_fun_args_dyn_phase= self._compute_layers(rng, NN_arch_phase, input_shape, output_shape_phase, reduce_shape_phase)

                
                                
                self.params=(params_log, params_phase)
                self.apply_fun_args    =(apply_fun_args_log,     apply_fun_args_phase)
                self.apply_fun_args_dyn=(apply_fun_args_dyn_log, apply_fun_args_dyn_phase)



                self.apply_layer = lambda params, inputs, apply_fun_args: ( apply_layer_log  (params[0], inputs, kwargs=apply_fun_args[0]) ,
                                                                            apply_layer_phase(params[1], inputs, kwargs=sapply_fun_args[1]) , 
                                                                        )


                self.apply_layer_dyn = lambda params, inputs, kwargs=self.apply_fun_args_dyn: ( apply_layer_dyn_log  (params[0], inputs, kwargs=kwargs[0]) ,
                                                                                                apply_layer_dyn_phase(params[1], inputs, kwargs=kwargs[1]) , 
                                                                                ) 
                

                self.NN_Tree = NN_Tree(self.params)
                self.NN_Tree_vec=[NN_Tree(params_log), NN_Tree(params_log)]
                
                N_var_log=NN_Tree(params_log).N_varl_params
                N_var_phase=NN_Tree(params_phase).N_varl_params

                self.N_varl_params=self.NN_Tree.N_varl_params 
                self.N_varl_params_vec=[N_var_log, N_var_phase, ]


            else:
                raise ValueError('NN_dtype not implemented.')

            
        
            self.input_shape=(-1,self.N_sites)
          
            

            


        elif NN_type=='CNN':
            
            self.N_symm=2*2*2 # no Z, Tx, Ty symmetry

            dim_nums=('NCHW', 'OIHW', 'NCHW') # default
            
            # define CNN
            init_value_W = 1E-3
            init_value_b = 1E-1
            W_init = partial(random.uniform, minval=-init_value_W, maxval=+init_value_W )
            b_init = partial(random.uniform, minval=-init_value_b, maxval=+init_value_b )


            self.NN_architecture = {
                                    'layer_1': GeneralConv_cpx(dim_nums, shapes['layer_1']['out_chan'], shapes['layer_1']['filter_shape'], strides=shapes['layer_1']['strides'], padding='PERIODIC', ignore_b=True, W_init=W_init, b_init=b_init), 
                                    'nonlin_1': Poly_cpx,
                                #    'batch_norm_1': Normalize_cpx,
                                #    'layer_2': GeneralConv_cpx(dim_nums, shapes['layer_2']['out_chan'], shapes['layer_2']['filter_shape'], strides=shapes['layer_2']['strides'], padding='PERIODIC', ignore_b=False, W_init=W_init, b_init=b_init),     
                                }

            init_params, self.apply_layer, self.apply_fun_args = serial(*self.NN_architecture)
            
            input_shape=np.array((1,1,_L,_L),dtype=np.int) # NCHW input format
            output_shape, self.params = init_params(rng,input_shape)

            
            self.input_shape = (-1,1,_L,_L) # reshape input data batch
            self.reduce_shape = (-1,self.N_symm,)+output_shape[1:] #(-1,self.N_symm,out_chan,_L,_L) # tuple to reshape output before symmetrization
            self.output_shape = (-1,np.prod(output_shape[1:]) ) #(-1,out_chan*_L*_L)
           

            self.NN_Tree = NN_Tree(self.params)
            self.N_varl_params=self.NN_Tree.N_varl_params


        else:
            raise ValueError("unsupported string for variable for NN_type.") 
        
        

        self.N_features=self.N_sites*self.N_symm



    def _compute_layers(self,rng,NN_architecture, input_shape, output_shape, reduce_shape):

        NN_architecture_dyn = NN_architecture.copy()
        for key in NN_architecture.keys():
            if 'batch_norm' in key:
                NN_architecture_dyn[key]=BatchNorm_cpx_dyn(axis=(0,))

        # create NN
        init_params, apply_layer, apply_fun_args = serial(*NN_architecture.values())
        _, apply_layer_dyn, apply_fun_args_dyn = serial(*NN_architecture_dyn.values())

        _, params = init_params(rng,input_shape)

        self._init_apply_fun_args(rng, NN_architecture, apply_fun_args,apply_fun_args_dyn, input_shape, output_shape, reduce_shape)

        return params, apply_layer, apply_fun_args, apply_layer_dyn, apply_fun_args_dyn



    def _init_apply_fun_args(self,rng, NN_architecture, apply_fun_args,apply_fun_args_dyn, input_shape, output_shape, reduce_shape):
        
        layers_type=list(NN_architecture.keys())
        init_funs, apply_funs = zip(*NN_architecture.values())

        for j, (init_fun, layer_type) in enumerate(zip(init_funs, layers_type)):        
            
            rng, layer_rng = random.split(rng)
            input_shape, _ = init_fun(layer_rng, input_shape)

            if 'batch_norm' in layer_type:
                mean, std_mat_inv = init_batchnorm_cpx_params(input_shape)
                
                # an update in the parameters of apply_fun_args_dyn UPDATES directly the params of apply_fun_args
                apply_fun_args[j]=dict(mean=mean, std_mat_inv=std_mat_inv, )        
                apply_fun_args_dyn[j]=dict(fixpoint_iter=False, mean=mean, std_mat_inv=std_mat_inv, comm=self.comm, )
        
            if 'reg' in layer_type:
                D=dict(reduce_shape=reduce_shape, output_shape=output_shape)
                apply_fun_args[j]=D
                apply_fun_args_dyn[j]=D



    def _init_evaluate(self):

        # self.evaluate_log  =self._evaluate_log
        # self.evaluate_phase=self._evaluate_phase

        # define network evaluation on GPU
        self.evaluate_log  =jit(self._evaluate_log)
        self.evaluate_phase=jit(self._evaluate_phase)
        
        if self.NN_type=='DNN':
            self.spin_config=<func_type>int_to_spinstate
        elif self.NN_type=='CNN':
            self.spin_config=<func_type>int_to_spinstate_conv


    def _init_variables(self,N_MC_chains):

        self.N_MC_chains=N_MC_chains
        self.N_spinconfigelmts=self.N_features

        self.spinstate_s=np.zeros(self.N_MC_chains*self.N_features,dtype=np.int8)
        self.spinstate_t=np.zeros(self.N_MC_chains*self.N_features,dtype=np.int8)

        # access data in device array; transfer memory from numpy to jax
        if self.NN_type=='DNN':
            spinstate_shape=[self.N_MC_chains*self.N_symm,self.N_sites]
        elif self.NN_type=='CNN':
            spinstate_shape=[self.N_MC_chains*self.N_symm,1,_L,_L]
        self.spinstate_s_py=np.asarray(self.spinstate_s).reshape(spinstate_shape)
        self.spinstate_t_py=np.asarray(self.spinstate_t).reshape(spinstate_shape)

        self.log_psi_s=np.zeros(self.N_MC_chains,dtype=np.float64)
        self.log_psi_t=np.zeros(self.N_MC_chains,dtype=np.float64)

        ###############################################################

        self.sites=np.arange(self.N_sites,dtype=np.uint16)


        self.random_float = uniform_real_distribution[double](0.0,1.0)
        self.random_int = uniform_int_distribution[int](0,self.N_sites-1)
        self.rand_int_ordinal = uniform_int_distribution[int](0,choose_n_k(self.N_sites, self.N_sites//2))


    def _init_MC_data(self, s0_vec=None, sf_vec=None, ):

        self.thread_seeds=np.zeros(self.N_MC_chains,dtype=np.uint)
        self.s0_vec=np.zeros(self.N_MC_chains,dtype=basis_type_py)
        self.sf_vec=np.zeros(self.N_MC_chains,dtype=basis_type_py)

        for i in range(self.N_MC_chains):

            if s0_vec is None:
                self.s0_vec[i]=0
            else:
                self.s0_vec[i]=s0_vec[i]

            if sf_vec is None:
                self.sf_vec[i]=(1<<(self.N_sites//2))-1
            else:
                self.sf_vec[i]=sf_vec[i]


            self.thread_seeds[i]=self.seed + 3333*self.MPI_rank + 7777*i   #(rand()%RAND_MAX)
            self.RNGs.push_back( mt19937(self.thread_seeds[i]) )




    property input_shape:
        def __get__(self):
            return self.input_shape
  
    property N_sites:
        def __get__(self):
            return self.N_sites

    property N_features:
        def __get__(self):
            return self.N_features

    property N_symm:
        def __get__(self):
            return self.N_symm

    property NN_type:
        def __get__(self):
            return self.NN_type

    property NN_dtype:
        def __get__(self):
            return self.NN_dtype

    property NN_architecture:
        def __get__(self):
            return self.NN_architecture

    property apply_fun_args:
        def __get__(self):
            return self.apply_fun_args
        def __set__(self,value):
            self.apply_fun_args=value

    property apply_fun_args_dyn:
        def __get__(self):
            return self.apply_fun_args_dyn
        def __set__(self,value):
            self.apply_fun_args_dyn=value

    property NN_Tree:
        def __get__(self):
            return self.NN_Tree

    property NN_Tree_vec:
        def __get__(self):
            return self.NN_Tree_vec

    property N_varl_params:
        def __get__(self):
            return self.N_varl_params 

    property N_varl_params_vec:
        def __get__(self):
            return self.N_varl_params_vec

    property params:
        def __get__(self):
            return self.params
        def __set__(self,value):
            self.params=value

    property evaluate_phase:
        def __get__(self):
            return self.evaluate_phase

    property evaluate_log:
        def __get__(self):
            return self.evaluate_log

    property apply_layer:
        def __get__(self):
            return self.apply_layer

    property s0_vec:
        def __get__(self):
            return self.s0_vec

    property sf_vec:
        def __get__(self):
            return self.sf_vec


    @cython.boundscheck(False)
    def update_params(self,params):
        self.params=params



    @cython.boundscheck(False)
    def evaluate_dyn(self, params, batch,):

        # reshaping required inside evaluate func because of per-sample gradients
        batch=batch.reshape(self.input_shape)

        # # apply dense layer
        # Re_Ws, Im_Ws = self.apply_layer_dyn(params,batch,kwargs=self.apply_fun_args_dyn)
        # # apply logcosh nonlinearity
        # Re_z, Im_z = poly_cpx((Re_Ws, Im_Ws))
        # #Re_z, Im_z = logcosh_cpx((Re_Ws, Im_Ws))

        # Re_z, Im_z = self.apply_layer_dyn(params,batch,kwargs=self.apply_fun_args_dyn)

        # # symmetrize
        # log_psi   = jnp.sum(Re_z.reshape(self.reduce_shape,order='C'), axis=[1,])
        # phase_psi = jnp.sum(Im_z.reshape(self.reduce_shape,order='C'), axis=[1,])
        # # 
        # log_psi   = jnp.sum(  log_psi.reshape(self.output_shape), axis=[1,])
        # phase_psi = jnp.sum(phase_psi.reshape(self.output_shape), axis=[1,])


        log_psi, phase_psi = self.apply_layer_dyn(params,batch,kwargs=self.apply_fun_args_dyn)
        
        return log_psi, phase_psi


    @cython.boundscheck(False)
    def evaluate(self, params, batch, apply_fun_args):

        # reshaping required inside evaluate func because of per-sample gradients
        batch=batch.reshape(self.input_shape)

        # # apply dense layer
        # Re_Ws, Im_Ws = self.apply_layer(params,batch,kwargs=apply_fun_args)
        # # apply logcosh nonlinearity
        # Re_z, Im_z = poly_cpx((Re_Ws, Im_Ws))
        # #Re_z, Im_z = logcosh_cpx((Re_Ws, Im_Ws))
        
        # Re_z, Im_z = self.apply_layer(params,batch,kwargs=apply_fun_args)

        # # symmetrize
        # log_psi   = jnp.sum(Re_z.reshape(self.reduce_shape,order='C'), axis=[1,])
        # phase_psi = jnp.sum(Im_z.reshape(self.reduce_shape,order='C'), axis=[1,])
        # # 
        # log_psi   = jnp.sum(  log_psi.reshape(self.output_shape), axis=[1,])
        # phase_psi = jnp.sum(phase_psi.reshape(self.output_shape), axis=[1,])

        log_psi, phase_psi = self.apply_layer_dyn(params,batch,kwargs=self.apply_fun_args_dyn)
        
        return log_psi, phase_psi



    @cython.boundscheck(False)
    cpdef object _evaluate_log(self, object params, object batch):

        # reshaping required inside evaluate func because of per-sample gradients
        batch=batch.reshape(self.input_shape)

        # # apply dense layer
        # Re_Ws, Im_Ws = self.apply_layer(params,batch,kwargs=self.apply_fun_args)
        # # apply logcosh nonlinearity
        # Re_z = poly_real((Re_Ws, Im_Ws))
        # #Re_z = logcosh_real((Re_Ws, Im_Ws))
        
        # Re_z, Im_z = self.apply_layer(params,batch,kwargs=self.apply_fun_args)
       
        # # symmetrize
        # log_psi = jnp.sum(Re_z.reshape(self.reduce_shape,order='C'), axis=[1,])
        # # sum over hidden units
        # log_psi = jnp.sum(  log_psi.reshape(self.output_shape), axis=[1,])

        log_psi, phase_psi = self.apply_layer_dyn(params,batch,kwargs=self.apply_fun_args_dyn)
        
        return log_psi



    @cython.boundscheck(False)
    cpdef object _evaluate_phase(self, object params, object batch):

        # reshaping required inside evaluate func because of per-sample gradients
        batch=batch.reshape(self.input_shape)

        # # apply dense layer
        # Re_Ws, Im_Ws = self.apply_layer(params,batch,kwargs=self.apply_fun_args)
        # # apply logcosh nonlinearity
        # Im_z = poly_imag((Re_Ws, Im_Ws))
        # #Im_z = logcosh_imag((Re_Ws, Im_Ws))
        
        # Re_z, Im_z = self.apply_layer(params,batch,kwargs=self.apply_fun_args)

        # # symmetrize
        # phase_psi = jnp.sum(Im_z.reshape(self.reduce_shape,order='C'), axis=[1,])
        # # 
        # phase_psi = jnp.sum(phase_psi.reshape(self.output_shape), axis=[1,])

        log_psi, phase_psi = self.apply_layer_dyn(params,batch,kwargs=self.apply_fun_args_dyn)

        return phase_psi


    # @jax.partial(jit, static_argnums=(0,3))
    # def evaluate_phase(self, params, batch, apply_fun_args):
    #     return self._evaluate_phase(params, batch, apply_fun_args)

    

    @cython.boundscheck(False)
    def sample(self,
                    int N_MC_points,
                    int thermalization_time,
                    double acceptance_ratio,
                    #
                    np.int8_t[::1] spin_states,
                    basis_type[::1] ket_states,
                    np.float64_t[::1] log_mod_kets,
                    #
                    # basis_type[::1] s0_vec,
                    bool thermal
                    ):

        cdef int N_accepted=0
        cdef int chain_n=0
        cdef vector[int] N_MC_proposals=np.zeros(self.N_MC_chains)
        # reduce MC points per chain
        cdef int n_MC_points=N_MC_points//self.N_MC_chains
        cdef int n_MC_points_leftover=N_MC_points%self.N_MC_chains
        
        #print(N_MC_points, n_MC_points,n_MC_points_leftover)
        #
        cdef int auto_correlation_time = 0.35/np.max([0.05, acceptance_ratio])*self.N_sites

        with nogil:

            for chain_n in prange(self.N_MC_chains,schedule='static', num_threads=self.N_MC_chains):
               
                N_accepted+=self._MC_core(
                                           n_MC_points,
                                           &N_MC_proposals[chain_n],
                                           thermalization_time,
                                           auto_correlation_time,
                                           #
                                           &spin_states[chain_n*n_MC_points*self.N_features],
                                           &ket_states[chain_n*n_MC_points],
                                           &log_mod_kets[chain_n*n_MC_points],
                                           # &s0_vec[0],
                                           # 
                                           &self.spinstate_s[chain_n*self.N_spinconfigelmts],
                                           &self.spinstate_t[chain_n*self.N_spinconfigelmts],
                                           #
                                           chain_n,
                                           self.RNGs[chain_n],
                                           thermal,
                                           self.sf_vec[chain_n]
                                        )

            # take care of left-over, continue chain_n = 0
            if n_MC_points_leftover>0:
 
                N_accepted+=self._MC_core(
                                           n_MC_points_leftover,
                                           &N_MC_proposals[0],
                                           0, # thermalization time
                                           auto_correlation_time,
                                           #
                                           &spin_states[self.N_MC_chains*n_MC_points*self.N_features],
                                           &ket_states[self.N_MC_chains*n_MC_points],
                                           &log_mod_kets[self.N_MC_chains*n_MC_points],
                                           # &s0_vec[0],
                                           # 
                                           &self.spinstate_s[0],
                                           &self.spinstate_t[0],
                                           #
                                           0, # chain_n
                                           self.RNGs[0],
                                           True, # thermal
                                           ket_states[n_MC_points-1]
                                        )

            
        # print(self.MPI_rank, np.array(ket_states))
        # print(N_MC_proposals,N_accepted, ket_states.shape)
        # exit()

        # print('thread seeds', self.MPI_rank, self.thread_seeds)
        # print('sfvec', np.array(self.sf_vec))

        return N_accepted, np.sum(N_MC_proposals);
   
    
    @cython.boundscheck(False)
    cdef int _MC_core(self, int N_MC_points,
                            int* N_MC_proposals,
                            int thermalization_time,
                            int auto_correlation_time,
                            #
                            np.int8_t * spin_states,
                            basis_type * ket_states,
                            double * log_mod_kets,
                            # basis_type[] s0_vec,
                            #
                            np.int8_t * spinstate_s,
                            np.int8_t * spinstate_t,
                            #
                            int chain_n,
                            mt19937& rng,
                            bool thermal,
                            basis_type s0
        ) nogil:           
        
        cdef int i=0, k=0; # counters
        cdef int N_accepted=0;
        cdef int thread_id = threadid();
        

        cdef double mod_psi_s=0.0, mod_psi_t=0.0;
        cdef double eps; # acceptance random float

        cdef basis_type t;
        cdef np.uint16_t _i,_j;

        
        # draw random initial state
        # cdef basis_type ordinal = self.rand_int_ordinal(rng);
        # cdef basis_type s=magnetized_int(self.N_sites//2, self.N_sites, ordinal)


        cdef basis_type s;
        cdef basis_type one=1;
        cdef int l;
            
        if thermal:
            s=s0;
        else:
            s=(one<<(self.N_sites//2))-one;
            for l in range(4*self.N_sites):
                t=s;
                while(t==s):
                    _i = self.random_int(rng)
                    _j = self.random_int(rng)
                    t = swap_bits(s,_i,_j);
                s=t;

           
        # store initial state for reproducibility
        self.s0_vec[chain_n] = s;
            
        
        # compute initial spin config and its amplitude value
        self.spin_config(self.N_sites,s,&spinstate_s[0]);


        # set omp barrier
        OMP_BARRIER_PRAGMA()
        # evaluate DNN on GPU
        if thread_id==0:
            with gil:
                self.log_psi_s=self.evaluate_log(self.params, self.spinstate_s_py);
        # set barrier
        OMP_BARRIER_PRAGMA()
        mod_psi_s=exp(self.log_psi_s[chain_n]);


        while(k < N_MC_points):
            
            # propose a new state until a nontrivial configuration is drawn
            t=s;
            while(t==s):
                _i = self.random_int(rng)
                _j = self.random_int(rng)
                
                t = swap_bits(s,_i,_j);
            

            self.spin_config(self.N_sites,t,&spinstate_t[0]);


            # set omp barrier
            OMP_BARRIER_PRAGMA()
            # evaluate DNN on GPU
            if thread_id==0:
                with gil:
                    self.log_psi_t=self.evaluate_log(self.params, self.spinstate_t_py);
            # set barrier
            OMP_BARRIER_PRAGMA()
            mod_psi_t=exp(self.log_psi_t[chain_n]);


            # MC accept/reject step
            eps = self.random_float(rng);
            if( (eps*mod_psi_s*mod_psi_s <= mod_psi_t*mod_psi_t) & (500*mod_psi_s*mod_psi_s >= mod_psi_t*mod_psi_t) ): # accept
                
                s = t;
                self.log_psi_s[chain_n] = self.log_psi_t[chain_n];
                mod_psi_s = mod_psi_t;
                # set spin configs
                for i in range(self.N_features):
                    spinstate_s[i] = spinstate_t[i];
                
                N_accepted+=1;
    

            if( (N_MC_proposals[0] > thermalization_time) & ((N_MC_proposals[0] % auto_correlation_time) == 0) ):
                
                for i in range(self.N_features):
                    spin_states[k*self.N_sites*self.N_symm + i] = spinstate_s[i];

                ket_states[k] = s;
                log_mod_kets[k]=self.log_psi_s[chain_n];

                k+=1;
                
            N_MC_proposals[0]+=1;


        # record last configuration
        self.sf_vec[chain_n]=ket_states[k-1]


        return N_accepted;

        

