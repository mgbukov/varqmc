# distutils: language=c++
# cython: language_level=2
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: profile=False

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
from libcpp cimport bool
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.math cimport exp #, sin, cos, acos, sqrt, fabs, M_PI, floor, ceil

from cython.parallel cimport prange, threadid, parallel
cimport openmp

from rng_wrapper cimport *
from local_sampling cimport *


from .NN_log import *
from .NN_phase import *
from .NN_cpx import *

#from DNN_architectures_cpx import *
# from DNN_architectures_real import *
# from DNN_architectures_common import *

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





cdef extern from "sample.h":
        
    int choose_n_k(int, int) nogil
    
    T swap_bits[T](const T, int, int) nogil
    T magnetized_int[T](int, int, T) nogil

    int update_offdiag_exact[I](const int, const char[], const int[], const double, const int, const I[], I[], np.int32_t[], double[] ) nogil  
    
    int update_offdiag[I](const int, const char[], const int[], const double, const int, const int, const I[], I[], np.int8_t [], np.int32_t[], double[], I (*)(const int,I,np.int8_t*) ) nogil  
    void update_diag[I](const int, const char[], const int[], const double, const int, const I[], double[] ) nogil
    
    void offdiag_sum(int,int[],double[],double[],np.uint32_t[],double[],const double[],const double[],const double[]) nogil

    void int_to_spinstate[T,J](const int,T ,J []) nogil
    void int_to_spinstate_conv[T,J](const int,T ,J []) nogil

    void int_to_spinstate_conv2[T,J](const int,T ,J [], const int) nogil

    T cyclicity[T](const int,T) nogil

    T rep_int_to_spinstate[T,J](const int,T ,J []) nogil
    T rep_int_to_spinstate_conv[T,J](const int,T ,J []) nogil

    basis_type ref_state(const basis_type) nogil



@cython.boundscheck(False)
def swap_spins(basis_type s, int i, int j):
    cdef basis_type t;

    with nogil:
        t = swap_bits(s,i,j)

    return t


@cython.boundscheck(False)
def integer_to_spinstate(basis_type[:] states,np.int8_t[::1] out, int N_features, object NN_type='DNN'): 
#def integer_to_spinstate(basis_type[:] states,np.ndarray out, int N_features, object NN_type='DNN'): 
    cdef int i;
    cdef int Ns=states.shape[0]
    cdef int Nsites=N_sites

#    cdef np.npy_int8 * out_ptr = <np.npy_int8*> np.PyArray_GETPTR1(out,0)

    cdef func_type spin_config

    if NN_type=='DNN':
        spin_config=<func_type>int_to_spinstate
    elif NN_type=='CNN':
        spin_config=<func_type>int_to_spinstate_conv
    else:
        raise ValueError("unsupported string for variable NN_type.")


    # with nogil:
    #     for i in range (Ns):
    #         int_to_spinstate_conv2(Nsites, states[i], out_ptr, i*N_features)



    # with nogil:
    #     for i in range (Ns*N_features):
    #         #out[i]=1
    #         out_ptr[i]=1

    with nogil:
        for i in range (Ns):
            spin_config(Nsites,states[i],&out[i*N_features])



@cython.boundscheck(False)
def representative(basis_type[:] states,basis_type[:] out,):
    cdef int i;
    cdef int Ns=states.shape[0]
    with nogil:
        for i in range (Ns):
            out[i]=ref_state(states[i]) 


@cython.boundscheck(False)
def integer_cyclicity(basis_type[:] states,np.npy_uint32[::1] cycl):
    cdef int i;
    cdef int Ns=states.shape[0];
    cdef int Nsites=N_sites;
    with nogil:
        for i in range (Ns):
            cycl[i]=cyclicity(Nsites,states[i])


@cython.boundscheck(False)
def update_offdiag_ME_exact(np.ndarray ket, basis_type[:] bra, np.int32_t[:] ket_indx,double[::1] M,object opstr,int[::1] indx,double J, ):
    cdef char[::1] c_opstr = bytearray(opstr,"utf-8")
    cdef int n_op = indx.shape[0]
    cdef int Ns = ket.shape[0]
    cdef int l
    
    cdef void * ket_ptr = np.PyArray_GETPTR1(ket,0)

    with nogil:
        l=update_offdiag_exact(n_op, &c_opstr[0], &indx[0], J, Ns, <basis_type*>ket_ptr, &bra[0] ,&ket_indx[0] ,&M[0])

    return l


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
def update_diag_ME(np.ndarray ket,double[::1] M,object opstr,int[::1] indx,double J):
    cdef char[::1] c_opstr = bytearray(opstr,"utf-8")
    cdef int n_op = indx.shape[0]
    cdef int Ns = ket.shape[0]
    cdef void * ket_ptr = np.PyArray_GETPTR1(ket,0)

    with nogil:
        update_diag(n_op,&c_opstr[0],&indx[0],J,Ns,<basis_type*>ket_ptr,&M[0])



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




def _compute_layers(rng, comm, NN_architecture, input_shape, output_shape, reduce_shape):

    NN_architecture_dyn = NN_architecture.copy()
    for key in NN_architecture.keys():
        if 'batch_norm' in key:
            print('check dynamic layers')
            exit()
            NN_architecture_dyn[key]=BatchNorm_cpx_dyn(axis=(0,))

    # create NN
    init_params, apply_layer, apply_fun_args = serial(*NN_architecture.values())
    _, apply_layer_dyn, apply_fun_args_dyn = serial(*NN_architecture_dyn.values())

    _, params = init_params(rng,input_shape)

    _init_apply_fun_args(rng, comm, NN_architecture, apply_fun_args,apply_fun_args_dyn, input_shape, output_shape, reduce_shape)

    return params, apply_layer, apply_fun_args, apply_layer_dyn, apply_fun_args_dyn




def _init_apply_fun_args(rng, comm, NN_architecture, apply_fun_args,apply_fun_args_dyn, input_shape, output_shape, reduce_shape):
    
    layers_type=list(NN_architecture.keys())
    init_funs, apply_funs = zip(*NN_architecture.values())

    for j, (init_fun, layer_type) in enumerate(zip(init_funs, layers_type)):        
        rng, layer_rng = random.split(rng)
        input_shape, _ = init_fun(layer_rng, input_shape)

        if 'batch_norm' in layer_type:
            mean, std_mat_inv = init_batchnorm_cpx_params(input_shape)
            
            # an update in the parameters of apply_fun_args_dyn UPDATES directly the params of apply_fun_args
            apply_fun_args[j]=dict(mean=mean, std_mat_inv=std_mat_inv, )        
            apply_fun_args_dyn[j]=dict(fixpoint_iter=False, mean=mean, std_mat_inv=std_mat_inv, comm=comm, )
    
        # if 'reg' in layer_type:
        #     D=dict(reduce_shape=reduce_shape, output_shape=output_shape)
        #     apply_fun_args[j]=D
        #     apply_fun_args_dyn[j]=D






@cython.boundscheck(False)
cdef class Log_Net:

    cdef bool semi_exact
    cdef object evaluate, evaluate_log, evaluate_phase, evaluate_sampling
    cdef object spin_config_inds

    cdef object NN_Tree
    cdef int N_varl_params

    #cdef object NN_architecture, NN_architecture_dyn
    cdef object apply_layer, apply_layer_dyn
    cdef object apply_fun_args, apply_fun_args_dyn
    
    #cdef object W_real, W_imag
    cdef object params, params_update
    cdef object input_shape, reduce_shape, output_shape, out_chan, strides, filter_shape 
    cdef object NN_type, NN_dtype
    cdef object comm

    cdef int MPI_rank, seed
    cdef vector[unsigned int] thread_seeds

    cdef int N_symm, N_sites, N_features
    cdef int N_MC_chains, N_spinconfigelmts

   
    cdef vector[double] log_psi_s, log_psi_t
    cdef vector[basis_type] spinconfig_s, spinconfig_t
    cdef vector[nb_func_type] neighbors
    cdef double prop_threshold
    
    cdef np.int8_t[::1] spinstate_s, spinstate_t
    cdef object spinstate_s_py, spinstate_t_py
    cdef func_type spin_config
    cdef basis_type[::1] sf_vec, s0_vec

    
    cdef vector[np.uint16_t] sites

    


    cdef uniform_real_distribution[double] random_float
    cdef uniform_int_distribution[int] random_int, random_int8, rand_int_ordinal
    cdef vector[mt19937] RNGs # hold a C++ instance
        

    # exact data
    cdef basis_type[::1] ints_ket_exact
    cdef object log_psi_exact, phase_psi_exact


    def __init__(self,comm,shapes,N_MC_chains,NN_type='DNN',NN_dtype='cpx',seed=0,prop_threshold=0.5):

        self.N_sites=_L*_L
        self.prop_threshold=prop_threshold

        self.NN_dtype=NN_dtype
        self.semi_exact=False


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
        self._init_MC_variables(N_MC_chains)
        self._init_MC_data()


    def load_exact_data(self,ints_ket_exact,log_psi_exact=None,phase_psi_exact=None):
        #self.ints_ket_exact=ints_ket_exact
        self.log_psi_exact=log_psi_exact
        self.phase_psi_exact=phase_psi_exact
        

        self.evaluate_sampling = self._evaluate_log_exact
        if self.NN_dtype=='real':
            self.evaluate=self._evaluate_log_exact

        elif self.NN_dtype=='cpx':
            self.evaluate=self._evaluate_cpx_exact
            self.evaluate_phase = self._evaluate_phase_exact
            self.evaluate_log=self._evaluate_log_exact


        # construct dictionary-table
        self.spin_config_inds=dict()
        self.spin_config_inds[0]=0
        for i, s in enumerate(ints_ket_exact):
            self.spin_config_inds[s]=i

        self.semi_exact=True



    def _init_NN(self,rng,shapes,NN_type,NN_dtype,):

        self.NN_type=NN_type
        self.NN_dtype=NN_dtype

        shape_last_layer = shapes['layer_5']
        

        if NN_type=='DNN':
           
            self.N_symm=_L*_L*2*2*2 # no Z symmetry

            # determine shape variables
            input_shape=(1,self.N_sites)
            self.input_shape=(-1,self.N_sites)

            reduce_shape = (-1,self.N_symm,shape_last_layer[1],1)
           
            output_shape = (-1,shape_last_layer[1],)
            
            # define variance of uniform distr for weights and biases
            scale=0.8
        
            # define DNN
            if self.NN_dtype=='real':
                NN_arch = NN_log_arch('DNN_1', shapes, input_shape, reduce_shape, output_shape, scale) 
            elif self.NN_dtype=='cpx':
                NN_arch = NN_cpx_arch('DNN_2', shapes, input_shape, reduce_shape, output_shape, scale) 
           
            
        elif NN_type=='CNN':
                
            self.N_symm=2*2*2 # no Z, Tx, Ty symmetry

            dim_nums=('NCHW', 'OIHW', 'NCHW') # default 

            # determine shape variables
            input_shape   =(+1,1,_L,_L)
            self.input_shape=(-1,1,_L,_L)

            # CNN symemtrization
            #reduce_shape =   (-1,self.N_symm,shape_last_layer[1],self.N_sites)
            # DNN symemtrization
            reduce_shape = (-1,self.N_symm*self.N_sites, shape_last_layer[1], 1)

            output_shape = (-1,shape_last_layer[1],)

            # define variance of uniform distr for weights and biases
            scale=0.8

            # define CNN
            if self.NN_dtype=='real':
                NN_arch = NN_log_arch('CNN_mixed_5', shapes, input_shape, reduce_shape, output_shape, scale)
                #NN_arch = NN_log_arch('CNN_as_dnn_2', shapes, input_shape, reduce_shape, output_shape, scale)
            elif self.NN_dtype=='cpx':   
                NN_arch = NN_cpx_arch('CNN_as_dnn_2', shapes, input_shape, reduce_shape, output_shape, scale)

        else:
            raise ValueError("unsupported string for variable for NN_type.") 


        # create a copy of the NN architecture to update the batch norm mean and variance dynamically
        self.params,   self.apply_layer,  self.apply_fun_args,   self.apply_layer_dyn,   self.apply_fun_args_dyn  = _compute_layers(rng, self.comm, NN_arch, input_shape, output_shape, reduce_shape)
        self.params_update=np.zeros_like(self.params)
    

        self.NN_Tree = NN_Tree(self.params)
        self.N_varl_params  =NN_Tree(self.params).N_varl_params
       
        self.N_features=self.N_sites*self.N_symm


        self.params_update=0.0*self.NN_Tree.ravel(self.params)._value
       


    def _init_evaluate(self):

        # self.evaluate  =self._evaluate
        # self.evaluate_sampling  =self._evaluate_sampling
        
        # define network evaluation on GPU
        self.evaluate=jit(self._evaluate)

        if self.NN_dtype=='real':
            self.evaluate_log      = None
            self.evaluate_phase    = None
            self.evaluate_sampling = jit(self._evaluate_sampling)
        
        elif self.NN_dtype=='cpx':
            self.evaluate_log      = jit(self._evaluate_log)
            self.evaluate_phase    = jit(self._evaluate_phase)
            self.evaluate_sampling = jit(self._evaluate_sampling_log)

        
        if self.NN_type=='DNN':
            self.spin_config=<func_type>int_to_spinstate
        elif self.NN_type=='CNN':
            self.spin_config=<func_type>int_to_spinstate_conv


    def _init_MC_variables(self,N_MC_chains):

        self.N_MC_chains=N_MC_chains
        self.N_spinconfigelmts=self.N_features

        self.spinstate_s=np.zeros(self.N_MC_chains*self.N_features,dtype=np.int8)
        self.spinstate_t=np.zeros(self.N_MC_chains*self.N_features,dtype=np.int8)

        self.spinconfig_s=np.zeros(self.N_MC_chains,dtype=basis_type_py)
        self.spinconfig_t=np.zeros(self.N_MC_chains,dtype=basis_type_py)

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
        self.random_int8 = uniform_int_distribution[int](0,7) # 8 = 4 nn + 4nnn on square lattice
        self.rand_int_ordinal = uniform_int_distribution[int](0,choose_n_k(self.N_sites, self.N_sites//2))

        
        self.neighbors.resize(8)

        self.neighbors[0]=neighbors_func_0
        self.neighbors[1]=neighbors_func_1
        self.neighbors[2]=neighbors_func_2
        self.neighbors[3]=neighbors_func_3
        self.neighbors[4]=neighbors_func_4
        self.neighbors[5]=neighbors_func_5
        self.neighbors[6]=neighbors_func_6
        self.neighbors[7]=neighbors_func_7


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

    property N_varl_params:
        def __get__(self):
            return self.N_varl_params

    property params:
        def __get__(self):
            return self.params
        def __set__(self,value):
            self.params=value

    property params_update:
        def __get__(self):
            return self.params_update
        def __set__(self,value):
            self.params_update=value

    property evaluate:
        def __get__(self):
            return self.evaluate

    property evaluate_log:
            def __get__(self):
                    return self.evaluate_log

    property evaluate_phase:
            def __get__(self):
                    return self.evaluate_phase

    property s0_vec:
        def __get__(self):
            return self.s0_vec

    property sf_vec:
        def __get__(self):
            return self.sf_vec

    property semi_exact:
        def __get__(self):
            return self.semi_exact


    @cython.boundscheck(False)
    def update_params(self,params):
        self.params=params


    @cython.boundscheck(False)
    cpdef object _evaluate(self, object params, object batch):
        # reshaping required inside evaluate func because of per-sample gradients
        #batch=batch.reshape(self.input_shape)
        log_psi = self.apply_layer(params,batch,kwargs=self.apply_fun_args)
        return log_psi

    @cython.boundscheck(False)
    cpdef object _evaluate_log(self, object params, object batch):
        # reshaping required inside evaluate func because of per-sample gradients
        log_psi, phase_psi = self.apply_layer(params,batch,kwargs=self.apply_fun_args)
        return log_psi

    @cython.boundscheck(False)
    cpdef object _evaluate_phase(self, object params, object batch):
        # reshaping required inside evaluate func because of per-sample gradients
        log_psi, phase_psi = self.apply_layer(params,batch,kwargs=self.apply_fun_args)
        return phase_psi


    @cython.boundscheck(False)
    cpdef object _evaluate_log_exact(self, object params, object batch):
        #s_inds=np.searchsorted(self.ints_ket_exact,batch,)
        s_inds=[self.spin_config_inds[s] for s in batch]
        return self.log_psi_exact[s_inds]

    @cython.boundscheck(False)
    cpdef object _evaluate_phase_exact(self, object params, object batch):
        #s_inds=np.searchsorted(self.ints_ket_exact,batch,)
        s_inds=[self.spin_config_inds[s] for s in batch]
        return self.phase_psi_exact[s_inds]

    @cython.boundscheck(False)
    cpdef object _evaluate_cpx_exact(self, object params, object batch):
        #s_inds=np.searchsorted(self.ints_ket_exact,batch,)
        s_inds=[self.spin_config_inds[s] for s in batch]
        return self.log_psi_exact[s_inds], self.phase_psi_exact[s_inds]




    @cython.boundscheck(False)
    cpdef object _evaluate_sampling(self, object params, object batch):
        log_psi = self.apply_layer(params,batch,kwargs=self.apply_fun_args)
        return log_psi

    @cython.boundscheck(False)
    cpdef object _evaluate_sampling_log(self, object params, object batch):
        log_psi, phase_psi  = self.apply_layer(params,batch,kwargs=self.apply_fun_args)
        return log_psi




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
        cdef int auto_correlation_time = 0.5/np.max([0.05, acceptance_ratio])*self.N_sites

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

        cdef int x,y;
        

        cdef double mod_psi_s=0.0, mod_psi_t=0.0;
        cdef double eps, delta; # acceptance random float

        cdef basis_type t;
        cdef np.uint16_t _i,_j,_k;

        
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
                if self.semi_exact:
                    self.spinconfig_s[chain_n] = ref_state(s);
                    self.log_psi_s=self.evaluate_sampling(self.params, self.spinconfig_s);
                else:
                    self.log_psi_s=self.evaluate_sampling(self.params, self.spinstate_s_py);
        # set barrier
        OMP_BARRIER_PRAGMA()
        mod_psi_s=exp(self.log_psi_s[chain_n]);


        while(k < N_MC_points):
            
            # propose a new state until a nontrivial configuration is drawn
            t=s;
            while(t==s):
                _i = self.random_int(rng)

                # drwa random number to decide whethr to look for local or nonlocal update
                delta = self.random_float(rng);
                if delta>self.prop_threshold: # local update
                    x=_i%_L;
                    y=_i//_L;

                    _k=self.random_int8(rng)
                    _j = self.neighbors[_k](x,y,_L)
                else: # any update
                    _j = self.random_int(rng)
                
                t = swap_bits(s,_i,_j);
            

            self.spin_config(self.N_sites,t,&spinstate_t[0]);


            # set omp barrier
            OMP_BARRIER_PRAGMA()
            # evaluate DNN on GPU
            if thread_id==0:
                with gil:
                    if self.semi_exact:
                        self.spinconfig_t[chain_n] = ref_state(t);
                        self.log_psi_t=self.evaluate_sampling(self.params, self.spinconfig_t);
                    else:
                        self.log_psi_t=self.evaluate_sampling(self.params, self.spinstate_t_py);
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

                for i in range(self.N_MC_chains):
                    self.spinconfig_s[i] = self.spinconfig_t[i];

                N_accepted+=1;
    

            if( (N_MC_proposals[0] > thermalization_time) & ((N_MC_proposals[0] % auto_correlation_time) == 0) ):
                
                for i in range(self.N_features):
                    spin_states[k*self.N_sites*self.N_symm + i] = spinstate_s[i];

                if self.semi_exact:
                    ket_states[k] = self.spinconfig_s[chain_n];
                else:
                    ket_states[k] = s;

                log_mod_kets[k]=self.log_psi_s[chain_n];

                k+=1;
                
            N_MC_proposals[0]+=1;


        # record last configuration
        self.sf_vec[chain_n]=ket_states[k-1]


        return N_accepted;

        


        

@cython.boundscheck(False)
cdef class Phase_Net:

    cdef bool semi_exact

    cdef object evaluate
    cdef object spin_config_inds

    cdef object NN_Tree
    cdef int N_varl_params

    #cdef object NN_architecture, NN_architecture_dyn
    cdef object apply_layer, apply_layer_dyn
    cdef object apply_fun_args, apply_fun_args_dyn
    
    #cdef object W_real, W_imag
    cdef object params, params_update
    cdef object input_shape, reduce_shape, output_shape, out_chan, strides, filter_shape 
    cdef object NN_type, NN_dtype
    cdef object comm

    cdef int MPI_rank, seed
    
    cdef int N_symm, N_sites, N_features

    # exact data
    cdef basis_type[::1] ints_ket_exact
    cdef object phase_psi_exact
    
        

    def __init__(self,comm,shapes,N_MC_chains,NN_type='DNN',NN_dtype='cpx',seed=0,prop_threshold=0.5):

        self.N_sites=_L*_L
        #self.prop_threshold=prop_threshold
        self.semi_exact=False


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
        

    def load_exact_data(self,ints_ket_exact,log_psi_exact=None,phase_psi_exact=None):
        #self.ints_ket_exact=ints_ket_exact
        self.phase_psi_exact=phase_psi_exact
        
        self.evaluate=self._evaluate_phase_exact

        self.spin_config_inds=dict()
        self.spin_config_inds[0]=0
        for i, s in enumerate(ints_ket_exact):
            self.spin_config_inds[s]=i
        
        self.semi_exact=True


    def _init_NN(self,rng,shapes,NN_type,NN_dtype,):

        self.NN_type=NN_type
        self.NN_dtype=NN_dtype

        #shapes=shapes[0]
        
        shape_last_layer = shapes['layer_3']
        

        if NN_type=='DNN':
           
            self.N_symm=_L*_L*2*2*2 # no Z symmetry


            # determine shape variables
            input_shape=(1,self.N_sites)
            self.input_shape=(-1,self.N_sites)

            reduce_shape = (-1,self.N_symm,shape_last_layer[1],1)
            output_shape = (-1,shape_last_layer[1],)

            # define variance of uniform distr for weights and biases
            scale=1.0
          
            # define DNN
            NN_arch = NN_phase_arch('DNN_2', shapes, input_shape, reduce_shape, output_shape, scale)
        
        elif NN_type=='CNN':
                
            self.N_symm=2*2*2 # no Z, Tx, Ty symmetry

            dim_nums=('NCHW', 'OIHW', 'NCHW') # default


            # determine shape variables
            input_shape   =(+1,1,_L,_L)
            self.input_shape=(-1,1,_L,_L)

            # CNN symemtrization
            #reduce_shape =   (-1,self.N_symm,shape_last_layer[1],self.N_sites)
            # DNN symemtrization
            reduce_shape = (-1,self.N_symm*self.N_sites, shape_last_layer[1], 1)
          
            output_shape = (-1,shape_last_layer[1],)

            # define variance of uniform distr for weights and biases
            scale=1.0 # 4.0

            # define CNN
            NN_arch = NN_phase_arch('CNN_mixed_3', shapes, input_shape, reduce_shape, output_shape, scale)   

            
        else:
            raise ValueError("unsupported string for variable for NN_type.") 


        # create a copy of the NN architecture to update the batch norm mean and variance dynamically
        self.params,   self.apply_layer,  self.apply_fun_args,   self.apply_layer_dyn,   self.apply_fun_args_dyn  = _compute_layers(rng, self.comm, NN_arch  , input_shape, output_shape, reduce_shape)      
        self.params_update=np.zeros_like(self.params)
    


        self.NN_Tree = NN_Tree(self.params)
        self.N_varl_params  =NN_Tree(self.params).N_varl_params
       
        self.N_features=self.N_sites*self.N_symm


        self.params_update=0.0*self.NN_Tree.ravel(self.params)._value
       


    def _init_evaluate(self):

        #self.evaluate_log  =self._evaluate_log
        #self.evaluate_phase=self._evaluate_phase

        # define network evaluation on GPU
        self.evaluate  =jit(self._evaluate)
    


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

    property N_varl_params:
        def __get__(self):
            return self.N_varl_params

    property params:
        def __get__(self):
            return self.params
        def __set__(self,value):
            self.params=value

    property params_update:
        def __get__(self):
            return self.params_update
        def __set__(self,value):
            self.params_update=value

    property evaluate:
        def __get__(self):
            return self.evaluate

    property semi_exact:
        def __get__(self):
            return self.semi_exact



    @cython.boundscheck(False)
    def update_params(self,params):
        self.params=params


    @cython.boundscheck(False)
    cpdef object _evaluate(self, object params, object batch):

        # reshaping required inside evaluate func because of per-sample gradients
        #batch=batch.reshape(self.input_shape)

        phase_psi = self.apply_layer(params,batch,kwargs=self.apply_fun_args)
        
        return phase_psi


    @cython.boundscheck(False)
    cpdef object _evaluate_phase_exact(self, object params, object batch):
        #s_inds=np.searchsorted(self.ints_ket_exact,batch,)
        s_inds=[self.spin_config_inds[s] for s in batch]
        return self.phase_psi_exact[s_inds]


