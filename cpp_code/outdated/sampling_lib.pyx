# distutils: language=c++
# cython: language_level=2
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: profile=True

from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, random, device_put
#from jax.experimental.stax import GeneralConv #relu, BatchNorm



cimport cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libc.stdlib cimport rand, srand, RAND_MAX
#from libcpp cimport bool

from cython.parallel cimport prange, threadid, parallel
cimport openmp

seed=1
np.random.seed(seed)
np.random.RandomState(seed)
rng = random.PRNGKey(seed)


from DNN_architectures import *



##############################################


L=4
ctypedef np.uint16_t basis_type

#L=6    
#ctypedef np.uint64_t basis_type



N_sites=L*L
ctypedef basis_type (*rep_func_type)(const int,basis_type,np.float64_t*) nogil;
ctypedef void (*func_type)(const int,basis_type,np.float64_t*) nogil;



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


cdef extern from "sample_4x4.h":
    
    T swap_bits[T](const T, int, int) nogil
    
    int update_offdiag[I](const int, const char[], const int[], const double, const int, const int, const I[], I[], np.float64_t [], np.int32_t[], double[], I (*)(const int,I,np.float64_t*) ) nogil  
    
    void update_diag[I](const int, const char[], const int[], const double, const int, const I[], double[] ) nogil
    
    void offdiag_sum(int,int[],double[],double[],np.uint32_t[],double[],const double[],const double[]) nogil

    void int_to_spinstate[T](const int,T ,np.float64_t []) nogil
    void int_to_spinstate_conv[T](const int,T ,np.float64_t []) nogil

    T rep_int_to_spinstate[T](const int,T ,np.float64_t []) nogil
    T rep_int_to_spinstate_conv[T](const int,T ,np.float64_t []) nogil



@cython.boundscheck(False)
def integer_to_spinstate(basis_type[:] states,np.float64_t[::1] out, int N_features, object NN_type='DNN'):
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
def update_diag_ME(np.ndarray ket,double[::1] M,object opstr,int[::1] indx,double J):
    cdef char[::1] c_opstr = bytearray(opstr,"utf-8")
    cdef int n_op = indx.shape[0]
    cdef int Ns = ket.shape[0]
    cdef void * ket_ptr = np.PyArray_GETPTR1(ket,0)

    with nogil:
        update_diag(n_op,&c_opstr[0],&indx[0],J,Ns,<basis_type*>ket_ptr,&M[0])


@cython.boundscheck(False)
def update_offdiag_ME(np.ndarray ket,basis_type[:] bra, np.float64_t[:,:] spin_bra,np.int32_t[:] ket_indx,double[::1] M,object opstr,int[::1] indx,double J,int N_symm, object NN_type='DNN'):
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
                const double[::1] psi_bras,
                const double[::1] phase_psi_bras):
    
    cdef int Ns = n_per_term.shape[0]
    
    with nogil:
        offdiag_sum(Ns,&n_per_term[0],&Eloc_cos[0],&Eloc_sin[0],&ket_indx[0],&MEs[0],&psi_bras[0],&phase_psi_bras[0])
        







###########################


@cython.boundscheck(False)
cdef class Neural_Net:

    cdef object apply_layer
    cdef object W_real, W_imag
    cdef object params
    cdef object input_shape, reduce_shape, output_shape, out_chan, strides, filter_shape 
    cdef object NN_type

    cdef np.ndarray shapes, dims 
    cdef int N_varl_params, N_symm, N_sites

    cdef object evaluate_phase, evaluate_log

    cdef np.float64_t[::1] spinstate_s, spinstate_t
    cdef object spinstate_s_py, spinstate_t_py

    cdef vector[double] mod_psi_s, mod_psi_t
    
    cdef vector[np.uint16_t] sites

    cdef int N_MC_chains, N_spinconfigelmts

    cdef vector[unsigned int] thread_seeds

    cdef func_type spin_config
        

    def __init__(self,shapes,N_MC_chains,NN_type='DNN',seed=0):

        W_shape=shapes[0]
        self.N_sites=L*L
 
        # fix seed
        srand(seed)

        # order important
        self._init_NN(W_shape,NN_type)
        self._init_evaluate()
        self._init_variables(N_MC_chains)

    

    def _init_NN(self,W_shape,NN_type):

        self.NN_type=NN_type

        if NN_type=='DNN':
            self.N_symm=L*L*2*2*2 # no Z symmetry
            init_params, self.apply_layer = GeneralDeep(W_shape, ignore_b=True)
            input_shape=None

            # tuple to reshape output before symmetrization
            self.input_shape = (-1,self.N_sites)
            self.reduce_shape = (-1,self.N_symm,W_shape[0]) 
            self.output_shape = (-1,W_shape[0]) 

        elif NN_type=='CNN':
            self.N_symm=2*2*2 # no Z, Tx, Ty symmetry

            dimension_numbers=('NCHW', 'OIHW', 'NCHW') # default
            self.out_chan=1
            self.filter_shape=(2,2)
            self.strides=(1,1)

            input_shape=np.array((1,1,L,L),dtype=np.int) # NCHW input format
            # add padding dimensions
            input_shape+=np.array((0,0)+self.strides)

            init_params, self.apply_layer = GeneralConv(dimension_numbers, self.out_chan, self.filter_shape, strides=self.strides, padding='VALID', ignore_b=True)
                
            # tuple to reshape output before symmetrization
            self.input_shape = (-1,1,L,L)
            self.reduce_shape = (-1,self.N_symm,self.out_chan,L,L)
            self.output_shape = (-1,self.out_chan*L*L)
        else:
            raise ValueError("unsupported string for variable NN_type.") 
        

        # initialize parameters
        W_real, = init_params(rng,input_shape)[1]
        W_imag, = init_params(rng,input_shape)[1]

        # W_real2, = init_params(rng,input_shape)[1]
        # W_imag2, = init_params(rng,input_shape)[1]

        self.params=[W_real, W_imag, ]
        #self.params=[W_real, W_imag, W_real2, W_imag2, ]


        self.shapes=np.array([W.shape for W in self.params])
        self.dims=np.array([np.prod(shape) for shape in self.shapes])
        self.N_varl_params=self.dims.sum()

        

    def _init_evaluate(self):

        # self.evaluate_mod  =self._evaluate_mod
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
        self.N_spinconfigelmts=self.N_symm*self.N_sites

        self.spinstate_s=np.zeros(self.N_MC_chains*self.N_symm*self.N_sites,dtype=np.float64)
        self.spinstate_t=np.zeros(self.N_MC_chains*self.N_symm*self.N_sites,dtype=np.float64)

        # access data in device array; transfer memory from numpy to jax
        if self.NN_type=='DNN':
            spinstate_shape=[self.N_MC_chains*self.N_symm,self.N_sites]
        elif self.NN_type=='CNN':
            spinstate_shape=[self.N_MC_chains*self.N_symm,1,L,L]
        self.spinstate_s_py=np.asarray(self.spinstate_s).reshape(spinstate_shape)
        self.spinstate_t_py=np.asarray(self.spinstate_t).reshape(spinstate_shape)

        self.mod_psi_s=np.zeros(self.N_MC_chains,dtype=np.float64)
        self.mod_psi_t=np.zeros(self.N_MC_chains,dtype=np.float64)

        ###############################################################

        self.sites=np.arange(self.N_sites,dtype=np.uint16)

        self.thread_seeds=np.zeros(self.N_MC_chains,dtype=np.uint)
        for i in range(self.N_MC_chains):
            self.thread_seeds[i]=(rand()%RAND_MAX)

  
    property N_sites:
        def __get__(self):
            return self.N_sites

    property N_symm:
        def __get__(self):
            return self.N_symm

    property NN_type:
        def __get__(self):
            return self.NN_type

    property shapes:
        def __get__(self):
            return self.shapes

    property dims:
        def __get__(self):
            return self.dims

    property N_varl_params:
        def __get__(self):
            return self.N_varl_params 

    property params:
        def __get__(self):
            return self.params

    property evaluate_phase:
        def __get__(self):
            return self.evaluate_phase

    property evaluate_log:
        def __get__(self):
            return self.evaluate_log



    @cython.boundscheck(False)
    def update_params(self,params):
        self.params=params



    @cython.boundscheck(False)
    def complex_inputlayer(self, params, batch):
        # apply dense layer
        Re_Ws = self.apply_layer(params[0], batch)
        Im_Ws = self.apply_layer(params[1], batch)
        return Re_Ws, Im_Ws

    @cython.boundscheck(False)
    def complex_deeplayer(self, params, Re_z, Im_z):
        # apply dense layer
        Re_Ws = self.apply_layer(params[0], Re_z) - self.apply_layer(params[1], Im_z)
        Im_Ws = self.apply_layer(params[1], Re_z) + self.apply_layer(params[0], Im_z)
        return Re_Ws, Im_Ws
        

    
    @cython.boundscheck(False)
    def evaluate(self, params, batch):

        # reshaping required inside evaluate func because of per-sample gradients
        batch=batch.reshape(self.input_shape)

        # apply dense layer
        Re_Ws, Im_Ws = self.complex_inputlayer(params[0:2],batch)
        # apply logcosh nonlinearity
        Re_z, Im_z = logcosh_cpx(Re_Ws, Im_Ws)

        # # apply dense layer
        # Re_Ws, Im_Ws = self.complex_deeplayer(params[2:4],Re_z,Im_z)
        # # apply logcosh nonlinearity
        # Re_z, Im_z = logcosh_cpx(Re_Ws, Im_Ws)


        # symmetrize
        log_psi   = jnp.sum(Re_z.reshape(self.reduce_shape,order='C'), axis=[1,])
        phase_psi = jnp.sum(Im_z.reshape(self.reduce_shape,order='C'), axis=[1,])
        # 
        log_psi   = jnp.sum(  log_psi.reshape(self.output_shape), axis=[1,])
        phase_psi = jnp.sum(phase_psi.reshape(self.output_shape), axis=[1,])
        
        return log_psi, phase_psi


    @cython.boundscheck(False)
    cpdef object _evaluate_log(self, object params, object batch):

        # reshaping required inside evaluate func because of per-sample gradients
        batch=batch.reshape(self.input_shape)

        # apply dense layer
        Re_Ws, Im_Ws = self.complex_inputlayer(params[0:2],batch)
        # apply logcosh nonlinearity
        Re_z = logcosh_real(Re_Ws, Im_Ws)

        # # apply dense layer
        # Re_Ws, Im_Ws = self.complex_inputlayer(params[0:2],batch)
        # # apply logcosh nonlinearity
        # Re_z, Im_z = logcosh_cpx(Re_Ws, Im_Ws)


        # # apply dense layer
        # Re_Ws, Im_Ws = self.complex_deeplayer(params[2:4],Re_z,Im_z)
        # # apply logcosh nonlinearity
        # Re_z = logcosh_real(Re_Ws, Im_Ws) 
        

        # symmetrize
        log_psi   = jnp.sum(Re_z.reshape(self.reduce_shape,order='C'), axis=[1,])
        # 
        log_psi   = jnp.sum(  log_psi.reshape(self.output_shape), axis=[1,])
        
        return log_psi


    @cython.boundscheck(False)
    cpdef object _evaluate_phase(self, object params, object batch):

        # reshaping required inside evaluate func because of per-sample gradients
        batch=batch.reshape(self.input_shape)

        # apply dense layer
        Re_Ws, Im_Ws = self.complex_inputlayer(params[0:2],batch)
        # apply logcosh nonlinearity
        Im_z = logcosh_imag(Re_Ws, Im_Ws)

        # # apply dense layer
        # Re_Ws, Im_Ws = self.complex_inputlayer(params[0:2],batch)
        # # apply logcosh nonlinearity
        # Re_z, Im_z = logcosh_cpx(Re_Ws, Im_Ws)


        # # apply dense layer
        # Re_Ws, Im_Ws = self.complex_deeplayer(params[2:4],Re_z,Im_z)
        # # apply logcosh nonlinearity
        # Im_z = logcosh_imag(Re_Ws, Im_Ws)  
 

        # symmetrize
        phase_psi = jnp.sum(Im_z.reshape(self.reduce_shape,order='C'), axis=[1,])
        # 
        phase_psi = jnp.sum(phase_psi.reshape(self.output_shape), axis=[1,])
        
        return phase_psi





    

    @cython.boundscheck(False)
    def sample(self,
                    int N_MC_points,
                    int thermalization_time,
                    int auto_correlation_time,
                    #
                    np.float64_t[::1] spin_states,
                    basis_type[:] ket_states,
                    np.float64_t[:] mod_kets

                    ):

        cdef int N_accepted=0, n_accepted=0
        cdef int chain_n
        # reduce MC points per chain
        cdef int n_MC_points=N_MC_points//self.N_MC_chains
        
    
        with nogil:

            for chain_n in prange(self.N_MC_chains,schedule='static', num_threads=self.N_MC_chains):
            
                N_accepted+=self._MC_core(
                                           n_MC_points,
                                           thermalization_time,
                                           auto_correlation_time,
                                           #
                                           &spin_states[chain_n*n_MC_points*self.N_symm*self.N_sites],
                                           &ket_states[chain_n*n_MC_points],
                                           &mod_kets[chain_n*n_MC_points],
                                           # 
                                           &self.spinstate_s[chain_n*self.N_spinconfigelmts],
                                           &self.spinstate_t[chain_n*self.N_spinconfigelmts],
                                           #
                                           chain_n
                                        )

            
        return N_accepted;
   
    
    @cython.boundscheck(False)
    cdef int _MC_core(self, int N_MC_points,
                            int thermalization_time,
                            int auto_correlation_time,
                            #
                            np.float64_t * spin_states,
                            basis_type * ket_states,
                            double * mod_kets,
                            #
                            np.float64_t * spinstate_s,
                            np.float64_t * spinstate_t,
                            #
                            int chain_n
        ) nogil:           
        
        cdef int i=0, j=0, k=0; # counters
        cdef int N_accepted=0;
        cdef int thread_id = threadid();
        cdef unsigned int * thread_seed = &self.thread_seeds[thread_id];

        cdef double eps; # acceptance random float

        cdef basis_type t;
        
        cdef np.uint16_t _i,_j;
        cdef basis_type s=0, one=1;


        # draw random initial state
        s=(one<<(self.N_sites//2))-one;
        t=s;
        while(t==s):
            _i = rand_r(thread_seed)%self.N_sites 
            _j = rand_r(thread_seed)%self.N_sites 
            t = swap_bits(s,_i,_j);
        s=t;
       
        # with gil:
        #     print(thread_id, s)

        
        # compute initial spin config and its amplitude value
        self.spin_config(self.N_sites,s,&spinstate_s[0]);


        # set omp barrier
        OMP_BARRIER_PRAGMA()
        # evaluate DNN on GPU
        if thread_id==0:
            with gil:
                self.mod_psi_s=jnp.exp(self.evaluate_log(self.params, self.spinstate_s_py));
                
        # set barrier
        OMP_BARRIER_PRAGMA()


     
        while(k < N_MC_points):

            t=s;
     
            # propose a new state until a nontrivial configuration is drawn
            while(t==s):
                _i = rand_r(thread_seed)%self.N_sites 
                _j = rand_r(thread_seed)%self.N_sites 
                t = swap_bits(s,_i,_j);
            

            self.spin_config(self.N_sites,t,&spinstate_t[0]);


            # set omp barrier
            OMP_BARRIER_PRAGMA()
            # evaluate DNN on GPU
            if thread_id==0:
                with gil:
                    self.mod_psi_t=jnp.exp(self.evaluate_log(self.params,self.spinstate_t_py));
            # set barrier
            OMP_BARRIER_PRAGMA()


            # MC step
            eps = float(rand_r(thread_seed))/float(RAND_MAX);
            if(eps*self.mod_psi_s[chain_n]*self.mod_psi_s[chain_n] <= self.mod_psi_t[chain_n]*self.mod_psi_t[chain_n]): # accept
                s = t;
                self.mod_psi_s[chain_n] = self.mod_psi_t[chain_n];
                # set spin configs
                for i in range(self.N_symm*self.N_sites):
                    spinstate_s[i] = spinstate_t[i];
                N_accepted+=1;
    

            if( (j > thermalization_time) & (j % auto_correlation_time) == 0):
                
                for i in range(self.N_symm*self.N_sites):
                    spin_states[k*self.N_sites*self.N_symm + i] = spinstate_s[i];

                ket_states[k] = s;
                mod_kets[k]=self.mod_psi_s[chain_n];


                k+=1;
                
            j+=1;



        return N_accepted;

        



'''

@jit
def melu(x):
    return jnp.where(jnp.abs(x)>1.0, jnp.abs(x)-0.5, 0.5*x**2)


def create_NN(shape):

    init_value_W=1E-1 
    init_value_b=1E-1
    
    W_fc_base = random.uniform(rng,shape=shape[0], minval=-init_value_W, maxval=+init_value_W)
    
    W_fc_log_psi = random.uniform(rng,shape=shape[1], minval=-init_value_W,   maxval=+init_value_W)
    W_fc_phase   = random.uniform(rng,shape=shape[2], minval=-init_value_W, maxval=+init_value_W)

    b_fc_log_psi = random.uniform(rng,shape=(shape[1][1],), minval=-init_value_b, maxval=+init_value_b)
    b_fc_phase   = random.uniform(rng,shape=(shape[2][1],), minval=-init_value_b, maxval=+init_value_b)

    
    architecture=[W_fc_base, W_fc_log_psi, W_fc_phase, b_fc_log_psi, b_fc_phase]

    return architecture



@jit
def evaluate_NN(params,batch):

    ### common layer
    Ws = jnp.einsum('ij,...lj->...il',params[0], batch)
    # nonlinearity
    #a_fc_base = jnp.log(jnp.cosh(Ws))
    a_fc_base = melu(Ws) 
    #a_fc_base = relu(Ws)
    # symmetrize
    a_fc_base = jnp.sum(a_fc_base, axis=[-1])

    
    ### log_psi head
    a_fc_log_psi = jnp.dot(a_fc_base, params[1]) + params[3]
    #log_psi = jnp.log(jnp.cosh(a_fc_log_psi))
    log_psi = melu(a_fc_log_psi) 
    #log_psi = relu(a_fc_log_psi)

    ### phase head
    a_fc_phase =jnp.dot(a_fc_base, params[2]) + params[4]
    #phase_psi = jnp.log(jnp.cosh(a_fc_phase))
    phase_psi = melu(a_fc_phase) 
    #phase_psi = relu(a_fc_phase)

    
    log_psi = jnp.sum(log_psi, axis=[1])#/log_psi.shape[0]
    phase_psi = jnp.sum(phase_psi, axis=[1])#/phase_psi.shape[0]    


    return log_psi, phase_psi  #

'''
