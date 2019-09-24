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
#from jax.experimental.stax import relu, BatchNorm

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


##############################################


L=4
ctypedef np.uint16_t basis_type 

#L=6    
#ctypedef np.uint64_t basis_type

N_sites=L*L



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
    cdef cppclass Monte_Carlo[I]:
        Monte_Carlo() except +

        unsigned int seed;
        # int world_size, world_rank;

        # Seed the random number generator 
        void set_seed(unsigned int) nogil

        int build_ED_dicts(int, int, double) nogil

        void evaluate_mod_dict(I [], double [], int) nogil

        void evaluate_phase_dict(I [], double [], int) nogil


        # void mpi_init() nogil

        # void mpi_close() nogil

        # void mpi_allgather[T](T*, int, T*, int) nogil


        int sample_DNN(
                            I ,
                            int ,
                            int ,
                            int ,
                            #
                            np.int8_t [],
                            basis_type [],
                            double [],
                            double [],
                            #
                            const double [],
                            const double [],
                            const int,

                        ) nogil

    T swap_bits[T](const T, int, int) nogil
    
    int update_offdiag[I](const int, const char[], const int[], const double, const int, const I[], I[], np.int8_t [], np.int32_t[], double[] ) nogil  
    
    void update_diag[I](const int, const char[], const int[], const double, const int, const I[], double[] ) nogil
    
    void offdiag_sum(int,int[],double[],double[],np.uint32_t[],double[],const double[],const double[]) nogil

    void int_to_spinstate[T](const int,T ,np.int8_t []) nogil

    T rep_int_to_spinstate[T](const int,T ,np.int8_t []) nogil



@cython.boundscheck(False)
def integer_to_spinstate(basis_type[:] states,np.int8_t[::1] out, int N_features):
    cdef int i;
    cdef int Ns=states.shape[0]
    cdef int Nsites=N_sites

    with nogil:
        for i in range (Ns):
            int_to_spinstate(Nsites,states[i],&out[i*N_features])




@cython.boundscheck(False)
def update_diag_ME(np.ndarray ket,double[::1] M,object opstr,int[::1] indx,double J):
    cdef char[::1] c_opstr = bytearray(opstr,"utf-8")
    cdef int n_op = indx.shape[0]
    cdef int Ns = ket.shape[0]
    cdef void * ket_ptr = np.PyArray_GETPTR1(ket,0)
    
    with nogil:
        update_diag(n_op,&c_opstr[0],&indx[0],J,Ns,<basis_type*>ket_ptr,&M[0])


@cython.boundscheck(False)
def update_offdiag_ME(np.ndarray ket,basis_type[:] bra, np.int8_t[:,:] spin_bra,np.int32_t[:] ket_indx,double[::1] M,object opstr,int[::1] indx,double J):
    cdef char[::1] c_opstr = bytearray(opstr,"utf-8")
    cdef int n_op = indx.shape[0]
    cdef int Ns = ket.shape[0]
    cdef int l
    
    cdef void * ket_ptr = np.PyArray_GETPTR1(ket,0)
    #cdef void * spin_bra_ptr = np.PyArray_GETPTR1(spin_bra,[0,0])

    with nogil:
        l=update_offdiag(n_op,&c_opstr[0],&indx[0],J,Ns,<basis_type*>ket_ptr,&bra[0],&spin_bra[0,0],&ket_indx[0],&M[0])

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
        


@cython.boundscheck(False)
cdef class cpp_Monte_Carlo:
    
    cdef Monte_Carlo[basis_type] * MC_c # hold a C++ instance
    
    def __cinit__(self):
        self.MC_c = new Monte_Carlo[basis_type]()

    #def __init__(self):
    #    self.seed=self.MC_c.seed_value
    
    def __dealloc__(self):
        del self.MC_c




    property seed:
        def __get__(self):
            return self.MC_c.seed
        #def __set__(self, int seed):
        #    self.MC_c.seed = seed


    def set_seed(self, unsigned int u):
        self.MC_c.set_seed(u)


    @cython.boundscheck(False)
    def build_ED_dicts(self,int sign, int L, double J2):
        return self.MC_c.build_ED_dicts(sign, L, J2)

    @cython.boundscheck(False)
    def evaluate_mod_dict(self,basis_type[:] keys, np.float64_t[:] values, int Ns):
        self.MC_c.evaluate_mod_dict(&keys[0], &values[0], Ns)

    @cython.boundscheck(False)
    def evaluate_phase_dict(self,basis_type[:] keys, np.float64_t[:] values, int Ns):
        self.MC_c.evaluate_phase_dict(&keys[0], &values[0], Ns)



    @cython.boundscheck(False)
    def sample_DNN(self,
                    int N_MC_points,
                    int thermalization_time,
                    int auto_correlation_time,
                    #
                    np.int8_t[::1] spin_states,
                    basis_type[:] ket_states,
                    np.float64_t[:] mod_kets,
                    np.float64_t[:] phase_kets,
                    #
                    const np.float64_t[:,::1] W_fc_real,
                    const np.float64_t[:,::1] W_fc_imag,
                    int N_fc,

                    ):

        cdef basis_type s_c

        s=[0 for _ in range(N_sites//2)] + [1 for _ in range(N_sites//2)]
        np.random.shuffle(s)
        s = np.array(s)
        s=''.join([str(j) for j in s])
        s=int(s,2)

        s_c=s

        with nogil:
            N_accepted=self.MC_c.sample_DNN(
                       s_c,
                       N_MC_points,
                       thermalization_time,
                       auto_correlation_time,
                       #
                       &spin_states[0],
                       &ket_states[0],
                       &mod_kets[0],
                       &phase_kets[0],
                       #
                       &W_fc_real[0,0],
                       &W_fc_imag[0,0],
                       N_fc

            )

        return N_accepted;




###########################


cdef int myrandom(int i) nogil:
    return rand()%i;




cdef class Neural_Net:

    cdef object W_fc_real, W_fc_imag 
    cdef object params

    cdef np.ndarray shapes, dims 
    cdef int N_varl_params, N_symm, N_sites

    cdef object evaluate_phase, evaluate_mod

    cdef np.int8_t[::1] spinstate_s, spinstate_t
    cdef object spinstate_s_py, spinstate_t_py

    cdef vector[double] mod_psi_s, mod_psi_t
    
    cdef vector[np.uint16_t] sites

    cdef int N_MC_chains, N_spinconfigelmts

    cdef vector[unsigned int] thread_seeds
        

    def __init__(self,shape,N_MC_chains,seed=0):

        # OMP ars
        openmp.omp_set_num_threads(N_MC_chains)

        # fix seed
        srand(seed)

        init_value_Re=1E-1#1E-3 
        init_value_Im=1E-1#1E-1 
        
        self.W_fc_real = random.uniform(rng,shape=shape, minval=-init_value_Re, maxval=+init_value_Re)
        self.W_fc_imag = random.uniform(rng,shape=shape, minval=-init_value_Im, maxval=+init_value_Im)

        self.params=[self.W_fc_real, self.W_fc_imag,]


        self.shapes=np.array([W.shape for W in self.params])
        self.dims=np.array([np.prod(shape) for shape in self.shapes])
        self.N_varl_params=self.dims.sum()

        self.N_symm=128
        self.N_sites=L*L


        ################################################


        # self.evaluate_mod  =self._evaluate_mod
        # self.evaluate_phase=self._evaluate_phase

        # define network evaluation on GPU
        self.evaluate_mod  =jit(self._evaluate_mod)
        self.evaluate_phase=jit(self._evaluate_phase)


        ################################################
        self.N_MC_chains=N_MC_chains
        self.N_spinconfigelmts=self.N_symm*self.N_sites

        self.spinstate_s=np.zeros(self.N_MC_chains*self.N_symm*self.N_sites,dtype=np.int8)
        self.spinstate_t=np.zeros(self.N_MC_chains*self.N_symm*self.N_sites,dtype=np.int8)

        # access data in device array; transfer memory from numpy to jax
        self.spinstate_s_py=np.asarray(self.spinstate_s).reshape([self.N_MC_chains*self.N_symm,self.N_sites])
        self.spinstate_t_py=np.asarray(self.spinstate_t).reshape([self.N_MC_chains*self.N_symm,self.N_sites])

        self.mod_psi_s=np.zeros(self.N_MC_chains,dtype=np.int8)
        self.mod_psi_t=np.zeros(self.N_MC_chains,dtype=np.int8)

        ###############################################################

        self.sites=np.arange(self.N_sites,dtype=np.uint16)

        self.thread_seeds=np.zeros(self.N_MC_chains,dtype=np.uint)
        for i in range(self.N_MC_chains):
            self.thread_seeds[i]=(rand()%RAND_MAX)

  
        

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

    property evaluate_mod:
        def __get__(self):
            return self.evaluate_mod

    



    def update_params(self,params):
        self.W_fc_real=params[0]
        self.W_fc_imag=params[1]
        self.params=params


    def evaluate(self, params, batch):

        # Cosh[a + I b] = Cos[b] Cosh[a] + I Sin[b] Sinh[a]
        # Re_Ws = jnp.einsum('ij,...lj->...li',self.W_fc_real, batch)
        # Im_Ws = jnp.einsum('ij,...lj->...li',self.W_fc_imag, batch)

        batch=batch.reshape(-1,self.N_sites)
        
        Re_Ws = jnp.einsum('ij,lj->il',params[0], batch)
        Im_Ws = jnp.einsum('ij,lj->il',params[1], batch)

        Re = jnp.cos(Im_Ws)*jnp.cosh(Re_Ws)
        Im = jnp.sin(Im_Ws)*jnp.sinh(Re_Ws)

        
        #a_fc_real = tf.log( tf.sqrt( (tf.cos(Im_Ws)*tf.cosh(Re_Ws))**2 + (tf.sin(Im_Ws)*tf.sinh(Re_Ws))**2 )  )
        #a_fc_imag = tf.atan( tf.tan(Im_Ws)*tf.tanh(Re_Ws) )
      
        a_fc_real = 0.5*jnp.log(Re**2+Im**2).reshape((self.shapes[0][0],self.N_symm,-1),order='F')#.squeeze()
        a_fc_imag = jnp.arctan2(Im,Re).reshape((self.shapes[1][0],self.N_symm,-1),order='F')#.squeeze()

        log_psi = jnp.sum(a_fc_real,axis=[0,1])
        phase_psi = jnp.sum(a_fc_imag,axis=[0,1])

        return log_psi, phase_psi


    cpdef object _evaluate_mod(self, object batch):

        # Re_Ws = jnp.einsum('ij,...lj->il...',self.W_fc_real, batch)
        # Im_Ws = jnp.einsum('ij,...lj->il...',self.W_fc_imag, batch)

        Re_Ws = jnp.einsum('ij,lj->il',self.W_fc_real, batch)
        Im_Ws = jnp.einsum('ij,lj->il',self.W_fc_imag, batch)

        Re = jnp.cos(Im_Ws)*jnp.cosh(Re_Ws)
        Im = jnp.sin(Im_Ws)*jnp.sinh(Re_Ws)

        a_fc_real = 0.5*jnp.log(Re**2 + Im**2).reshape(self.shapes[0][0],self.N_symm,-1,order='F')
  
        log_psi = jnp.sum(a_fc_real,axis=[0,1])

        return jnp.exp(log_psi)     

    # cpdef so it can be jitted
    cpdef object _evaluate_phase(self, object batch):

        #batch=batch.reshape(-1,self.N_sites)

        # Re_Ws = jnp.einsum('ij,...lj->il...',self.W_fc_real, batch)
        # Im_Ws = jnp.einsum('ij,...lj->il...',self.W_fc_imag, batch)

        Re_Ws = jnp.einsum('ij,lj->il',self.W_fc_real, batch)
        Im_Ws = jnp.einsum('ij,lj->il',self.W_fc_imag, batch)

        Re = jnp.cos(Im_Ws)*jnp.cosh(Re_Ws)
        Im = jnp.sin(Im_Ws)*jnp.sinh(Re_Ws)

        a_fc_imag = jnp.arctan2(Im,Re).reshape(self.shapes[1][0],self.N_symm,-1,order='F')

        phase_psi = jnp.sum(a_fc_imag,axis=[0,1])

        return phase_psi



    

    @cython.boundscheck(False)
    def sample(self,
                    int N_MC_points,
                    int thermalization_time,
                    int auto_correlation_time,
                    #
                    np.int8_t[::1] spin_states,
                    basis_type[:] ket_states,
                    np.float64_t[:] mod_kets

                    ):

        cdef int N_accepted=0, n_accepted=0
        cdef int chain_n

        #cdef int n_threads=0


        # reduce MC points per chain
        cdef int n_MC_points=N_MC_points//self.N_MC_chains
        #print("{0:d} MC points per MC chain.".format(n_MC_points) )

    
        with nogil:

            for chain_n in prange(self.N_MC_chains,schedule='static', num_threads=self.N_MC_chains):
            #START_OMP_PARALLEL_PRAGMA()
            #for chain_n in range(self.N_MC_chains):

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

            

        #print(N_accepted)
        #exit()

        return N_accepted;
   
    # OpenMPI: has headers 
    # cython: from header define function from cpp MPI that I wanna use; nogil etc
    # cpp version better than python version

    @cython.boundscheck(False)
    cdef int _MC_core(self, int N_MC_points,
                            int thermalization_time,
                            int auto_correlation_time,
                            #
                            np.int8_t * spin_states,
                            basis_type * ket_states,
                            double * mod_kets,
                            #
                            np.int8_t * spinstate_s,
                            np.int8_t * spinstate_t,
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
        int_to_spinstate(self.N_sites,s,&spinstate_s[0]);


        # set omp barrier
        OMP_BARRIER_PRAGMA()
        # evaluate DNN on GPU
        if thread_id==0:
            with gil:
                self.mod_psi_s=self.evaluate_mod(self.spinstate_s_py);
        # set barrier
        OMP_BARRIER_PRAGMA()


     
        while(k < N_MC_points):

            t=s;
     
            # propose a new state until a nontrivial configuration is drawn
            while(t==s):
                _i = rand_r(thread_seed)%self.N_sites 
                _j = rand_r(thread_seed)%self.N_sites 
                t = swap_bits(s,_i,_j);
            

            int_to_spinstate(self.N_sites,t,&spinstate_t[0]);


            # set omp barrier
            OMP_BARRIER_PRAGMA()
            # evaluate DNN on GPU
            if thread_id==0:
                with gil:
                    self.mod_psi_t=self.evaluate_mod(self.spinstate_t_py);
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
