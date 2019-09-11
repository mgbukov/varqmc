# cython: language_level=2
## cython: profile=True
# distutils: language=c++

from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, vmap, random, ops, partial
from jax.experimental.stax import relu, BatchNorm

cimport cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
#from libcpp cimport bool

seed=1
np.random.seed(seed)
np.random.RandomState(seed)
rng = random.PRNGKey(seed)


##############################################


L=4
N_symms=128 
ctypedef np.uint16_t basis_type 

#L=6    
#ctypedef np.uint64_t basis_type

N_sites=L*L


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
                            np.int8_t [],
                            basis_type [],
                            basis_type [],
                            double [],
                            double [],
                            #
                            const double [],
                            const double [],
                            const int,

                        ) nogil

    T swap_bits[T](const T, int, int) nogil
    
    int update_offdiag[I](const int, const char[], const int[], const double, const int, const I[], I[], np.int8_t [], np.uint32_t[], double[] ) nogil  
    
    void update_diag[I](const int, const char[], const int[], const double, const int, const I[], double[] ) nogil
    
    void offdiag_sum(int,int[],double[],double[],np.uint32_t[],double[],const double[],const double[]) nogil

    T int_to_spinstate[T](const int,T ,np.int8_t []) nogil


    double evaluate_mod(
                                np.int8_t[],
                                const double[],
                                const double[],
                                const int,
                                const int
                                
                                )

    double evaluate_phase(
                                np.int8_t[],
                                const double[],
                                const double[],
                                const int,
                                const int
                                
                                )

    void evaluate_rbm[I](
                    I [],
                    np.int8_t [],
                    double [],
                    double [],
                    const double [],
                    const double [],
                    const int ,
                    const int ,
                    const int 
                    
                    ) nogil


@cython.boundscheck(False)
def c_evaluate_NN(basis_type[::1] states,
                    np.int8_t[::1] spinstate,
                    np.float64_t[::1] mod_psi,
                    np.float64_t[::1] phase_psi,
                    const np.float64_t[:,::1] W_fc_real,
                    const np.float64_t[:,::1] W_fc_imag,
                    int N_sites,
                    int N_fc,
                    int Ns,
                    ):
    with nogil:
        evaluate_rbm(&states[0],&spinstate[0],&mod_psi[0],&phase_psi[0],&W_fc_real[0,0],&W_fc_imag[0,0],N_sites,N_fc,Ns)


@cython.boundscheck(False)
def c_evaluate_mod(
                    np.int8_t[::1] spinstate,
                    const np.float64_t[:,::1] W_fc_real,
                    const np.float64_t[:,::1] W_fc_imag,
                    int N_sites,
                    int N_fc):
     return evaluate_mod(&spinstate[0], &W_fc_real[0,0], &W_fc_imag[0,0],N_sites,N_fc)
    
@cython.boundscheck(False)
def c_evaluate_phase(
                    np.int8_t[::1] spinstate,
                    const np.float64_t[:,::1] W_fc_real,
                    const np.float64_t[:,::1] W_fc_imag,
                    int N_sites,
                    int N_fc):
    return evaluate_phase(&spinstate[0], &W_fc_real[0,0], &W_fc_imag[0,0],N_sites,N_fc)
  


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
def update_offdiag_ME(np.ndarray ket,basis_type[:] bra, np.int8_t[:,:] spin_bra,np.uint32_t[:] ket_indx,double[::1] M,object opstr,int[::1] indx,double J):
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
                    np.int8_t[::1] rep_spin_states,
                    basis_type[:] ket_states,
                    basis_type[:] rep_ket_states,
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
                       &rep_spin_states[0],
                       &ket_states[0],
                       &rep_ket_states[0],
                       &mod_kets[0],
                       &phase_kets[0],
                       #
                       &W_fc_real[0,0],
                       &W_fc_imag[0,0],
                       N_fc

            )

        return N_accepted;




###########################



cdef class Neural_Net:

    cdef object W_fc_real, W_fc_imag 
    cdef object params

    cdef np.ndarray shapes, dims 
    cdef int N_varl_params

    cdef object evaluate_phase, evaluate_mod

    cdef np.ndarray spinstate_s, spinstate_t
    cdef void * spinstate_s_ptr
    cdef void * spinstate_t_ptr
    


    def __init__(self,shape):

        init_value_Re=1E-1#1E-3 
        init_value_Im=1E-1#1E-1 
        
        self.W_fc_real = random.uniform(rng,shape=shape, minval=-init_value_Re, maxval=+init_value_Re)
        self.W_fc_imag = random.uniform(rng,shape=shape, minval=-init_value_Im, maxval=+init_value_Im)

        self.params=[self.W_fc_real, self.W_fc_imag,]


        self.shapes=np.array([W.shape for W in self.params])
        self.dims=np.array([np.prod(shape) for shape in self.shapes])
        self.N_varl_params=self.dims.sum()


        #self.evaluate_mod  =self._evaluate_mod
        #self.evaluate_phase=self._evaluate_phase

        self.evaluate_mod  =jit(self._evaluate_mod)
        self.evaluate_phase=jit(self._evaluate_phase)


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

        #print(params[0]-self.W_fc_real)

     
        Re_Ws = jnp.einsum('ij,...lj->il...',params[0], batch)
        Im_Ws = jnp.einsum('ij,...lj->il...',params[1], batch)

        Re = jnp.cos(Im_Ws)*jnp.cosh(Re_Ws)
        Im = jnp.sin(Im_Ws)*jnp.sinh(Re_Ws)

        #a_fc_real = tf.log( tf.sqrt( (tf.cos(Im_Ws)*tf.cosh(Re_Ws))**2 + (tf.sin(Im_Ws)*tf.sinh(Re_Ws))**2 )  )
        #a_fc_imag = tf.atan( tf.tan(Im_Ws)*tf.tanh(Re_Ws) )
        a_fc_real = 0.5*jnp.log(Re**2 + Im**2)
        a_fc_imag = jnp.arctan2(Im,Re)

        log_psi = jnp.sum(a_fc_real,axis=[0,1])
        phase_psi = jnp.sum(a_fc_imag,axis=[0,1])


        return log_psi, phase_psi


    cpdef object _evaluate_mod(self, object batch):
    
        Re_Ws = jnp.einsum('ij,...lj->il...',self.W_fc_real, batch)
        Im_Ws = jnp.einsum('ij,...lj->il...',self.W_fc_imag, batch)

        Re = jnp.cos(Im_Ws)*jnp.cosh(Re_Ws)
        Im = jnp.sin(Im_Ws)*jnp.sinh(Re_Ws)

        a_fc_real = 0.5*jnp.log(Re**2 + Im**2)
  
        log_psi = jnp.sum(a_fc_real,axis=[0,1])
         
        return jnp.exp(log_psi)     

    # cpdef so it can be jitted
    cpdef object _evaluate_phase(self, object batch):

        Re_Ws = jnp.einsum('ij,...lj->il...',self.W_fc_real, batch)
        Im_Ws = jnp.einsum('ij,...lj->il...',self.W_fc_imag, batch)

        Re = jnp.cos(Im_Ws)*jnp.cosh(Re_Ws)
        Im = jnp.sin(Im_Ws)*jnp.sinh(Re_Ws)

        a_fc_imag = jnp.arctan2(Im,Re)

        phase_psi = jnp.sum(a_fc_imag,axis=[0,1])

        return phase_psi



    

    @cython.boundscheck(False)
    def sample(self,
                    int N_MC_points,
                    int thermalization_time,
                    int auto_correlation_time,
                    #
                    np.int8_t[::1] spin_states,
                    np.int8_t[::1] rep_spin_states,
                    basis_type[:] ket_states,
                    basis_type[:] rep_ket_states,
                    np.float64_t[:] mod_kets,
                    np.float64_t[:] phase_kets

                    ):

        cdef basis_type s_c
        cdef int N_accepted

        # initial configuration
        s=[0 for _ in range(N_sites//2)] + [1 for _ in range(N_sites//2)]
        np.random.shuffle(s)
        s = np.array(s)
        s=''.join([str(j) for j in s])
        s=int(s,2)

        s_c=s

        #
        self.spinstate_s=np.zeros(N_sites,N_symms,dtype=np.int8)
        self.spinstate_t=np.zeros(N_sites,N_symms,dtype=np.int8)

        self.spinstate_s_ptr = np.PyArray_GETPTR1(self.spinstate_s,0)
        self.spinstate_t_ptr = np.PyArray_GETPTR1(self.spinstate_s,0)



        #with nogil:
        N_accepted=self._MC_core(
                   s_c,
                   N_MC_points,
                   thermalization_time,
                   auto_correlation_time,
                   #
                   &spin_states[0],
                   &rep_spin_states[0],
                   &ket_states[0],
                   &rep_ket_states[0],
                   &mod_kets[0]

        )


        phase_kets[:]=self.evaluate_phase(spin_states)


        return N_accepted;
   
    #'''
    cdef int _MC_core(self, basis_type s,
                            int N_MC_points,
                            int thermalization_time,
                            int auto_correlation_time,
                            #
                            np.int8_t * spin_states,
                            np.int8_t * rep_spin_states,
                            basis_type * ket_states,
                            basis_type * rep_ket_states,
                            double * mod_kets
        ):           
        
        cdef int i=0;
        cdef int j=0;
        cdef int k=0;
        cdef int N_accepted=0;

        cdef double eps;

        cdef basis_type t, s_rep, t_rep;
        cdef double mod_psi_s, mod_psi_t

        cdef np.uint16_t _i,_j;
       

        s_rep=int_to_spinstate(N_sites,s,<np.int8_t*>self.spinstate_s_ptr);
        mod_psi_s=self.evaluate_mod(self.spinstate_s);
                        
        

        while(k < N_MC_points):

            t=s;
            self.spinstate_t=self.spinstate_s;

            # propose a new state until a nontrivial configuration is drawn
            while(t==s):
                _i = np.random.randint(N_sites)
                _j = np.random.randint(N_sites)
                t = swap_bits(s,_i,_j);
            

            t_rep=int_to_spinstate(N_sites,t,<np.int8_t*>self.spinstate_t_ptr);
            
                
            # compute amplitude
            mod_psi_t=self.evaluate_mod(self.spinstate_t);


            eps = np.random.uniform(0,1);
            if(eps*mod_psi_s*mod_psi_s <= mod_psi_t*mod_psi_t): # accept
                s = t;
                s_rep = t_rep;
                mod_psi_s = mod_psi_t;
                self.spinstate_s = self.spinstate_t;
                N_accepted+=1;
             
            

            if( (j > thermalization_time) & (j % auto_correlation_time) == 0):
                            
                for i in range(N_sites): 
                    spin_states[k*N_sites*N_symms + i] = self.spinstate_s[i];


                ket_states[k] = s;
                rep_ket_states[k] = s_rep;
                
                mod_kets[k]=mod_psi_s;


                k+=1;
                
            j+=1;

        

        return N_accepted;
    #'''
        

   



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
