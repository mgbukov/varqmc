# cython: language_level=2
## cython: profile=True
# distutils: language=c++

cimport cython
import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
#from libcpp cimport bool

L=4
ctypedef np.uint16_t basis_type 

#L=6    
#ctypedef np.uint64_t basis_type

N_sites=L*L


cdef extern from "sample_4x4.h":
    cdef cppclass Monte_Carlo[I]:
        Monte_Carlo() except +

        unsigned int seed;
        int world_size, world_rank;

        # Seed the random number generator 
        void set_seed(unsigned int) nogil

        int build_ED_dicts(int, int, double) nogil

        void evaluate_mod_dict(I [], double [], int) nogil

        void evaluate_phase_dict(I [], double [], int) nogil


        void mpi_init() nogil

        void mpi_close() nogil

        void mpi_allgather[T](T*, int, T*, int) nogil


        int sample_DNN(
                            I ,
                            int ,
                            int ,
                            int ,
                            #
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


    
    int update_offdiag[I](const int, const char[], const int[], const double, const int, const I[], I[], np.uint32_t [], np.int8_t [], np.uint32_t[], double[] ) nogil  
    
    void update_diag[I](const int, const char[], const int[], const double, const int, const I[], double[] ) nogil
    
    void offdiag_sum(int,int[],double[],double[],np.uint32_t[],double[],const double[],const double[]) nogil

    int int_to_spinstate[T](const int,T ,np.int8_t []) nogil


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
def integer_to_spinstate(basis_type[:] states,np.int8_t[::1] out, np.uint32_t[::1] cyclicity, int N_features):
    cdef int i;
    cdef int Ns=states.shape[0]
    cdef int Nsites=N_sites

    with nogil:
        for i in range (Ns):
            cyclicity[i]=int_to_spinstate(Nsites,states[i],&out[i*N_features])



@cython.boundscheck(False)
def update_diag_ME(np.ndarray ket,double[::1] M,object opstr,int[::1] indx,double J):
    cdef char[::1] c_opstr = bytearray(opstr,"utf-8")
    cdef int n_op = indx.shape[0]
    cdef int Ns = ket.shape[0]
    cdef void * ket_ptr = np.PyArray_GETPTR1(ket,0)
    
    with nogil:
        update_diag(n_op,&c_opstr[0],&indx[0],J,Ns,<basis_type*>ket_ptr,&M[0])


@cython.boundscheck(False)
def update_offdiag_ME(np.ndarray ket,basis_type[:] bra,np.uint32_t[::1] cyclicities_bra_holder, np.int8_t[:,:] spin_bra,np.uint32_t[:] ket_indx,double[::1] M,object opstr,int[::1] indx,double J):
    cdef char[::1] c_opstr = bytearray(opstr,"utf-8")
    cdef int n_op = indx.shape[0]
    cdef int Ns = ket.shape[0]
    cdef int l
    
    cdef void * ket_ptr = np.PyArray_GETPTR1(ket,0)
    #cdef void * spin_bra_ptr = np.PyArray_GETPTR1(spin_bra,[0,0])

    with nogil:
        l=update_offdiag(n_op,&c_opstr[0],&indx[0],J,Ns,<basis_type*>ket_ptr,&bra[0],&cyclicities_bra_holder[0],&spin_bra[0,0],&ket_indx[0],&M[0])

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

    property world_size:
        def __get__(self):
            return self.MC_c.world_size

    property world_rank:
        def __get__(self):
            return self.MC_c.world_rank

    

    def mpi_init(self):
        self.MC_c.mpi_init()

    def mpi_close(self):
        self.MC_c.mpi_close()

    @cython.boundscheck(False)
    def mpi_allgather(self,np.ndarray send_data,int send_count,np.ndarray recv_data,int recv_count):
        #cdef int send_count=send_data.shape[0]
        #cdef int recv_count=recv_data.shape[0]
        cdef void * send_data_ptr = np.PyArray_GETPTR1(send_data,0)
        cdef void * recv_data_ptr = np.PyArray_GETPTR1(recv_data,0)


        if send_data.dtype == np.int8:
            with nogil:
                self.MC_c.mpi_allgather(<np.int8_t*>send_data_ptr,send_count,<np.int8_t*>recv_data_ptr,recv_count)
        elif send_data.dtype == np.float64:
            with nogil:
                self.MC_c.mpi_allgather(<double*>send_data_ptr,send_count,<double*>recv_data_ptr,recv_count)


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



