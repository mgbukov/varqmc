import sys,os

os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel

quspin_path = os.path.join(os.path.expanduser('~'),"quspin/QuSpin_dev/")
sys.path.insert(0,quspin_path)

from jax.config import config
config.update("jax_enable_x64", True)
from jax import jit

from mpi4py import MPI
import numpy as np
import jax.numpy as jnp
import time






class ED_data():

	def __init__(self,comm, L, ):

		
		self.comm=comm
		self.L=L


	def compute_acceptance_ratio(self,N_accepted,N_MC_proposals,):

		self.acceptance_ratio_g[0]=-1.0

		




	def init_global_vars(self,L,N_batch,N_symm,NN_type,basis_type,MPI_basis_dtype,n_iter):

		self.N_batch=N_batch
		self.N_sites=L**2
		self.N_symm=N_symm
		self.N_features=self.N_symm*self.N_sites	
		self.basis_type=basis_type
		self.MPI_basis_dtype=MPI_basis_dtype

		
		self.NN_type=NN_type


	
		self.ints_ket=np.zeros((N_batch,),dtype=self.basis_type)
		

		self.log_mod_kets=np.zeros((N_batch,),dtype=np.float64)
		self.phase_kets=np.zeros((N_batch,),dtype=np.float64)


		self.log_psi_shift_g=np.zeros((n_iter,),dtype=np.float64)
		if self.comm.Get_rank()==0:
			self.ints_ket_g=np.array([[None],[None]])
			self.log_mod_kets_g=np.zeros((n_iter,N_MC_points,),dtype=np.float64)
			self.phase_kets_g=np.zeros((n_iter,N_MC_points,),dtype=np.float64)
		else:
			self.ints_ket_g=np.array([[None],[None]])
			self.log_mod_kets_g=np.array([[None],[None]])
			self.phase_kets_g=np.array([[None],[None]])


		
		self._reset_global_vars()
		
	def debug_helper(self):

		if self.comm.Get_rank()==0:

			self.log_psi_shift_g[:-1]=self.log_psi_shift_g[1:]
			self.ints_ket_g[:-1,...]=self.ints_ket_g[1:,...]
			self.log_mod_kets_g[:-1,...]=self.log_mod_kets_g[1:,...]
			self.phase_kets_g[:-1,...]=self.phase_kets_g[1:,...]

			self.log_psi_shift_g[-1]=self.log_psi_shift
			
		self.comm.Barrier()

		# collect data from multiple processes to root
		# self.comm.Gatherv([self.ints_ket,    self.MPI_basis_dtype], [self.ints_ket_g[-1,:],   self.MPI_basis_dtype], root=0)
		# self.comm.Gatherv([self.log_mod_kets,    MPI.DOUBLE], [self.log_mod_kets_g[-1,:],   MPI.DOUBLE], root=0)
		# self.comm.Gatherv([self.phase_kets,  MPI.DOUBLE], [self.phase_kets_g[-1,:], MPI.DOUBLE], root=0)



	
	def _reset_global_vars(self):
		self.spinstates_ket=np.zeros((self.N_batch*self.N_features,),dtype=np.int8)
		



	def exact(self, DNN_log, DNN_phase):

		if DNN_phase is not None: # real nets
			self.log_mod_kets[:] = DNN_log.evaluate(DNN_log.params,self.spinstates_ket.reshape(DNN_log.input_shape),  )
			self.phase_kets[:]   = DNN_phase.evaluate(DNN_phase.params,self.spinstates_ket.reshape(DNN_log.input_shape),  )
			
		else:
			self.log_mod_kets[:], self.phase_kets[:] = DNN_log.evaluate(DNN_log.params,self.spinstates_ket.reshape(DNN_log.input_shape),  )


		self.log_psi_shift=0.0 # np.max(self.log_mod_kets[:])
		self.log_mod_kets[:] -= self.log_psi_shift 

		#exit()
		
		# print(self.phase_kets[-1])
		# print(self.log_mod_kets[-1])
		
		#print('THERE', self.phase_kets[-16], self.phase_kets[-1])
		#exit()
		
		# print(self.spinstates_ket.reshape(self.N_batch,self.N_symm,self.N_sites)[-1,...])

		#exit()


		self.debug_helper()


		self.compute_acceptance_ratio(0,0)


