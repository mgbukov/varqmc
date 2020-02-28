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
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import time






class MC_sampler():

	def __init__(self,comm, N_MC_chains, build_dict_args=None):

		
		self.comm=comm
		self.N_MC_chains=N_MC_chains

		self.N_accepted=np.zeros(1, dtype=np.int)
		self.N_MC_proposals=np.zeros(1, dtype=np.int)
		self.acceptance_ratio_g=np.array([1.0])
		self.acceptance_ratio=np.zeros(shape=self.comm.Get_size(),)

		self.thermal=False




	def compute_acceptance_ratio(self,N_accepted,N_MC_proposals,mode='MC'):

		if mode=='exact':
			self.acceptance_ratio_g[0]=-1.0

		elif mode=='MC':
			# sum up MC metaata over the MPI processes
			self.comm.Allreduce(np.array(N_accepted,dtype=np.int), self.N_accepted,  op=MPI.SUM)
			self.comm.Allreduce(np.array(N_MC_proposals,dtype=np.int), self.N_MC_proposals,  op=MPI.SUM)
			self.acceptance_ratio_g[0]=self.N_accepted/self.N_MC_proposals

			self.comm.Allgatherv([N_accepted/N_MC_proposals,  MPI.DOUBLE], [self.acceptance_ratio, MPI.DOUBLE])





	def init_global_vars(self,L,N_MC_points,N_batch,N_symm,basis_type,MPI_basis_dtype,n_iter):

		self.N_batch=N_batch
		self.N_sites=L**2
		self.N_symm=N_symm
		self.N_features=self.N_symm*self.N_sites	
		self.basis_type=basis_type
		self.MPI_basis_dtype=MPI_basis_dtype

		self.thermalization_time=10*self.N_sites
		#self.auto_correlation_time=self.N_sites  # min(0.05, 0.4/acc_ratio * N_site_


	
		self.ints_ket=np.zeros((N_batch,),dtype=self.basis_type)
		#self.ints_ket_reps=np.zeros_like(self.ints_ket)
		self.log_mod_kets=np.zeros((N_batch,),dtype=np.float64)
		self.phase_kets=np.zeros((N_batch,),dtype=np.float64)


		self.log_psi_shift_g=np.zeros((n_iter,),dtype=np.float64)
		if self.comm.Get_rank()==0:
			self.ints_ket_g=np.zeros((n_iter,N_MC_points,),dtype=self.basis_type)
			self.log_mod_kets_g=np.zeros((n_iter,N_MC_points,),dtype=np.float64)
			self.phase_kets_g=np.zeros((n_iter,N_MC_points,),dtype=np.float64)
		else:
			self.ints_ket_g=np.array([[None],[None]])
			self.log_mod_kets_g=np.array([[None],[None]])
			self.phase_kets_g=np.array([[None],[None]])


		self.s0=np.zeros(self.N_MC_chains,dtype=self.basis_type)
		self.s0_g=np.zeros(self.comm.Get_size()*self.N_MC_chains,dtype=self.basis_type)


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
		self.comm.Gatherv([self.ints_ket,    self.MPI_basis_dtype], [self.ints_ket_g[-1,:],   self.MPI_basis_dtype], root=0)
		self.comm.Gatherv([self.log_mod_kets,    MPI.DOUBLE], [self.log_mod_kets_g[-1,:],   MPI.DOUBLE], root=0)
		self.comm.Gatherv([self.phase_kets,  MPI.DOUBLE], [self.phase_kets_g[-1,:], MPI.DOUBLE], root=0)



	
	def _reset_global_vars(self):
		self.spinstates_ket=np.zeros((self.N_batch*self.N_features,),dtype=np.int8)
		


	def sample(self,DNN, compute_phases=True):

		self._reset_global_vars()
		#assert(self.spinstates_ket.max()==0)

		N_accepted, N_MC_proposals = DNN.sample(self.N_batch,self.thermalization_time,self.acceptance_ratio_g,
												self.spinstates_ket,self.ints_ket,self.log_mod_kets,self.s0, self.thermal,
												)

		#print(self.ints_ket)
		#exit()

		if compute_phases:
			self.phase_kets[:]=DNN.evaluate_phase(DNN.params, self.spinstates_ket.reshape(self.N_batch*self.N_symm,self.N_sites), )#._value

		
		### normalize all kets
		#
		self.log_psi_shift=0.0
		
		# compute global max
		local_max=np.max(self.log_mod_kets).astype(np.float64)
		global_max=np.zeros(1, dtype=np.float64)
		self.comm.Reduce(local_max, global_max, op=MPI.MAX) # broadcast to root=0
		
		if self.comm.Get_rank()==0:
			self.log_psi_shift=global_max[0]
		# broadcast sys_data
		self.log_psi_shift = self.comm.bcast(self.log_psi_shift, root=0)
		#
		# normalize
		self.log_mod_kets-=self.log_psi_shift


		### gather seeds

		if self.comm.Get_size()*self.N_MC_chains > 1:
			self.comm.Allgatherv([self.s0,  self.MPI_basis_dtype], [self.s0_g, self.MPI_basis_dtype])
		else:
			self.s0_g=self.s0.copy()


		### compute acceptance ratio
		self.compute_acceptance_ratio(N_accepted,N_MC_proposals,mode='MC')
			
		
		self.debug_helper()

		return self.acceptance_ratio_g





	def exact(self,evaluate_NN,DNN, E_estimator,):

		self.log_mod_kets[:], self.phase_kets[:] = evaluate_NN(DNN.params,self.spinstates_ket.reshape(self.N_batch,self.N_symm,self.N_sites), DNN.apply_fun_args )
		
		#print(self.log_mod_kets)
		#exit()

		#print("MEAN log_psi: ",  np.sum(E_estimator.count*self.log_mod_kets)/E_estimator.basis.Ns,  np.max(self.log_mod_kets)  )
		#exit()

		self.log_psi_shift=np.max(self.log_mod_kets[:])#._value
		self.log_mod_kets[:] -= self.log_psi_shift 
		


		# for j, spin_config in enumerate(self.spinstates_ket.reshape(self.N_batch,self.N_symm,self.N_sites)):
		# 	print(spin_config[0,...].reshape(4,4))
		# 	print()
		
		# print(log_psi)
		# print()
		# exit()


		self.debug_helper()


		self.compute_acceptance_ratio(0,0,mode='exact')


