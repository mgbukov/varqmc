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
		self.acceptance_ratio=np.array([0.4])


	def compute_acceptance_ratio(self,N_accepted,N_MC_proposals,mode='MC'):

		if mode=='exact':
			self.acceptance_ratio[0]=-1.0

		elif mode=='MC':
			# sum up MC metaata over the MPI processes
			self.comm.Allreduce(np.array(N_accepted,dtype=np.int), self.N_accepted,  op=MPI.SUM)
			self.comm.Allreduce(np.array(N_MC_proposals,dtype=np.int), self.N_MC_proposals,  op=MPI.SUM)
			self.acceptance_ratio[0]=N_accepted/N_MC_proposals 



	def init_global_vars(self,L,N_MC_points,N_batch,N_symm,basis_type):

		self.N_batch=N_batch
		self.N_sites=L**2
		self.N_symm=N_symm
		self.N_features=self.N_symm*self.N_sites	
		self.basis_type=basis_type
		

		self.thermalization_time=10*self.N_sites
		#self.auto_correlation_time=self.N_sites  # min(0.05, 0.4/acc_ratio * N_site_


	
		self.ints_ket=np.zeros((N_batch,),dtype=self.basis_type)
		#self.ints_ket_reps=np.zeros_like(self.ints_ket)
		self.mod_kets=np.zeros((N_batch,),dtype=np.float64)
		self.phase_kets=np.zeros((N_batch,),dtype=np.float64)

		#self.spinstates_ket_tot=np.zeros((N_MC_points*self.N_features,),dtype=np.int8)
		#self.mod_kets_tot=np.zeros((N_MC_points,),dtype=np.float64)
		#self.phase_kets_tot=np.zeros((N_MC_points,),dtype=np.float64)

		
		self.s0=np.zeros(self.N_MC_chains,dtype=self.basis_type)
		self.s0_tot=np.zeros(self.comm.Get_size()*self.N_MC_chains,dtype=self.basis_type)


		self._reset_global_vars()
		
	'''
	def Allgather(self):

		self.mod_kets_tot*=0.0

		
		
		#self.comm.Allgatherv([self.spinstates_ket,  MPI.DOUBLE], [self.spinstates_ket_tot, MPI.DOUBLE])
		self.comm.Allgatherv([self.mod_kets,    MPI.DOUBLE], [self.mod_kets_tot,   MPI.DOUBLE])
		self.comm.Allgatherv([self.phase_kets,  MPI.DOUBLE], [self.phase_kets_tot, MPI.DOUBLE])


		if self.comm.Get_size()*self.N_MC_chains > 1:
			self.comm.Allgatherv([self.s0,  MPI.INT], [self.s0_tot, MPI.INT])
		else:
			self.s0_tot=self.s0.copy()
	'''	
	
	def _reset_global_vars(self):
		self.spinstates_ket=np.zeros((self.N_batch*self.N_features,),dtype=np.int8)
		


	def sample(self,DNN):

		self._reset_global_vars()
		#assert(self.spinstates_ket.max()==0)

		N_accepted, N_MC_proposals = DNN.sample(self.N_batch,self.thermalization_time,self.acceptance_ratio,
												self.spinstates_ket,self.ints_ket,self.mod_kets,self.s0,
												)

		# N_accepted, N_MC_proposals = sample(self.N_batch,self.thermalization_time,self.acceptance_ratio,
		# 									self.spinstates_ket,self.ints_ket,self.mod_kets,self.s0,
		# 									DNN)


		self.phase_kets[:]=DNN.evaluate_phase(DNN.params, self.spinstates_ket.reshape(self.N_batch*self.N_symm,self.N_sites), DNN.apply_fun_args)#._value
		
		self.log_psi_shift=0.0 


		self.compute_acceptance_ratio(N_accepted,N_MC_proposals,mode='MC')
			
		
		return self.acceptance_ratio





	def exact(self,evaluate_NN,DNN):

		log_psi, phase_kets = evaluate_NN(DNN.params,self.spinstates_ket.reshape(self.N_batch,self.N_symm,self.N_sites))
		
		#print(log_psi)
		#exit()

		self.log_psi_shift=log_psi[0]._value
		self.mod_kets[:] = jnp.exp((log_psi-self.log_psi_shift)).block_until_ready()#._value
		#self.mod_kets = np.exp(log_psi._value)
		self.phase_kets[:]= phase_kets#._value

		self.compute_acceptance_ratio(0,0,mode='exact')


	'''
	def check_consistency(self,evaluate_NN,NN_params):

		# reshape
		spinstates_ket=self.spinstates_ket.reshape(-1,self.N_symm,self.N_sites)
		
		# combine results from all cores
		mod_kets_tot=np.zeros((self.comm.Get_size()*self.N_batch,),dtype=np.float64)
		phase_kets_tot=np.zeros((self.comm.Get_size()*self.N_batch,),dtype=np.float64)

		log_psi_tot=np.zeros((self.comm.Get_size()*self.N_batch,),dtype=np.float64)
		phase_psi_tot=np.zeros((self.comm.Get_size()*self.N_batch,),dtype=np.float64)
		
		self.comm.Allgather([self.mod_kets,  MPI.DOUBLE], [mod_kets_tot, MPI.DOUBLE])
		self.comm.Allgather([self.phase_kets,  MPI.DOUBLE], [phase_kets_tot, MPI.DOUBLE])
		

		# evaluate network in python
		log_psi, phase_psi = evaluate_NN(NN_params,spinstates_ket) 
		log_psi-=self.log_psi_shift

		self.comm.Allgather([log_psi._value,  MPI.DOUBLE], [log_psi_tot, MPI.DOUBLE])
		self.comm.Allgather([phase_psi._value,  MPI.DOUBLE], [phase_psi_tot, MPI.DOUBLE])
		


		# print(phase_kets_tot)
		# print(phase_psi)
		# # print()
		# print(mod_kets_tot)
		# print(np.exp(log_psi))
		# print(mod_kets_tot-np.exp(log_psi))
		# exit()

		
		# test results for consistency
		np.testing.assert_allclose(phase_psi_tot, phase_kets_tot)
		np.testing.assert_allclose(np.exp(log_psi_tot), mod_kets_tot)
	'''
