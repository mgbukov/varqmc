import sys,os

os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel

quspin_path = os.path.join(os.path.expanduser('~'),"quspin/QuSpin_dev/")
sys.path.insert(0,quspin_path)

from jax.config import config
config.update("jax_enable_x64", True)
from jax import jit

from cpp_code import cpp_Monte_Carlo

from mpi4py import MPI
import numpy as np
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import time




class MC_sampler():

	def __init__(self,comm,build_dict_args=None):

		seed_vec=seed_vec=[3,7,19,117]

		self.comm=comm

		self.MC_sampler=cpp_Monte_Carlo()
		self.MC_sampler.set_seed(seed_vec[self.comm.Get_rank()])
		self.MC_sampler.build_ED_dicts(*build_dict_args)

	def init_global_vars(self,L,N_MC_points,N_symms,basis_type):

		self.N_MC_points=N_MC_points
		self.N_sites=L**2
		self.N_symms=N_symms
		self.N_features=self.N_symms*self.N_sites	
		self.basis_type=basis_type
		

		self.thermalization_time=10*self.N_sites
		self.auto_correlation_time=self.N_sites


	
		self.ints_ket=np.zeros((N_MC_points,),dtype=self.basis_type)
		#self.ints_ket_reps=np.zeros_like(self.ints_ket)
		self.mod_kets=np.zeros((N_MC_points,),dtype=np.float64)
		self.phase_kets=np.zeros((N_MC_points,),dtype=np.float64)

		self.spinstates_ket_tot=np.zeros((self.comm.Get_size()*self.N_MC_points*self.N_features,),dtype=np.int8)
		self.mod_kets_tot=np.zeros((self.comm.Get_size()*N_MC_points,),dtype=np.float64)
		self.phase_kets_tot=np.zeros((self.comm.Get_size()*N_MC_points,),dtype=np.float64)

		self._reset_global_vars()
		

	def Allgather(self):

		self.comm.Allgather([self.spinstates_ket,  MPI.INT], [self.spinstates_ket_tot, MPI.INT])
		self.comm.Allgather([self.mod_kets,  MPI.DOUBLE], [self.mod_kets_tot, MPI.DOUBLE])
		self.comm.Allgather([self.phase_kets,  MPI.DOUBLE], [self.phase_kets_tot, MPI.DOUBLE])

		

	def _reset_global_vars(self):
		self.spinstates_ket=np.zeros((self.N_MC_points*self.N_features,),dtype=np.int8)
		


	def sample(self,NN_params,N_neurons,DNN=None):

		self._reset_global_vars()
		assert(self.spinstates_ket.max()==0)
		
		#ti = time.time()

		# N_accepted=self.MC_sampler.sample_DNN(self.N_MC_points,self.thermalization_time,self.auto_correlation_time,
		# 				self.spinstates_ket,self.ints_ket,self.mod_kets,self.phase_kets,
		# 				*NN_params,N_neurons,
		# 				)

		N_accepted=DNN.sample(self.N_MC_points,self.thermalization_time,self.auto_correlation_time,
						self.spinstates_ket,self.ints_ket,self.mod_kets,
						)
		self.phase_kets[:]=DNN.evaluate_phase(self.spinstates_ket.reshape(self.N_MC_points,self.N_symms,self.N_sites))#._value
		

		self.log_psi_shift=0.0 #log_psi[0]
		
		#print("cpp sampling took {0:.4f} sec with acceptance ratio {1:.4f}".format(time.time()-ti, N_accepted/self.N_MC_points))


	def GS_data(self,ints_ket):

		self.ints_ket=ints_ket

		self.MC_sampler.evaluate_mod_dict(ints_ket, self.mod_kets, self.N_MC_points)
		self.MC_sampler.evaluate_phase_dict(ints_ket, self.phase_kets, self.N_MC_points)



	def exact(self,NN_params,evaluate_NN=None):
		
		# int_ket=np.array([60097],dtype=ints_ket.dtype)
		# spinstate_ket=np.zeros(self.N_features,dtype=self.spinstates_ket.dtype)
		# integer_to_spinstate(int_ket, spinstate_ket, self.N_features)
		# print(spinstate_ket.reshape(-1,4,4))
		# print(np.unique(spinstate_ket.reshape(-1,self.N_sites),axis=0).shape )
		# print(int_ket)
		# exit()
		
		if self.N_symms>1:
			#log_psi, phase_kets = evaluate_NN(NN_params,self.spinstates_ket.reshape(self.N_MC_points,self.N_symms,self.N_sites))
			log_psi, phase_kets = evaluate_NN(NN_params,self.spinstates_ket.reshape(self.N_MC_points,self.N_symms,self.N_sites))
		else:
			log_psi, phase_kets = evaluate_NN(NN_params,self.spinstates_ket.reshape(self.N_MC_points,self.N_sites))
		self.log_psi_shift=0.0 #log_psi[0]
		self.mod_kets[:] = jnp.exp((log_psi-self.log_psi_shift)).block_until_ready()#._value
		#self.mod_kets = np.exp(log_psi._value)
		self.phase_kets[:]= phase_kets#._value



	#def MPI_allgather(data_local,N_local,data_all,N_all):
	#	self.MC_sampler.mpi_allgather(data_local,N_local,data_all,N_all)


	def check_consistency(self,evaluate_NN,NN_params,symmetrized):

		# reshape
		if symmetrized:
			spinstates_ket=self.spinstates_ket.reshape(-1,self.N_symms,self.N_sites)
		else:
			spinstates_ket=self.spinstates_ket.reshape(-1,self.N_sites)

		# combine results from all cores
		mod_kets_tot=np.zeros((self.comm.Get_size()*self.N_MC_points,),dtype=np.float64)
		phase_kets_tot=np.zeros((self.comm.Get_size()*self.N_MC_points,),dtype=np.float64)

		log_psi_tot=np.zeros((self.comm.Get_size()*self.N_MC_points,),dtype=np.float64)
		phase_psi_tot=np.zeros((self.comm.Get_size()*self.N_MC_points,),dtype=np.float64)
		
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

		
		# test results for consistency
		np.testing.assert_allclose(phase_psi_tot, phase_kets_tot)
		np.testing.assert_allclose(np.exp(log_psi_tot), mod_kets_tot)

