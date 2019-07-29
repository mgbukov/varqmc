import sys,os

os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel

quspin_path = os.path.join(os.path.expanduser('~'),"quspin/QuSpin_dev/")
sys.path.insert(0,quspin_path)

from jax.config import config
config.update("jax_enable_x64", True)

from cpp_code import cpp_Monte_Carlo
from cpp_code import c_evaluate_NN
#from cpp_code import c_evaluate_mod, c_evaluate_phase, integer_to_spinstate

import numpy as np
import time




class MC_sampler():

	def __init__(self,build_dict_args=None):

		seed_vec=seed_vec=[3,7,19,117]

		self.MC_sampler=cpp_Monte_Carlo()
		self.MC_sampler.mpi_init()
		self.MC_sampler.set_seed(seed_vec[self.MC_sampler.world_rank])
		self.MC_sampler.build_ED_dicts(*build_dict_args)

	def init_global_vars(self,L,N_MC_points,N_symms,basis_type):

		self.N_MC_points=N_MC_points
		self.N_sites=L**2
		self.N_symms=N_symms
		self.N_features=self.N_symms*self.N_sites	
		self.basis_type=basis_type
		

		self.thermalization_time=10*self.N_sites
		self.auto_correlation_time=self.N_sites


		self.cyclicities_ket=np.zeros(self.N_MC_points,dtype=np.uint32)

		self.ints_ket=np.zeros((N_MC_points,),dtype=self.basis_type)
		self.ints_ket_reps=np.zeros_like(self.ints_ket)
		self.mod_kets=np.zeros((N_MC_points,),dtype=np.float64)
		self.phase_kets=np.zeros((N_MC_points,),dtype=np.float64)

		self.spinstates_ket_tot=np.zeros((self.MC_sampler.world_size*self.N_MC_points*self.N_features,),dtype=np.int8)
		self.mod_kets_tot=np.zeros((self.MC_sampler.world_size*N_MC_points,),dtype=np.float64)
		self.phase_kets_tot=np.zeros((self.MC_sampler.world_size*N_MC_points,),dtype=np.float64)

		self._reset_global_vars()
		

	def all_gather(self):

		self.MC_sampler.mpi_allgather(self.spinstates_ket,self.N_MC_points*self.N_features,self.spinstates_ket_tot,self.N_MC_points*self.N_features)
		self.MC_sampler.mpi_allgather(self.mod_kets,self.N_MC_points,self.mod_kets_tot,self.N_MC_points)
		self.MC_sampler.mpi_allgather(self.phase_kets,self.N_MC_points,self.phase_kets_tot,self.N_MC_points)


	def _reset_global_vars(self):
		self.spinstates_ket=np.zeros((self.N_MC_points*self.N_features,),dtype=np.int8)
		


	def sample(self,NN_params,N_neurons):

		self._reset_global_vars()
		assert(self.spinstates_ket.max()==0)

		#ti = time.time()
		N_accepted=self.MC_sampler.sample_DNN(self.N_MC_points,self.thermalization_time,self.auto_correlation_time,
						self.spinstates_ket,self.ints_ket,self.ints_ket_reps,self.mod_kets,self.phase_kets,
						*NN_params,N_neurons,
						)
		
		#print("cpp sampling took {0:.4f} sec with acceptance ratio {1:.4f}".format(time.time()-ti, N_accepted/self.N_MC_points))


	def GS_data(self,ints_ket):

		self.ints_ket=ints_ket

		self.MC_sampler.evaluate_mod_dict(ints_ket, self.mod_kets, self.N_MC_points)
		self.MC_sampler.evaluate_phase_dict(ints_ket, self.phase_kets, self.N_MC_points)



	def exact(self,ints_ket,NN_params,N_neurons,evaluate_NN=None):
		
		# int_ket=np.array([60097],dtype=ints_ket.dtype)
		# spinstate_ket=np.zeros(self.N_features,dtype=self.spinstates_ket.dtype)
		# integer_to_spinstate(int_ket, spinstate_ket, self.N_features)
		# print(spinstate_ket.reshape(-1,4,4))
		# print(np.unique(spinstate_ket.reshape(-1,self.N_sites),axis=0).shape )
		# print(int_ket)
		# exit()
		
		#c_evaluate_NN(ints_ket,self.spinstates_ket,self.mod_kets,self.phase_kets,*NN_params,self.N_sites,N_neurons,self.N_MC_points)
		if self.N_symms>1:
			#log_psi, phase_kets = evaluate_NN(NN_params,self.spinstates_ket.reshape(self.N_MC_points,self.N_symms,self.N_sites))
			log_psi, phase_kets = evaluate_NN(NN_params,self.spinstates_ket.reshape(self.N_MC_points,self.N_symms,self.N_sites),self.cyclicities_ket)
		else:
			log_psi, phase_kets = evaluate_NN(NN_params,self.spinstates_ket.reshape(self.N_MC_points,self.N_sites),self.cyclicities_ket)
		self.mod_kets = np.exp(log_psi._value)
		self.phase_kets= phase_kets._value



	#def MPI_allgather(data_local,N_local,data_all,N_all):
	#	self.MC_sampler.mpi_allgather(data_local,N_local,data_all,N_all)


	def check_consistency(self,evaluate_NN,NN_params):

		# combine results from all cores
		mod_kets_tot=np.zeros((self.MC_sampler.world_size*self.N_MC_points,),dtype=np.float64)
		phase_kets_tot=np.zeros((self.MC_sampler.world_size*self.N_MC_points,),dtype=np.float64)
		

		self.MC_sampler.mpi_allgather(self.phase_kets,self.N_MC_points,phase_kets_tot,self.N_MC_points)
		self.MC_sampler.mpi_allgather(self.mod_kets,self.N_MC_points,mod_kets_tot,self.N_MC_points)

		# evaluate network in python
		log_psi, phase_psi = evaluate_NN(NN_params,self.spinstates_ket_tot,self.cyclicities_ket) 


		# print(phase_kets_tot)
		# print(phase_psi)
		# # print()
		# print(mod_kets_tot)
		# print(np.exp(log_psi))
		
		# test results for consistency
		np.testing.assert_allclose(phase_psi, phase_kets_tot)
		np.testing.assert_allclose(np.exp(log_psi), mod_kets_tot)

