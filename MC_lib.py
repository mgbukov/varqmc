import sys,os

os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel

quspin_path = os.path.join(os.path.expanduser('~'),"quspin/QuSpin_dev/")
sys.path.insert(0,quspin_path)

from jax.config import config
config.update("jax_enable_x64", True)
from jax import jit

from cpp_code import representative

from mpi4py import MPI
import numpy as np
import jax.numpy as jnp
import time


# A function to print all prime factors of  
# a given number n 
def primeFactors(n): 
	  
	primes=[]

	# Print the number of two's that divide n 
	while n % 2 == 0: 
		primes.append(2) 
		n = n / 2
		  
	# n must be odd at this point 
	# so a skip of 2 ( i = i + 2) can be used 
	for i in range(3,int(np.sqrt(n))+1,2): 
		  
		# while i divides n , print i ad divide n 
		while n % i== 0: 
			primes.append(i) 
			n = n / i 
			  
	# Condition if n is a prime 
	# number greater than 2 
	if n > 2: 
		primes.append(n)

	return primes 


def closest(lst, K): 
	return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]



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





	def init_global_vars(self,L,N_MC_points,N_batch,N_minibatches,N_symm,NN_type,basis_type,MPI_basis_dtype,n_iter,mode):

		self.N_batch=N_batch
		

		self.N_sites=L**2
		self.N_symm=N_symm
		self.N_features=self.N_symm*self.N_sites
		
		self.basis_type=basis_type
		self.MPI_basis_dtype=MPI_basis_dtype

		self.thermalization_time=10*self.N_sites
		
		self.NN_type=NN_type


		#######

	
		self.ints_ket=np.zeros((N_batch,),dtype=self.basis_type)
		self.log_mod_kets=np.zeros((N_batch,),dtype=np.float64)
		self.phase_kets=np.zeros((N_batch,),dtype=np.float64)


		if mode=='MC':
			#self.log_mod_kets=np.zeros((N_batch,),dtype=np.float64)
			self.log_psi_shift_g=np.zeros((n_iter,),dtype=np.float64)
			if self.comm.Get_rank()==0:
				self.ints_ket_g=np.zeros((n_iter,N_MC_points,),dtype=self.basis_type)
				self.log_mod_kets_g=np.zeros((n_iter,N_MC_points,),dtype=np.float64)
				self.phase_kets_g=np.zeros((n_iter,N_MC_points,),dtype=np.float64)
			else:
				self.ints_ket_g=np.array([[None],[None]])
				self.log_mod_kets_g=np.array([[None],[None]])
				self.phase_kets_g=np.array([[None],[None]])


			self.s0_g=np.zeros(self.comm.Get_size()*self.N_MC_chains,dtype=self.basis_type)
			self.sf_g=np.zeros_like(self.s0_g)


			self.minibatch_size=self.N_batch*self.N_symm
			self.N_minibatches=1

		else:

			#primes=primeFactors(self.N_batch)

			self.ints_ket_g=np.zeros((N_MC_points,),dtype=self.basis_type)
			
			self.log_mod_kets_g=np.zeros((N_MC_points,),dtype=np.float64)
			self.phase_kets_g=np.zeros((N_MC_points,),dtype=np.float64)
			
			self.psi=np.zeros((N_batch,),dtype=np.complex128)
			

			self.N_minibatches=N_minibatches
			self.minibatch_size=self.N_symm*np.int(np.ceil(self.N_batch/self.N_minibatches))


		self.psi_batch_size = np.int(self.minibatch_size/self.N_symm*self.N_minibatches)


		self.log_mod_kets_aux=np.zeros((self.psi_batch_size,),dtype=np.float64)
		self.phase_kets_aux=np.zeros((self.psi_batch_size,),dtype=np.float64)

		#print(self.N_minibatches*self.minibatch_size, self.N_batch*self.N_symm)
		#exit()

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
		self.comm.Gatherv([self.ints_ket,	self.MPI_basis_dtype], [self.ints_ket_g[-1,:],   self.MPI_basis_dtype], root=0)
		self.comm.Gatherv([self.log_mod_kets,	MPI.DOUBLE], [self.log_mod_kets_g[-1,:],   MPI.DOUBLE], root=0)
		self.comm.Gatherv([self.phase_kets,  MPI.DOUBLE], [self.phase_kets_g[-1,:], MPI.DOUBLE], root=0)


	def all_gather(self):

			
		self.comm.Barrier()

		# collect data from multiple processes to root
		# self.comm.Gatherv([self.log_mod_kets,	MPI.DOUBLE], [self.log_mod_kets_g[:],   MPI.DOUBLE], root=0)
		# self.comm.Gatherv([self.phase_kets,  MPI.DOUBLE], [self.phase_kets_g[:], MPI.DOUBLE], root=0)
		self.comm.Allgatherv([self.log_mod_kets,	MPI.DOUBLE], [self.log_mod_kets_g[:],   MPI.DOUBLE], )
		self.comm.Allgatherv([self.phase_kets,  MPI.DOUBLE], [self.phase_kets_g[:], MPI.DOUBLE], )


		mod_psi=np.exp(self.log_mod_kets)

		norm_2=np.zeros(1, dtype=np.float64)
		self.comm.Allreduce(np.sum(self.count*mod_psi**2), norm_2,  op=MPI.SUM)
		
		self.psi[:] = mod_psi*np.exp(+1j*self.phase_kets)/np.sqrt(norm_2[0])

	
	def _reset_global_vars(self):
		self.spinstates_ket=np.zeros((self.N_minibatches*self.minibatch_size*self.N_sites,),dtype=np.int8)
		


	def sample(self, DNN_log, DNN_phase, compute_phases=True):

		self._reset_global_vars()

		N_accepted, N_MC_proposals = DNN_log.sample(self.N_batch,self.thermalization_time,self.acceptance_ratio_g,
												self.spinstates_ket,self.ints_ket,self.log_mod_kets, self.thermal,
												)

		if compute_phases:
			if DNN_phase is not None: # real nets
				if DNN_phase.semi_exact==False:
					self.phase_kets=np.asarray(DNN_phase.evaluate(DNN_phase.params, self.spinstates_ket.reshape(DNN_phase.input_shape), ))
				else: # exact phases
					representative(self.ints_ket,self.ints_ket,)
					self.phase_kets=np.asarray(DNN_phase.evaluate(DNN_phase.params, self.ints_ket, ))
			else: # cpx nets
				if DNN_log.semi_exact==False:
					self.phase_kets=np.asarray(DNN_log.evaluate_phase(DNN_log.params, self.spinstates_ket.reshape(DNN_log.input_shape), ))
				else: # exact phases
					representative(self.ints_ket,self.ints_ket,)
					self.phase_kets=np.asarray(DNN_log.evaluate_phase(DNN_log.params, self.ints_ket, ))


		#print(DNN_log.N_varl_params, DNN_phase.N_varl_params)
		# print(self.log_mod_kets.mean(), self.log_mod_kets.std() )
		# print(self.phase_kets.mean(), self.phase_kets.std() )
		# print(self.log_mod_kets)
		# exit()


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
			self.comm.Allgatherv([DNN_log.s0_vec,  self.MPI_basis_dtype], [self.s0_g, self.MPI_basis_dtype])
			self.comm.Allgatherv([DNN_log.sf_vec,  self.MPI_basis_dtype], [self.sf_g, self.MPI_basis_dtype])
		else:
			self.s0_g=DNN_log.s0_vec.copy()
			self.sf_g=DNN_log.sf_vec.copy()


		### compute acceptance ratio
		self.compute_acceptance_ratio(N_accepted,N_MC_proposals,mode='MC')
			
		
		self.debug_helper()

		return self.acceptance_ratio_g





	def exact(self, DNN_log, DNN_phase, logfile=None):


		if DNN_phase is not None: # real nets
			if DNN_phase.semi_exact==False:

				ti=time.time()
				for j in range(self.N_minibatches):

					batch_idx=np.arange(j*self.minibatch_size*self.N_sites, (j+1)*self.minibatch_size*self.N_sites)	
					array_idx=np.arange(j*self.minibatch_size//self.N_symm, (j+1)*self.minibatch_size//self.N_symm)
					
					batch=self.spinstates_ket[batch_idx]
					
					self.phase_kets_aux[array_idx] = DNN_phase.evaluate(DNN_phase.params,batch.reshape(DNN_phase.input_shape),  )
			
				self.phase_kets[:]=self.phase_kets_aux[:self.N_batch]

				print("phase network evaluation on {0:d} configs took {1:0.6} secs.".format(self.psi_batch_size, time.time()-ti) )

				#print(A.flags['OWNDATA'], B.flags['OWNDATA'], np.shares_memory(A,B))
				
				#self.phase_kets[:] = DNN_phase.evaluate(DNN_phase.params,self.spinstates_ket.reshape(DNN_log.input_shape),  )
				#self.phase_kets = np.asarray(DNN_phase.evaluate(DNN_phase.params,self.spinstates_ket.reshape(DNN_log.input_shape),  ))
			
			else:
				self.phase_kets = np.asarray(DNN_phase.evaluate(DNN_phase.params, self.ints_ket, ))


			if DNN_log.semi_exact==False:

				ti=time.time()
				for j in range(self.N_minibatches):

					batch_idx=np.arange(j*self.minibatch_size*self.N_sites, (j+1)*self.minibatch_size*self.N_sites)	
					array_idx=np.arange(j*self.minibatch_size//self.N_symm, (j+1)*self.minibatch_size//self.N_symm)
					
					batch=self.spinstates_ket[batch_idx]
					
					self.log_mod_kets_aux[array_idx] = DNN_log.evaluate(DNN_log.params,batch.reshape(DNN_log.input_shape),  )
			
				self.log_mod_kets[:]=self.log_mod_kets_aux[:self.N_batch]

				print("log network evaluation on {0:d} configs took {1:0.6} secs.".format(self.psi_batch_size, time.time()-ti) )
	
				
				#self.log_mod_kets = np.asarray(DNN_log.evaluate(DNN_log.params,self.spinstates_ket.reshape(DNN_log.input_shape),  ))
		

			else:
				self.log_mod_kets = np.asarray(DNN_log.evaluate(DNN_phase.params, self.ints_ket, ))

		else: # cpx nets
			if DNN_log.semi_exact==False:
				self.log_mod_kets, self.phase_kets = np.asarray( DNN_log.evaluate(DNN_log.params,self.spinstates_ket.reshape(DNN_log.input_shape),  ) )
			else:
				self.log_mod_kets, self.phase_kets = np.asarray( DNN_log.evaluate(DNN_phase.params, self.ints_ket, ) )

		self.all_gather()


		self.log_psi_shift=0.0 

		
		# print('PHASES', np.min(self.phase_kets), np.max(self.phase_kets))
		# exit()

		# for s, ph in zip(self.ints_ket, self.phase_kets):
		# 	print(s,ph%(2*np.pi))

		# exit()
		
		# print(self.phase_kets[-1])
		# print(self.log_mod_kets[-1])
		
		#print('THERE', self.phase_kets[-16], self.phase_kets[-1])
		#exit()
		
		# print(self.spinstates_ket.reshape(self.N_batch,self.N_symm,self.N_sites)[-1,...])

		#exit()


		self.compute_acceptance_ratio(0,0,mode='exact')


