from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, device_put

from mpi4py import MPI
import numpy as np
from scipy.sparse.linalg import cg,cgs,bicg,bicgstab
#from cg import cg
from scipy.linalg import eigh,eig

import pickle



class natural_gradient():

	def __init__(self,comm,N_MC_points,N_batch,N_varl_params,compute_grad_log_psi, NN_Tree, grad_update_mode='normal'):
				 
		self.comm=comm


		self.compute_grad_log_psi=compute_grad_log_psi


		#self.TDVP_type='cpx' 
		self.TDVP_type='real'

		dtype=np.float64
		
		# if self.TDVP_type=='real':
		# 	dtype=np.float64 
		# elif self.TDVP_type=='cpx':
		# 	dtype=np.complex128 #

		self.N_batch=N_batch
		self.N_MC_points=N_MC_points
		self.N_varl_params=N_varl_params
		
		self.NN_Tree=NN_Tree


		######  preallocate memory
		
		self.dlog_psi=np.zeros([self.N_batch,N_varl_params],dtype=np.complex128)

		self.F_vector=np.zeros(N_varl_params,dtype=dtype)
		self.F_vector_log=np.zeros_like(self.F_vector)
		self.F_vector_phase=np.zeros_like(self.F_vector)

		self.nat_grad=np.zeros_like(self.F_vector)
		self.current_grad_guess=np.zeros_like(self.F_vector)
		self.S_matrix=np.zeros(2*self.F_vector.shape,dtype=dtype)
		self.S_matrix_reg=1E-15*np.eye(self.F_vector.shape[0])
		self.nat_grad_guess=np.zeros_like(self.F_vector)

		self.O_expt=np.zeros(N_varl_params,dtype=np.complex128)
		self.OO_expt=np.zeros([N_varl_params,N_varl_params],dtype=np.float64)
		self.O_expt2=np.zeros_like(self.OO_expt)

		self.E_diff_weighted=np.zeros(self.N_batch,dtype=np.complex128)

		
		# CG params
		self.check_on=True # toggles S-matrix checks
	
		self.cg_maxiter=1E4
		self.tol=1E-7 # CG tolerance
		self.delta=100.0 # S-matrix regularizer
		self.S_norm=0.0 # S-matrix norm
		self.F_norm=0.0 # F norm
		self.F_log_norm=0.0 
		self.F_phase_norm=0.0 


		self.run_debug_helper=None
		self.debug_mode=True

		self.iteration=0
		self.r2_cost=0.0
		self.max_grads=0.0


		# RK params
		self.RK_on=False

		self.RK_step_size=0.0
		self.RK_time=0.0
		self.RK_tol=0.0
		self.RK_inv_p=0.0
		self.counter=0
		

	def init_global_variables(self,n_iter):
		
		if self.comm.Get_rank()==0:
			self.S_lastiters=np.zeros([n_iter,self.N_varl_params,self.N_varl_params],dtype=np.float64) # array to store last S-matrices
			self.F_lastiters=np.zeros([n_iter,self.N_varl_params,],dtype=np.float64) # array to store last F-vectors
		else:
			self.S_lastiters=np.array([[None],[None]])
			self.F_lastiters=np.array([[None],[None]])
		
	def debug_helper(self):

		if self.comm.Get_rank()==0:

			# store last n_iter data points
			
			self.S_lastiters[:-1,...]=self.S_lastiters[1:,...]
			self.F_lastiters[:-1,...]=self.F_lastiters[1:,...]
			
			# set last step data
			self.S_lastiters[-1,...]=self.S_matrix
			self.F_lastiters[-1,...]=self.F_vector

		self.comm.Barrier() # Gatherv is blocking, so this is probably superfluous
		

	
	def compute_fisher_metric(self,mode='MC',Eloc_params_dict=None):
		
		if mode=='exact':
			abs_psi_2=Eloc_params_dict['abs_psi_2']#.copy()

			self.O_expt[:]=jnp.einsum('s,sj->j',abs_psi_2,self.dlog_psi).block_until_ready() 


			if self.TDVP_type=='real':
				self.OO_expt[:] = jnp.einsum('s,sk,sl->kl',abs_psi_2,self.dlog_psi.real, self.dlog_psi.real).block_until_ready() \
								 +jnp.einsum('s,sk,sl->kl',abs_psi_2,self.dlog_psi.imag, self.dlog_psi.imag).block_until_ready()
			elif self.TDVP_type=='cpx':
				self.OO_expt[:] = jnp.einsum('s,sk,sl->kl',abs_psi_2,self.dlog_psi.real, self.dlog_psi.imag).block_until_ready() \
								 -jnp.einsum('s,sk,sl->kl',abs_psi_2,self.dlog_psi.imag, self.dlog_psi.real).block_until_ready()

		elif mode=='MC':

			# self.O_expt[:]=jnp.sum(self.dlog_psi,axis=0)

			# OO_expt = jnp.einsum('sk,sl->kl',self.dlog_psi.real, self.dlog_psi.real)/self.N_MC_points \
			# 		 +jnp.einsum('sk,sl->kl',self.dlog_psi.imag, self.dlog_psi.imag)/self.N_MC_points

			self.comm.Allreduce(jnp.sum(self.dlog_psi,axis=0).block_until_ready()._value, self.O_expt, op=MPI.SUM)
			self.O_expt/=self.N_MC_points


			if self.TDVP_type=='real':
				self.comm.Allreduce((  jnp.einsum('sk,sl->kl',self.dlog_psi.real, self.dlog_psi.real).block_until_ready() \
					                  +jnp.einsum('sk,sl->kl',self.dlog_psi.imag, self.dlog_psi.imag).block_until_ready()    )._value, \
									self.OO_expt[:], op=MPI.SUM
									)

			elif self.TDVP_type=='cpx':
				self.comm.Allreduce((  jnp.einsum('sk,sl->kl',self.dlog_psi.real, self.dlog_psi.imag).block_until_ready() \
					                  -jnp.einsum('sk,sl->kl',self.dlog_psi.imag, self.dlog_psi.real).block_until_ready()    )._value, \
									self.OO_expt[:], op=MPI.SUM
									)

			
			self.OO_expt/=self.N_MC_points



		if self.TDVP_type=='real':
			self.O_expt2[:] = (   jnp.einsum('k,l->kl',self.O_expt.real,self.O_expt.real).block_until_ready() \
					  		      + jnp.einsum('k,l->kl',self.O_expt.imag,self.O_expt.imag).block_until_ready()    )._value

			self.S_matrix[:] = self.OO_expt - self.O_expt2 + self.S_matrix_reg

		elif self.TDVP_type=='cpx':
			self.O_expt2[:] = (    jnp.einsum('k,l->kl',self.O_expt.real,self.O_expt.imag).block_until_ready() \
					  		     - jnp.einsum('k,l->kl',self.O_expt.imag,self.O_expt.real).block_until_ready()    )._value

			#self.S_matrix[:] =1j*(self.OO_expt - self.O_expt2)
			self.S_matrix[:] = self.OO_expt - self.O_expt2


	

	def compute_gradients(self,mode='MC',Eloc_params_dict=None):

		self.E_diff_weighted[:]=Eloc_params_dict['E_diff'].copy()

		if mode=='exact':
			abs_psi_2=Eloc_params_dict['abs_psi_2'].copy()
			self.E_diff_weighted*=abs_psi_2

			self.F_vector_log[:]   = jnp.dot(self.E_diff_weighted.real,self.dlog_psi.real).block_until_ready()
			self.F_vector_phase[:] = jnp.dot(self.E_diff_weighted.imag,self.dlog_psi.imag).block_until_ready()
			

			# self.F_vector[:] = jnp.dot(self.E_diff_weighted.real,self.dlog_psi.real).block_until_ready() \
			# 			     + jnp.dot(self.E_diff_weighted.imag,self.dlog_psi.imag).block_until_ready()

		elif mode=='MC':
			# self.F_vector[:] = jnp.dot(self.E_diff_weighted.real,self.dlog_psi.real)/self.N_MC_points \
			#              + jnp.dot(self.E_diff_weighted.imag,self.dlog_psi.imag)/self.N_MC_points


			if self.TDVP_type=='real':

				self.comm.Allreduce((  jnp.dot(self.E_diff_weighted.real,self.dlog_psi.real).block_until_ready() )._value, \
									self.F_vector_log[:], op=MPI.SUM
									)

				self.comm.Allreduce((  jnp.dot(self.E_diff_weighted.imag,self.dlog_psi.imag).block_until_ready() )._value, \
									self.F_vector_phase[:], op=MPI.SUM
									)


				# self.comm.Allreduce((  jnp.dot(self.E_diff_weighted.real,self.dlog_psi.real).block_until_ready() \
				#              		 + jnp.dot(self.E_diff_weighted.imag,self.dlog_psi.imag).block_until_ready()   )._value, \
				# 					self.F_vector[:], op=MPI.SUM
				# 					)

			elif self.TDVP_type=='cpx':
				self.comm.Allreduce((  jnp.dot(self.E_diff_weighted.real,self.dlog_psi.imag).block_until_ready() \
				             		 - jnp.dot(self.E_diff_weighted.imag,self.dlog_psi.real).block_until_ready()   )._value + 0.0j, \
									self.F_vector[:], op=MPI.SUM
									)
				#self.F_vector*=1j


			self.F_vector_log/=self.N_MC_points
			self.F_vector_phase/=self.N_MC_points


		self.F_vector[:]=self.F_vector_log+self.F_vector_phase




	def _compute_r2_cost(self,Eloc_params_dict):
		Eloc_var=Eloc_params_dict['Eloc_var']
		return ( (np.dot(self.nat_grad.conj(), np.dot(self.S_matrix,self.nat_grad)) - 2.0*np.dot(self.F_vector.conj(),self.nat_grad).real + Eloc_var )/Eloc_var ).real

	def _S_matrix_checks(self):

		# import pickle
		# file_name='./bug'
		# with open(file_name+'.pkl', 'wb') as handle:
		# 	pickle.dump([self.dlog_psi, self.OO_expt, self.O_expt2, self.O_expt, self.S_matrix, self.F_vector], handle, protocol=pickle.HIGHEST_PROTOCOL)


		
		norm=jnp.linalg.norm(self.S_matrix).block_until_ready()

		
		if np.linalg.norm((self.S_matrix-self.S_matrix.T.conj())/norm ) > 1E-14: # and np.linalg.norm(self.S_matrix-self.S_matrix.T.conj()) > 1E-14:
		
			print('F : {:.20f}'.format(np.linalg.norm((self.S_matrix-self.S_matrix.T.conj())/norm )) )
			print('OO: {:.20f}'.format(np.linalg.norm( (self.OO_expt-self.OO_expt.T.conj())/np.linalg.norm(self.OO_expt) )) )
			print('O2: {:.20f}'.format(np.linalg.norm( (self.O_expt2-self.O_expt2.T.conj())/np.linalg.norm(self.O_expt2) )) )
			print('non-hermitian Fisher matrix')
			
			np.testing.assert_allclose(self.S_matrix/norm,self.S_matrix.T.conj()/norm, rtol=1E-14, atol=1E-14)

			exit()
		
		E = eigh(self.S_matrix/norm,eigvals_only=True)
		

		if np.any(E <- 1E-14):
			print('E ', E)
			print()
			print('OO',eigh(self.OO_expt/np.linalg.norm(self.OO_expt),eigvals_only=True))
			print()
			print('O2',eigh(self.O_expt2/np.linalg.norm(self.O_expt2),eigvals_only=True))

			print('negative eigenvalues')


			np.linalg.cholesky(self.S_matrix/norm)


			exit()



	def compute(self,NN_params,batch,Eloc_params_dict,mode='MC',):
		

		self.dlog_psi[:]=self.compute_grad_log_psi(NN_params,batch,self.iteration)

		# print(self.dlog_psi[:,-1])
		# exit()

		self.compute_gradients(Eloc_params_dict=Eloc_params_dict,mode=mode)
		self.compute_fisher_metric(Eloc_params_dict=Eloc_params_dict,mode=mode)


		### compute natural_gradients using cg
		# regularize Fisher metric
		self.S_matrix += self.delta*np.diag(np.diag(self.S_matrix))
		#self.S_matrix += self.delta*np.linalg.norm(self.S_matrix)*np.eye(self.S_matrix.shape[0]) 

		self.debug_helper()

		#self.nat_grad=np.linalg.inv(self.S_matrix).dot(self.F_vector)

		# E, V = eigh(self.S_matrix)
		# print(E)
		# exit()


		####################################################### 
		
		if self.debug_mode:
			# run debug helper	
			self.run_debug_helper()
			
			# check for symmetry and positivity
			if self.check_on and self.TDVP_type=='real':
				self._S_matrix_checks()

		# compute norm
		self.S_norm=np.linalg.norm(self.S_matrix)
		self.F_norm=np.linalg.norm(self.F_vector)
		self.F_log_norm=np.linalg.norm(self.F_vector_log)
		self.F_phase_norm=np.linalg.norm(self.F_vector_phase)

		#######################################################

		# apply conjugate gradient a few times
		info=1
		while info>0 and self.cg_maxiter<1E5:
			# apply cg
			#self.nat_grad, info, iter_ = cg(self.S_matrix,self.F_vector,x0=self.nat_grad_guess,maxiter=self.cg_maxiter,atol=self.tol,tol=self.tol)
			if self.TDVP_type=='real':
				self.nat_grad, info = cg(self.S_matrix,self.F_vector,x0=self.nat_grad_guess,maxiter=self.cg_maxiter,atol=self.tol,tol=self.tol)
			# elif self.TDVP_type=='cpx':
			# 	self.nat_grad, info = bicg(self.S_matrix,self.F_vector,x0=self.nat_grad_guess,maxiter=self.cg_maxiter,atol=self.tol,tol=self.tol)
			 
			# 
			if info>0:
				self.cg_maxiter*=2
				print('cg failed to converge in {0:d} iterations to tolerance {1:0.14f}; increasing maxiter to {2:d}'.format(info,self.tol,int(self.cg_maxiter)))
				


		# store guess for next true
		self.current_grad_guess[:]=self.nat_grad

		

		# normalize gradients
		if not self.RK_on:
			self.r2_cost=self._compute_r2_cost(Eloc_params_dict)
			self.max_grads=[np.max(jnp.abs(self.F_vector.real)), np.max(jnp.abs(self.nat_grad.real))]


			#self.nat_grad /= np.sqrt(jnp.dot(self.F_vector.conj(),self.nat_grad).real)
			
			return self.nat_grad
		else:
			return self.nat_grad
		

	def update_params(self,self_time=1.0):

		#self.delta *= np.exp(-0.075*self_time)

		if self.delta>self.tol:
			self.delta *= np.exp(-0.075*self_time)
		# else:
		# 	self.delta=0.0

		# if self.tol>1E-7:
		# 	 self.tol *= 0.95 #np.exp(-0.05*self_time)
	

		self.nat_grad_guess[:]=self.current_grad_guess[:]
		#print('delta={0:0.4f}'.format(self.delta))

		self.iteration+=1


	def init_RK_params(self,learning_rate):

		self.RK_step_size=learning_rate
		self.RK_time=0.0
		self.RK_tol=5E-5 #self.tol
		self.RK_inv_p=1.0/3.0
		self.counter=0
		self.delta=0.001 # NG S matrix regularization strength

		self.RK_on=True
		
		self.dy=np.zeros_like(self.nat_grad)
		self.dy_star=np.zeros_like(self.dy)

		self.k1=np.zeros_like(self.dy)
		self.k2=np.zeros_like(self.dy)
		self.k3=np.zeros_like(self.dy)
		self.k4=np.zeros_like(self.dy)
		self.k5=np.zeros_like(self.dy)
		self.k6=np.zeros_like(self.dy)



	def Runge_Kutta(self,NN_params,batch,Eloc_params_dict,mode,get_training_data):

		# flatten weights
		params=self.NN_Tree.ravel(NN_params,)
		
		max_param=jnp.max(jnp.abs(params)).block_until_ready()
		#max_param=jnp.max(np.abs(params[:32]+1j*params[32:]))

		initial_grad=self.compute(NN_params,batch,Eloc_params_dict,mode=mode)
		# cost and loss
		self.max_grads=[jnp.max(jnp.abs(self.F_vector.real)).block_until_ready(), jnp.max(jnp.abs(self.nat_grad.real)).block_until_ready()]	
		self.r2_cost=self._compute_r2_cost(Eloc_params_dict)
	
		error_ratio=0.0
		while error_ratio<1.0:

			### RK step 1
			self.k1[:]=-self.RK_step_size*initial_grad
			
			### RK step 2
			NN_params_shifted=self.NN_Tree.unravel(params+self.k1)
			batch, Eloc_params_dict=get_training_data(self.iteration,NN_params_shifted)
			self.nat_grad_guess[:]=self.k1/self.RK_step_size
			self.k2[:]=-self.RK_step_size*self.compute(NN_params_shifted,batch,Eloc_params_dict,mode=mode)
			self.dy[:]=0.5*self.k1 + 0.5*self.k2


			#######
			### RK step 1
			self.k3[:]=-0.5*self.RK_step_size*initial_grad # 0.5*k1
			
			### RK step 2
			NN_params_shifted=self.NN_Tree.unravel(params+self.k3)
			batch, Eloc_params_dict=get_training_data(self.iteration,NN_params_shifted)
			self.nat_grad_guess[:]=self.k3/(0.5*self.RK_step_size)
			self.k4[:]=-0.5*self.RK_step_size*self.compute(NN_params_shifted,batch,Eloc_params_dict,mode=mode)

			# first half-step solution difference
			self.dy_star[:]=0.5*self.k3 + 0.5*self.k4
			
			### RK step 1
			NN_params_shifted=self.NN_Tree.unravel(params+self.dy_star)
			batch, Eloc_params_dict=get_training_data(self.iteration,NN_params_shifted)
			self.nat_grad_guess[:]=self.k4/(0.5*self.RK_step_size)
			self.k5[:]=-0.5*self.RK_step_size*self.compute(NN_params_shifted,batch,Eloc_params_dict,mode=mode)

			### RK step 2
			NN_params_shifted=self.NN_Tree.unravel(params+self.dy_star+self.k5)
			batch, Eloc_params_dict=get_training_data(self.iteration,NN_params_shifted)
			self.nat_grad_guess[:]=self.k5/(0.5*self.RK_step_size)
			self.k6[:]=-0.5*self.RK_step_size*self.compute(NN_params_shifted,batch,Eloc_params_dict,mode=mode)

			# second half-step solution difference
			self.dy_star[:]+=0.5*self.k5 + 0.5*self.k6


			#######
			
			# dy=self.dy[:32]+1j*self.dy[32:]
			# dy_star=self.dy_star[:32]+1j*self.dy_star[32:]

			norm=jnp.max(jnp.abs(self.dy-self.dy_star)).block_until_ready()/max_param
			#norm=jnp.max(jnp.abs(dy-dy_star))/max_param

			error_ratio=6.0*self.RK_tol/norm

			
			self.RK_step_size*=min(2.0,max(0.2,0.9*error_ratio**self.RK_inv_p))

			
			self.counter+=5 # five gradient calculations

			
		# update NG params
		self.update_params(self_time=self.RK_time)
		self.RK_time+=self.RK_step_size
		
		
		print('steps={0:d}-step_size={1:0.12f}-time={2:0.4f}-delta={3:0.10f}-cg_tol-{4:0.12f}.\n'.format(self.counter, self.RK_step_size, self.RK_time, self.delta, self.tol) )
	
		return self.NN_Tree.unravel(params + self.dy_star - 1.0/6.0*(self.dy-self.dy_star), )

