from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, device_put

from jax.experimental.optimizers import make_schedule

from mpi4py import MPI
import numpy as np
from scipy.sparse.linalg import cg #,cgs,bicg,bicgstab
#from cg import cg
from scipy.linalg import eigh,eig

import pickle



class natural_gradient():

	def __init__(self,comm,compute_grad_log_psi, NN_Tree, TDVP_opt, mode='MC', RK=False):
				 
		self.comm=comm
		self.logfile=None

		self.NN_Tree=NN_Tree
		self.mode=mode
		self.RK=RK

	
		self.compute_grad_log_psi=compute_grad_log_psi


		self.TDVP_opt = TDVP_opt # 'svd' # 'inv' # 'cg' #

		self.dE=0.0
		

		self.debug_mode=True
		
		# CG params
		self.check_on=False # toggles S-matrix checks
		self.cg_maxiter=1E4
		self.tol=1E-7 # CG tolerance

		self.step_size=1.0

		if self.RK:
			self.delta=0.0 # NG S matrix regularization strength
			self.grad_clip=1E4
		else:
			self.delta=100.0 # S-matrix regularizer
			self.grad_clip=1E4 #50.0


		

	def init_global_variables(self,N_MC_points,N_batch,N_varl_params_vec,n_iter):


		self.N_batch=N_batch
		self.N_MC_points=N_MC_points
		self.N_varl_params=np.sum(N_varl_params_vec)
		self.N_varl_params_vec=N_varl_params_vec
			

		######  preallocate memory
		dtype=np.float64
				
		self.dlog_psi=np.zeros([self.N_batch,self.N_varl_params],dtype=dtype)

		self.F_vector=np.zeros(self.N_varl_params,dtype=dtype)
		
		self.nat_grad=np.zeros_like(self.F_vector)
		self.current_grad_guess=np.zeros_like(self.F_vector)
		self.S_matrix=np.zeros(2*self.F_vector.shape,dtype=dtype)
		self.S_matrix_reg=1E-15*np.eye(self.F_vector.shape[0])
		self.nat_grad_guess=np.zeros_like(self.F_vector)

		self.O_expt=np.zeros(self.N_varl_params,dtype=dtype)
		self.OO_expt=np.zeros([self.N_varl_params,self.N_varl_params],dtype=dtype)
		self.O_expt2=np.zeros_like(self.OO_expt)

		self.E_diff_weighted=np.zeros(self.N_batch,dtype=dtype)


		self.S_norm=0.0 # S-matrix norm
		self.F_norm=0.0 # F norm
		self.S_logcond=0.0

		
		if self.comm.Get_rank()==0:
			self.S_lastiters=np.zeros([n_iter,self.N_varl_params,self.N_varl_params],dtype=dtype) # array to store last S-matrices
			self.F_lastiters=np.zeros([n_iter,self.N_varl_params,],dtype=dtype) # array to store last F-vectors
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
		

	
	def compute_S_matrix(self,Eloc_params_dict=None):
		
		if self.mode=='exact':
			abs_psi_2=Eloc_params_dict['abs_psi_2']#.copy()

			self.O_expt[:]=jnp.einsum('s,sj->j',abs_psi_2,self.dlog_psi).block_until_ready() 
			self.OO_expt[:] = jnp.dot(self.dlog_psi.T, jnp.dot(jnp.diag(abs_psi_2), self.dlog_psi)	).block_until_ready()  		

			
		elif self.mode=='MC':

			self.comm.Allreduce(jnp.sum(self.dlog_psi,axis=0).block_until_ready()._value, self.O_expt[:], op=MPI.SUM)
			self.O_expt/=self.N_MC_points

			self.comm.Allreduce((  jnp.dot(self.dlog_psi.T, self.dlog_psi).block_until_ready()  )._value, \
								self.OO_expt[:], op=MPI.SUM
								)

			self.OO_expt/=self.N_MC_points


		self.O_expt2[:] = jnp.outer(self.O_expt,self.O_expt)	
		self.S_matrix[:] = self.OO_expt - self.O_expt2 + self.S_matrix_reg

		
	

	def compute_F_vector(self,Eloc_params_dict=None):

		self.E_diff_weighted[:]=Eloc_params_dict['E_diff'].copy()

		if self.mode=='exact':
			abs_psi_2=Eloc_params_dict['abs_psi_2'].copy()
			self.E_diff_weighted*=abs_psi_2

			self.F_vector[:]   = jnp.dot(self.E_diff_weighted,self.dlog_psi).block_until_ready()
			
		elif self.mode=='MC':
		
			self.comm.Allreduce((  jnp.dot(self.E_diff_weighted, self.dlog_psi).block_until_ready() )._value, \
								self.F_vector[:], op=MPI.SUM
								)

			self.F_vector/=self.N_MC_points





	def compute_r2_cost(self,Eloc_params_dict):
		Eloc_var=Eloc_params_dict['Eloc_var']
		return ( (np.dot(self.nat_grad, np.dot(self.S_matrix,self.nat_grad)) - 2.0*np.dot(self.F_vector,self.nat_grad) + Eloc_var )/Eloc_var )

	def _S_matrix_checks(self):

		# import pickle
		# file_name='./bug'
		# with open(file_name+'.pkl', 'wb') as handle:
		# 	pickle.dump([self.dlog_psi, self.OO_expt, self.O_expt2, self.O_expt, self.S_matrix, self.F_vector], handle, protocol=pickle.HIGHEST_PROTOCOL)


		
		norm=jnp.linalg.norm(self.S_matrix).block_until_ready()

		
		if np.linalg.norm((self.S_matrix-self.S_matrix.T.conj())/norm ) > 1E-13: # and np.linalg.norm(self.S_matrix-self.S_matrix.T.conj()) > 1E-14:
		
			print('F : {:.20f}'.format(np.linalg.norm((self.S_matrix-self.S_matrix.T.conj())/norm )) )
			print('OO: {:.20f}'.format(np.linalg.norm( (self.OO_expt-self.OO_expt.T.conj())/np.linalg.norm(self.OO_expt) )) )
			print('O2: {:.20f}'.format(np.linalg.norm( (self.O_expt2-self.O_expt2.T.conj())/np.linalg.norm(self.O_expt2) )) )
			print('non-hermitian Fisher matrix with norm:', norm)
			
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


	def _TDVP_solver(self, S, F, nat_grad_guess, ):

		info=0

		if self.TDVP_opt == 'cg':
			self.nat_grad[:], info = cg(S,F,x0=nat_grad_guess,maxiter=self.cg_maxiter,atol=self.tol,tol=self.tol) # 
		
		elif self.TDVP_opt == 'inv':
			self.nat_grad[:]=jnp.dot(jnp.linalg.inv(S), F).block_until_ready()._value
		
		elif self.TDVP_opt == 'svd':
			lmbda, V = jnp.linalg.eigh(S/self.S_norm,)
			lmbda*=self.S_norm
			self.nat_grad[:] = jnp.dot(V ,  jnp.dot( np.diag(lmbda/(lmbda**2 + (self.tol)**2)), jnp.dot(V.T.conj(), F) ) )

		return info


	def compute(self,NN_params,batch,Eloc_params_dict,):
		

		self.dlog_psi[:]=self.compute_grad_log_psi(NN_params,batch,)
	
		self.compute_F_vector(Eloc_params_dict=Eloc_params_dict,)
		self.compute_S_matrix(Eloc_params_dict=Eloc_params_dict,)

		### compute natural_gradients using cg
		# regularize Fisher metric
		self.S_matrix += self.delta*np.diag(np.diag(self.S_matrix))
		#self.S_matrix += self.delta*np.linalg.norm(self.S_matrix)*np.eye(self.S_matrix.shape[0]) 

		self.debug_helper()


		####################################################### 
		
		if self.check_on:
			# check for symmetry and positivity
			self._S_matrix_checks()


		# compute norm
		self.S_norm=np.linalg.norm(self.S_matrix)
		self.F_norm=np.linalg.norm(self.F_vector)
		self.S_logcond=np.log(np.linalg.cond(self.S_matrix))



		#######################################################

		
		# solve TDVP EOM

		info=1
		while info>0 and self.cg_maxiter<1E5:
			
			info = self._TDVP_solver(self.S_matrix,self.F_vector, self.nat_grad_guess)
				
			# affects CG solver only
			if info>0:
				self.cg_maxiter*=2
				print('cg failed to converge in {0:d} iterations to tolerance {1:0.14f}; increasing maxiter to {2:d}'.format(info,self.tol,int(self.cg_maxiter)))
		
		
		###############

		# clip gradients
		self.nat_grad[:]=np.where(np.abs(self.nat_grad) < self.grad_clip, self.nat_grad, self.grad_clip) 



		# normalize gradients
		self.dE=2.0*np.dot(self.F_vector,self.nat_grad)

		# if not self.RK:
		# 	self.nat_grad /= np.sqrt(0.5*self.dE)
			
		return self.nat_grad
		
		

	def update_NG_params(self,grad_guess,step_size,self_time=1.0):

		
		if self.delta>self.tol:
			self.delta *= np.exp(-0.075*self_time)
		

		self.nat_grad_guess[:]=grad_guess
		self.dE*=step_size
		
		#self.iteration+=1


class Runge_Kutta_solver():

	def __init__(self,step_size, NN_Tree, return_grads, reestimate_local_energy, compute_r2):

		self.NN_Tree=NN_Tree
		self.return_grads=return_grads

		self.reestimate_local_energy=reestimate_local_energy
		self.compute_r2=compute_r2
		self.r2=0.0

		# RK params
		self.step_size=step_size
		self.time=0.0
		self.RK_tol=1E-4 # 5E-5 
		self.RK_inv_p=1.0/3.0
		
		self.counter=0
		self.iteration=0
			
		self.dy=np.zeros(NN_Tree.N_varl_params,dtype=np.float64)
		self.dy_star=np.zeros_like(self.dy)

		self.k1=np.zeros_like(self.dy)
		self.k2=np.zeros_like(self.dy)
		self.k3=np.zeros_like(self.dy)
		self.k4=np.zeros_like(self.dy)
		self.k5=np.zeros_like(self.dy)
		self.k6=np.zeros_like(self.dy)



	def run(self,NN_params,batch,Eloc_params_dict,):

		# flatten weights
		params=self.NN_Tree.ravel(NN_params,)
		
		max_param=jnp.max(jnp.abs(params)).block_until_ready()
		#max_param=jnp.max(np.abs(params[:32]+1j*params[32:]))

		initial_grad=self.return_grads(NN_params,batch,Eloc_params_dict,)
		self.r2=self.compute_r2(Eloc_params_dict)
		self.counter+=1

		error_ratio=0.0
		while error_ratio<1.0:

			### RK step 1
			self.k1[:]=-self.step_size*initial_grad
			
			### RK step 2
			NN_params_shifted=self.NN_Tree.unravel(params+self.k1)
			Eloc_params_dict = self.reestimate_local_energy(NN_params_shifted, batch, Eloc_params_dict)
			self.k2[:]=-self.step_size*self.return_grads(NN_params_shifted,batch,Eloc_params_dict,)
			self.dy[:]=0.5*self.k1 + 0.5*self.k2

			#######
			### RK step 1
			self.k3[:]=-0.5*self.step_size*initial_grad # 0.5*k1
			
			### RK step 2
			NN_params_shifted=self.NN_Tree.unravel(params+self.k3)
			Eloc_params_dict = self.reestimate_local_energy(NN_params_shifted, batch, Eloc_params_dict)
			self.k4[:]=-0.5*self.step_size*self.return_grads(NN_params_shifted,batch,Eloc_params_dict,)

			# first half-step solution difference
			self.dy_star[:]=0.5*self.k3 + 0.5*self.k4
			
			### RK step 1
			NN_params_shifted=self.NN_Tree.unravel(params+self.dy_star)
			Eloc_params_dict = self.reestimate_local_energy(NN_params_shifted, batch, Eloc_params_dict)
			self.k5[:]=-0.5*self.step_size*self.return_grads(NN_params_shifted,batch,Eloc_params_dict,)

			### RK step 2
			NN_params_shifted=self.NN_Tree.unravel(params+self.dy_star+self.k5)
			Eloc_params_dict = self.reestimate_local_energy(NN_params_shifted, batch, Eloc_params_dict)
			self.k6[:]=-0.5*self.step_size*self.return_grads(NN_params_shifted,batch,Eloc_params_dict,)

			# second half-step solution difference
			self.dy_star[:]+=0.5*self.k5 + 0.5*self.k6

			#######
			
			norm=np.max(np.abs(self.dy-self.dy_star))/max_param
			
			error_ratio=6.0*self.RK_tol/norm

			self.step_size*=min(2.0,max(0.2,0.9*error_ratio**self.RK_inv_p))
			
			self.counter+=4 # five gradient calculations


		# update params
		self.iteration+=1
		self.time+=self.step_size
		
		print('RK_steps={0:d}-step_size={1:0.15f}-time={2:0.4f}.\n'.format(self.counter, self.step_size, self.time,) )


		return self.dy_star # - 1.0/6.0*(self.dy-self.dy_star)


