from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit

from mpi4py import MPI
import numpy as np
from scipy.sparse.linalg import cg
from scipy.linalg import eigh,eig





class natural_gradient():

	def __init__(self,comm,N_MC_points,N_batch,N_varl_params,compute_grad_log_psi,
				 reshape_to_gradient_format,reshape_from_gradient_format,NN_dims, NN_shapes):

		self.comm=comm

		dtype = np.float64

		self.N_batch=N_batch
		self.N_MC_points=N_MC_points
		self.N_varl_params=N_varl_params
		self.compute_grad_log_psi=compute_grad_log_psi

		self.reshape_from_gradient_format=reshape_from_gradient_format
		self.reshape_to_gradient_format=reshape_to_gradient_format
		self.NN_dims=NN_dims
		self.NN_shapes=NN_shapes

		# preallocate memory
		
		self.dlog_psi=np.zeros([self.N_batch,N_varl_params],dtype=np.complex128)
		self.grad=np.zeros(N_varl_params,dtype=dtype)
		self.nat_grad=np.zeros_like(self.grad)
		self.current_grad_guess=np.zeros_like(self.grad)
		self.Fisher=np.zeros(2*self.grad.shape,dtype=dtype)
		self.nat_grad_guess=np.zeros_like(self.grad)

		self.O_expt=np.zeros(N_varl_params,dtype=np.complex128)
		self.OO_expt=np.zeros([N_varl_params,N_varl_params],dtype=np.float64)

		
		# CG params
		self.RK_on=False
		self.tol=1E-7 # CG tolerance
		self.delta=50.0
		self.epoch=0
		self.r2_cost=0.0
		self.max_grads=0.0
	
	
	def compute_fisher_metric(self,mode='MC',params_dict=None):
		
		if mode=='exact':
			abs_psi_2=params_dict['abs_psi_2']#.copy()

			self.O_expt[:]=jnp.einsum('s,sj->j',abs_psi_2,self.dlog_psi).block_until_ready() 

			self.OO_expt[:] = jnp.einsum('s,sk,sl->kl',abs_psi_2,self.dlog_psi.real, self.dlog_psi.real).block_until_ready() \
							 +jnp.einsum('s,sk,sl->kl',abs_psi_2,self.dlog_psi.imag, self.dlog_psi.imag).block_until_ready()

		elif mode=='MC':

			# self.O_expt[:]=jnp.sum(self.dlog_psi,axis=0)

			# OO_expt = jnp.einsum('sk,sl->kl',self.dlog_psi.real, self.dlog_psi.real)/self.N_MC_points \
			# 		 +jnp.einsum('sk,sl->kl',self.dlog_psi.imag, self.dlog_psi.imag)/self.N_MC_points


			self.comm.Allreduce(jnp.sum(self.dlog_psi,axis=0).block_until_ready()._value, self.O_expt, op=MPI.SUM)
			self.O_expt/=self.N_MC_points



			self.comm.Allreduce( ( jnp.einsum('sk,sl->kl',self.dlog_psi.real, self.dlog_psi.real) \
				                  +jnp.einsum('sk,sl->kl',self.dlog_psi.imag, self.dlog_psi.imag)   ).block_until_ready()._value, \
								self.OO_expt, op=MPI.SUM
								)
			self.OO_expt/=self.N_MC_points


		O_expt2=jnp.einsum('k,l->kl',self.O_expt.real,self.O_expt.real).block_until_ready() + jnp.einsum('k,l->kl',self.O_expt.imag,self.O_expt.imag).block_until_ready()

		self.Fisher[:] = (self.OO_expt - O_expt2)#._value		
		
		norm=jnp.linalg.norm(self.Fisher).block_until_ready()

		#print(self.dlog_psi)
		#print(np.linalg.norm(abs_psi_2), np.linalg.norm(OO_expt), np.linalg.norm(O_expt2))

		#print(np.linalg.norm(self.dlog_psi), norm, np.linalg.norm((self.Fisher-self.Fisher.T.conj())/norm ), np.linalg.norm(self.Fisher-self.Fisher.T.conj()) )
	
		if np.linalg.norm((self.Fisher-self.Fisher.T.conj())/norm ) > 1E-14: # and np.linalg.norm(self.Fisher-self.Fisher.T.conj()) > 1E-14:
		
			print('F : {:.20f}'.format(np.linalg.norm((self.Fisher-self.Fisher.T.conj())/norm )) )
			print('OO: {:.20f}'.format(np.linalg.norm( (OO_expt._value-OO_expt._value.T.conj())/np.linalg.norm(OO_expt) )) )
			print('O2: {:.20f}'.format(np.linalg.norm( (O_expt2._value-O_expt2._value.T.conj())/np.linalg.norm(O_expt2) )) )
			print('non-hermitian fisher matrix')
			
			np.testing.assert_allclose(self.Fisher/norm,self.Fisher.T.conj()/norm, rtol=1E-14, atol=1E-14)

			exit()
		
		E = eigh(self.Fisher/norm,eigvals_only=True)
		# E2 = eigh(self.Fisher/norm,eigvals_only=True)
		# E2 = E2/(E2.max()-E2.min())
		
		# E,_ = eig(self.Fisher/np.linalg.norm(self.Fisher))
		# E=np.sort(E)
		
		if np.any(E <- 1E-14):
			print('E ', E)
			print()
			print('OO',eigh(OO_expt._value/np.linalg.norm(OO_expt._value),eigvals_only=True))
			print()
			print('O2',eigh(O_expt2._value/np.linalg.norm(O_expt2._value),eigvals_only=True))

			print('negative eigenvalues')


			exit()
			
	def compute_gradients(self,mode='MC',params_dict=None):

		E_diff=params_dict['E_diff'].copy()
				
	
		if mode=='exact':
			abs_psi_2=params_dict['abs_psi_2'].copy()
			E_diff*=abs_psi_2
			self.grad[:] = jnp.dot(E_diff.real,self.dlog_psi.real).block_until_ready() + jnp.dot(E_diff.imag,self.dlog_psi.imag).block_until_ready()

		elif mode=='MC':
			# self.grad[:] = jnp.dot(E_diff.real,self.dlog_psi.real)/self.N_MC_points \
			#              + jnp.dot(E_diff.imag,self.dlog_psi.imag)/self.N_MC_points

			self.comm.Allreduce((  jnp.dot(E_diff.real,self.dlog_psi.real) \
			             		 + jnp.dot(E_diff.imag,self.dlog_psi.imag) ).block_until_ready()._value, \
								self.grad[:], op=MPI.SUM
								)
			self.grad/=self.N_MC_points


	def _compute_r2_cost(self,params_dict):
		Eloc_var=params_dict['Eloc_var']
		return (jnp.dot(self.nat_grad.conj(), jnp.dot(self.Fisher,self.nat_grad)).block_until_ready() - 2.0*jnp.dot(self.grad.conj(),self.nat_grad).block_until_ready().real + Eloc_var )/Eloc_var 


	def compute(self,NN_params,batch,params_dict,mode='MC',):
		
		self.dlog_psi[:]=self.compute_grad_log_psi(NN_params,batch)

		self.compute_fisher_metric(params_dict=params_dict,mode=mode)
		self.compute_gradients(params_dict=params_dict,mode=mode)

		#print(self.grad)
		

		### compute natural_gradients using cg
		# regularize Fisher metric
		self.Fisher += self.delta*np.diag(np.diag(self.Fisher)) #np.eye(self.Fisher.shape[0])
		
		# apply conjugate gradient
		self.nat_grad, info = cg(self.Fisher,self.grad,x0=self.nat_grad_guess,maxiter=1E4,atol=self.tol,tol=self.tol)
		if info>0:
			print('cg failed to converge in {0:d} iterations'.format(info))
			exit()

	
		# store guess for next true
		self.current_grad_guess[:]=self.nat_grad

		# normalize gradients
		if not self.RK_on:
			self.r2_cost=self._compute_r2_cost(params_dict)
			self.max_grads=[jnp.max(jnp.abs(self.grad.real)).block_until_ready(), jnp.max(jnp.abs(self.nat_grad.real)).block_until_ready()]
			self.nat_grad /= jnp.sqrt(jnp.dot(self.grad.conj(),self.nat_grad).real).block_until_ready()
		
			
			return self.reshape_to_gradient_format(self.nat_grad, self.NN_dims, self.NN_shapes)
		else:
			return self.nat_grad
		

	def update_params(self,self_time=1.0):

		if self.delta>5E-5:
			#self.delta *= np.exp(-0.075*self_time)
			self.delta *= np.exp(-0.075)
		else:
			self.delta=0.0

		# if self.tol>1E-7:
		# 	 self.tol *= 0.95 #np.exp(-0.05*self_time)
	

		self.nat_grad_guess[:]=self.current_grad_guess[:]
		#print('delta={0:0.4f}'.format(self.delta))

		self.epoch+=1



	def _apply_gradients(self,params,learning_rate):

		return params[0] - learning_rate*self.nat_grads_tuple[0],   params[1] - learning_rate*self.nat_grads_tuple[1]


	def init_RK_params(self,learning_rate):

		self.RK_step_size=learning_rate
		self.RK_time=0.0
		self.RK_tol=5E-5 #self.tol
		self.RK_inv_p=1.0/3.0
		self.counter=0
		self.delta=0.001 # NG Fisher matrix regularization strength

		self.RK_on=True
		
		self.dy=np.zeros_like(self.nat_grad)
		self.dy_star=np.zeros_like(self.dy)

		self.k1=np.zeros_like(self.dy)
		self.k2=np.zeros_like(self.dy)
		self.k3=np.zeros_like(self.dy)
		self.k4=np.zeros_like(self.dy)
		self.k5=np.zeros_like(self.dy)
		self.k6=np.zeros_like(self.dy)



	def Runge_Kutta(self,NN_params,batch,params_dict,mode,get_training_data):

		# flatten weights
		params=self.reshape_from_gradient_format(NN_params,self.NN_dims,self.NN_shapes)
		
		max_param=jnp.max(jnp.abs(params)).block_until_ready()
		#max_param=jnp.max(np.abs(params[:32]+1j*params[32:]))

		initial_grad=self.compute(NN_params,batch,params_dict,mode=mode)
		# cost and loss
		self.max_grads=[jnp.max(jnp.abs(self.grad.real)).block_until_ready(), jnp.max(jnp.abs(self.nat_grad.real)).block_until_ready()]	
		self.r2_cost=self._compute_r2_cost(params_dict)
	
		error_ratio=0.0
		while error_ratio<1.0:

			### RK step 1
			self.k1[:]=-self.RK_step_size*initial_grad
			
			### RK step 2
			NN_params_shifted=self.reshape_to_gradient_format(params+self.k1,self.NN_dims,self.NN_shapes)
			batch, params_dict=get_training_data(NN_params_shifted)
			self.nat_grad_guess[:]=self.k1/self.RK_step_size
			self.k2[:]=-self.RK_step_size*self.compute(NN_params_shifted,batch,params_dict,mode=mode)
			self.dy[:]=0.5*self.k1 + 0.5*self.k2


			#######
			### RK step 1
			self.k3[:]=-0.5*self.RK_step_size*initial_grad # 0.5*k1
			
			### RK step 2
			NN_params_shifted=self.reshape_to_gradient_format(params+self.k3,self.NN_dims,self.NN_shapes)
			batch, params_dict=get_training_data(NN_params_shifted)
			self.nat_grad_guess[:]=self.k3/(0.5*self.RK_step_size)
			self.k4[:]=-0.5*self.RK_step_size*self.compute(NN_params_shifted,batch,params_dict,mode=mode)

			# first half-step solution difference
			self.dy_star[:]=0.5*self.k3 + 0.5*self.k4
			
			### RK step 1
			NN_params_shifted=self.reshape_to_gradient_format(params+self.dy_star,self.NN_dims,self.NN_shapes)
			batch, params_dict=get_training_data(NN_params_shifted)
			self.nat_grad_guess[:]=self.k4/(0.5*self.RK_step_size)
			self.k5[:]=-0.5*self.RK_step_size*self.compute(NN_params_shifted,batch,params_dict,mode=mode)

			### RK step 2
			NN_params_shifted=self.reshape_to_gradient_format(params+self.dy_star+self.k5,self.NN_dims,self.NN_shapes)
			batch, params_dict=get_training_data(NN_params_shifted)
			self.nat_grad_guess[:]=self.k5/(0.5*self.RK_step_size)
			self.k6[:]=-0.5*self.RK_step_size*self.compute(NN_params_shifted,batch,params_dict,mode=mode)

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
		self.update_params(self.RK_time)
		self.RK_time+=self.RK_step_size
		
		
		print('steps={0:d}-step_size={1:0.12f}-time={2:0.4f}-delta={3:0.10f}-RK_grad_norm-{4:0.12f}-cg_tol-{5:0.12f}'.format(self.counter, self.RK_step_size, self.RK_time, self.delta, self.RK_step_size*np.linalg.norm(params + self.dy), self.tol) )
	
		return self.reshape_to_gradient_format(params + self.dy_star - 1.0/6.0*(self.dy-self.dy_star), self.NN_dims, self.NN_shapes)

