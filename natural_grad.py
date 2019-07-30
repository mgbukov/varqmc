from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np
from scipy.sparse.linalg import cg
from scipy.linalg import eigh,eig

# seed=0
# np.random.seed(seed)
# np.random.RandomState(seed)
# rng = random.PRNGKey(seed)





class natural_gradient():

	def __init__(self,N_MC_points,N_varl_params,compute_grad_log_psi,
				 reshape_to_gradient_format,reshape_from_gradient_format,NN_dims, NN_shapes):

		dtype = np.float64

		self.N_MC_points=N_MC_points
		self.N_varl_params=N_varl_params
		self.compute_grad_log_psi=compute_grad_log_psi

		self.reshape_from_gradient_format=reshape_from_gradient_format
		self.reshape_to_gradient_format=reshape_to_gradient_format
		self.NN_dims=NN_dims
		self.NN_shapes=NN_shapes

		# preallocate memory
		
		self.dlog_psi=np.zeros([self.N_MC_points,N_varl_params],dtype=np.complex128)
		self.grad=np.zeros(N_varl_params,dtype=dtype)
		self.nat_grad=np.zeros_like(self.grad)
		self.current_grad_guess=np.zeros_like(self.grad)
		self.Fisher=np.zeros(2*self.grad.shape,dtype=dtype)
		self.nat_grad_guess=np.zeros_like(self.grad)

		
		# CG params
		self.RK_on=False
		self.tol=1E-6
		self.delta=100.0
		self.epoch=0
		self.r2_cost=0.0
		self.max_grads=0.0
	
	
	def compute_fisher_metric(self,mode='MC',params_dict=None):
		
		if mode=='exact':
			abs_psi_2=params_dict['abs_psi_2'].copy()

			O_expt=jnp.einsum('s,sj->j',abs_psi_2,self.dlog_psi) 

			OO_expt = jnp.einsum('s,sk,sl->kl',abs_psi_2,self.dlog_psi.real, self.dlog_psi.real) \
					 +jnp.einsum('s,sk,sl->kl',abs_psi_2,self.dlog_psi.imag, self.dlog_psi.imag)

		elif mode=='MC':

			O_expt=jnp.sum(self.dlog_psi,axis=0)/self.N_MC_points
			OO_expt = jnp.einsum('sk,sl->kl',self.dlog_psi.real, self.dlog_psi.real)/self.N_MC_points \
					 +jnp.einsum('sk,sl->kl',self.dlog_psi.imag, self.dlog_psi.imag)/self.N_MC_points

		self.Fisher[:] = OO_expt._value - (   jnp.einsum('k,l->kl',O_expt.real,O_expt.real)._value \
											+ jnp.einsum('k,l->kl',O_expt.imag,O_expt.imag)._value   )
		
		symm_check=np.linalg.norm((self.Fisher-self.Fisher.T.conj())/np.linalg.norm(self.Fisher) )
		if symm_check > 1E-14:
			print(symm_check)
			print(np.linalg.norm(OO_expt._value-OO_expt._value.T.conj()))
			print(np.linalg.norm(jnp.einsum('k,l->kl',O_expt.conj(),O_expt)._value-jnp.einsum('k,l->kl',O_expt.conj(),O_expt)._value.T.conj()))
			print('non-hermitian fisher matrix')
			
			F=np.outer(O_expt.conj(),O_expt)
			jF=jnp.outer(O_expt.conj(),O_expt)
			print(np.linalg.norm(F-F.T.conj()))
			print(np.linalg.norm(jF-jF.T.conj()))

			exit()
		
		E = eigh(self.Fisher,eigvals_only=True)
		E = E/(E.max()-E.min())
		#E,_ = eig(self.Fisher)
		if np.any(E <- 1E-14):
			print(E)
			print(np.abs(self.dlog_psi).max())
			print('negative eigenvalues')


			exit()
			
	def compute_gradients(self,mode='MC',params_dict=None):

		E_diff=params_dict['E_diff'].copy()
	
		if mode=='exact':
			abs_psi_2=params_dict['abs_psi_2'].copy()
			E_diff*=abs_psi_2
			self.grad[:] = jnp.dot(E_diff.real,self.dlog_psi.real) + jnp.dot(E_diff.imag,self.dlog_psi.imag) 

		elif mode=='MC':
			self.grad[:] = jnp.dot(E_diff.real,self.dlog_psi.real)/self.N_MC_points \
			             + jnp.dot(E_diff.imag,self.dlog_psi.imag)/self.N_MC_points



	def _compute_r2_cost(self,params_dict):
		Eloc_var=params_dict['Eloc_var']
		return (jnp.dot(self.nat_grad.conj(), jnp.dot(self.Fisher,self.nat_grad)) - 2.0*jnp.dot(self.grad.conj(),self.nat_grad).real + Eloc_var )/Eloc_var 


	def compute(self,NN_params,batch,params_dict,mode='MC',):
	
		self.dlog_psi[:]=self.compute_grad_log_psi(NN_params,batch)

		self.compute_fisher_metric(params_dict=params_dict,mode=mode)
		self.compute_gradients(params_dict=params_dict,mode=mode)

		### compute natural_gradients using cg
		# regularize Fisher metric
		self.Fisher += self.delta*np.diag(np.diag(self.Fisher)) #np.eye(self.Fisher.shape[0])
		# apply conjugate gradient
		self.nat_grad, info = cg(self.Fisher,self.grad,x0=self.nat_grad_guess,maxiter=1E4,atol=self.tol,tol=self.tol)
		if info>0:
			print('cg failed to converge in {0:d} iterations'.format(info))

		# store guess for next true
		self.current_grad_guess[:]=self.nat_grad

		# normalize gradients
		if not self.RK_on:
			self.r2_cost=self._compute_r2_cost(params_dict)
			self.max_grads=[jnp.max(np.abs(self.grad.real)), jnp.max(np.abs(self.nat_grad.real))]
			self.nat_grad /= jnp.sqrt(jnp.dot(self.grad.conj(),self.nat_grad).real)
		
			return self.reshape_to_gradient_format(self.nat_grad, self.NN_dims, self.NN_shapes)
		else:

			return self.nat_grad
		

	def update_params(self,self_time=1.0):

		if self.delta>5E-7:
			self.delta *= np.exp(-0.075*self_time)
		else:
			self.delta=0.0

		if self.tol>1E-7:
			 self.tol *= 0.95 #np.exp(-0.05*self_time)
	

		self.nat_grad_guess[:]=self.current_grad_guess[:]
		#print('delta={0:0.4f}'.format(self.delta))

		self.epoch+=1



	def _apply_gradients(self,params,learning_rate):

		return params[0] - learning_rate*self.nat_grads_tuple[0],   params[1] - learning_rate*self.nat_grads_tuple[1]


	def init_RK_params(self,learning_rate):

		self.RK_step_size=learning_rate
		self.RK_time=0.0
		self.RK_tol=1E-0*self.tol
		self.RK_inv_p=1.0/5.0
		self.counter=0
		self.delta=1E-3#0.001 # NG Fisher matrix regularization strength

		self.RK_on=True
		
		self.y=np.zeros_like(self.nat_grad)
		self.y_star=np.zeros_like(self.y)

		self.k1=np.zeros_like(self.y)
		self.k2=np.zeros_like(self.y)
		self.k3=np.zeros_like(self.y)
		self.k4=np.zeros_like(self.y)
		self.k5=np.zeros_like(self.y)
		self.k6=np.zeros_like(self.y)
		self.k7=np.zeros_like(self.y)

		# Constants for DOPRI algorithm
		self.RK_A11 = (1./5)
		self.RK_A21 = (3./40)
		self.RK_A22 = (9./40)
		self.RK_A31 = (44./45)
		self.RK_A32 = (-56./15)
		self.RK_A33 = (32./9)
		self.RK_A41 = (19372./6561)
		self.RK_A42 = (-25360./2187)
		self.RK_A43 = (64448./6561)
		self.RK_A44 = (-212./729)
		self.RK_A51 = (9017./3168)
		self.RK_A52 = (-355./33)
		self.RK_A53 = (46732./5247)
		self.RK_A54 = (49./176)
		self.RK_A55 = (-5103./18656)
		self.RK_A61 = (35./384)
		self.RK_A62 = (0.)
		self.RK_A63 = (500./1113)
		self.RK_A64 = (125./192)
		self.RK_A65 = (-2187./6784)
		self.RK_A66 = (11./84)
		self.RK_B21 = (5179./57600)
		self.RK_B22 = (0.)
		self.RK_B23 = (7571./16695)
		self.RK_B24 = (393./640)
		self.RK_B25 = (-92097./339200)
		self.RK_B26 = (187./2100)
		self.RK_B27 = (1./40)


	def Runge_Kutta(self,NN_params,batch,params_dict,mode,get_training_data):

		# flatten weights
		params=self.reshape_from_gradient_format(NN_params,self.NN_dims,self.NN_shapes)
		max_param=jnp.max(params)


		initial_grad=self.compute(NN_params,batch,params_dict,mode=mode)
		# cost and loss
		self.max_grads=[jnp.max(np.abs(self.grad.real)), jnp.max(np.abs(self.nat_grad.real))]	
		self.r2_cost=self._compute_r2_cost(params_dict)
	
		error_ratio=0.0
		while error_ratio<1.0:
			### RK step 1
			self.k1[:]=-self.RK_step_size*initial_grad
			
	
			### RK step 2
			self.y[:]=self.RK_A11*self.k1
			NN_params_shifted=self.reshape_to_gradient_format(params+self.y,self.NN_dims,self.NN_shapes)
			params_dict=get_training_data(NN_params_shifted)
			self.k2[:]=-self.RK_step_size*self.compute(NN_params_shifted,batch,params_dict,mode=mode)

			### RK step 3
			self.y[:]=self.RK_A21*self.k1+self.RK_A22*self.k2
			NN_params_shifted=self.reshape_to_gradient_format(params+self.y,self.NN_dims,self.NN_shapes)
			params_dict=get_training_data(NN_params_shifted)
			self.k3[:]=-self.RK_step_size*self.compute(NN_params_shifted,batch,params_dict,mode=mode)

			### RK step 4
			self.y[:]=self.RK_A31*self.k1+self.RK_A32*self.k2+self.RK_A33*self.k3
			NN_params_shifted=self.reshape_to_gradient_format(params+self.y,self.NN_dims,self.NN_shapes)
			params_dict=get_training_data(NN_params_shifted)
			self.k4[:]=-self.RK_step_size*self.compute(NN_params_shifted,batch,params_dict,mode=mode)
			
			### RK step 5
			self.y[:]=self.RK_A41*self.k1+self.RK_A42*self.k2+self.RK_A43*self.k3+self.RK_A44*self.k4
			NN_params_shifted=self.reshape_to_gradient_format(params+self.y,self.NN_dims,self.NN_shapes)
			params_dict=get_training_data(NN_params_shifted)
			# print('there', params_dict['abs_psi_2'] )
			# exit()
			self.k5[:]=-self.RK_step_size*self.compute(NN_params_shifted,batch,params_dict,mode=mode)
			#print('here')
			
			### RK step 6
			self.y[:]=self.RK_A51*self.k1+self.RK_A52*self.k2+self.RK_A53*self.k3+self.RK_A54*self.k4+self.RK_A55*self.k5
			NN_params_shifted=self.reshape_to_gradient_format(params+self.y,self.NN_dims,self.NN_shapes)
			params_dict=get_training_data(NN_params_shifted)
			self.k6[:]=-self.RK_step_size*self.compute(NN_params_shifted,batch,params_dict,mode=mode)

			### RK step 7
			self.y[:]=self.RK_A61*self.k1+self.RK_A62*self.k2+self.RK_A63*self.k3+self.RK_A64*self.k4+self.RK_A65*self.k5+self.RK_A66*self.k6
			NN_params_shifted=self.reshape_to_gradient_format(params+self.y,self.NN_dims,self.NN_shapes)
			params_dict=get_training_data(NN_params_shifted)
			self.k7[:]=-self.RK_step_size*self.compute(NN_params_shifted,batch,params_dict,mode=mode)

			self.y_star[:]=self.RK_B21*self.k1 \
						  +self.RK_B22*self.k2 \
						  +self.RK_B23*self.k3 \
						  +self.RK_B24*self.k4 \
						  +self.RK_B25*self.k5 \
						  +self.RK_B26*self.k6 \
						  +self.RK_B27*self.k7

			norm=jnp.max(jnp.abs(self.y-self.y_star))/max_param
			
			error_ratio=self.RK_tol/norm
			#error_ratio=self.RK_tol/jnp.max(jnp.abs((y-y_star)))
			#error_ratio=np.sqrt(NN_dims[0])*self.RK_tol/jnp.linalg.norm(y-y_star)
			
			self.RK_step_size*=min(2.0,max(0.2,0.9*error_ratio**(self.RK_inv_p)))
			
			
			self.counter+=7 # seven gradient calculations

			
		# update NG params
		self.update_params(self.RK_time)
		self.RK_time+=self.RK_step_size
		
		
		print('steps={0:d}-step_size={1:0.8f}-time={2:0.4f}-delta={3:0.10f}-RK_grad_norm-{4:0.12f}-cg_tol-{5:0.12f}'.format(self.counter, self.RK_step_size, self.RK_time, self.delta, self.RK_step_size*np.linalg.norm(params + self.y), self.tol) )
		
		return self.reshape_to_gradient_format(params + self.y, self.NN_dims, self.NN_shapes) 


