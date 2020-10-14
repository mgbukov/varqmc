from mpi4py import MPI

from jax.config import config
config.update("jax_enable_x64", True)
from jax.experimental import optimizers
#from optimizers import sgd, adam
from jax import jit, grad, vmap, random, ops, partial, disable_jit
from jax import jacfwd, jacrev

import jax.numpy as jnp
import numpy as np

from natural_grad import natural_gradient, Runge_Kutta_solver

import time



class optimizer(object):

	def __init__(self, comm, opt, cost, mode, NN_dtype, NN_Tree, label='', step_size=1.0, adaptive_step=True, adaptive_SR_cutoff=False ):

		self.label=label

		self.comm=comm

		self.NN_dtype=NN_dtype
		
		self.opt=opt
		self.cost=cost
		self.mode=mode
		self.adaptive_step=adaptive_step
		self.adaptive_SR_cutoff=adaptive_SR_cutoff

		self.NN_Tree=NN_Tree

		self.logfile=''
		if self.opt=='RK':
			self.RK=True
		else:
			self.RK=False

		if self.opt=='NG' and self.cost=='energy':
			raise ValueError("NG incompatible with energy cost!")

		if self.opt=='adam' and self.cost=='SR':
			raise ValueError('adam incompatible with SR cost!')

		self.is_finite=True

		self.iteration=0
		self.step_size=step_size
		self.time=0.0


	

	def init_global_variables(self, N_MC_points, N_batch, N_varl_params, n_iter, N_minibatches):

		self.N_batch=N_batch
		self.N_MC_points=N_MC_points
		self.N_varl_params=N_varl_params
		self.n_iter=n_iter
		self.N_minibatches=N_minibatches


	def init_opt_state(self,NN_params):
		if self.opt == 'adam':
			self.opt_state = self.opt_init(NN_params)
		

	def _init_optimizer(self, reestimate_local_energy):
		
		if self.opt=='adam':
			self.opt_init, self.opt_update, self.get_params = optimizers.adam(step_size=self.step_size, b1=0.99, b2=0.999, eps=1e-08)
			
		elif self.opt=='sgd':
			#self.opt_init, self.opt_update, self.get_params = optimizers.sgd(step_size=self.step_size)
			self.opt_state=None
			self.opt_init=None
			self.get_params=None

		elif self.opt=='RK':

			if self.cost=='SR':
				compute_r2=self.NG.compute_r2_cost
				NG=self.NG
			else:
				compute_r2=lambda x: 0.0
				NG=None

			self.Runge_Kutta=Runge_Kutta_solver(self.step_size, self.NN_Tree, self.compute_grad, reestimate_local_energy, compute_r2, NG, self.adaptive_step)
			self.opt_state=None
			self.opt_init=None
			self.get_params=None
			

	
	def define_grad_func(self, NN_evaluate=None, NN_evaluate_log=None, NN_evaluate_phase=None, start_iter=0, TDVP_opt=None, reestimate_local_energy=None ):


		if self.cost=='energy':

			if self.mode=='MC':

				if self.NN_dtype=='real':
					#@partial(jit, static_argnums=(2,))
					@jit
					def loss(NN_params,batch,params_dict):
						prediction = NN_evaluate(NN_params,batch,)
						energy = 2.0*jnp.sum(prediction*params_dict['E_diff'])/params_dict['N_MC_points']
						return energy

				else:

					@jit
					def loss(NN_params,batch,params_dict):
						log_psi, phase_psi = NN_evaluate(NN_params,batch,)
						energy = 2.0*jnp.sum(log_psi*params_dict['E_diff'].real + phase_psi*params_dict['E_diff'].imag)/params_dict['N_MC_points']
						return energy

			elif self.mode in ['exact','ED']:	

				if self.NN_dtype=='real':
					#@partial(jit, static_argnums=(2,))
					@jit
					def loss(NN_params,batch,params_dict):
						prediction = NN_evaluate(NN_params,batch,)
						energy = 2.0*jnp.sum(params_dict['abs_psi_2']*(prediction*params_dict['E_diff']) )
						return energy			

				else: # cpx

					@jit
					def loss(NN_params,batch,params_dict):
						log_psi, phase_psi = NN_evaluate(NN_params,batch,)
						energy = 2.0*jnp.sum(params_dict['abs_psi_2']*(log_psi*params_dict['E_diff'].real+phase_psi*params_dict['E_diff'].imag) )
						return energy

			#grad_func=jit(grad(loss), static_argnums=(2,))
			grad_func=jit(grad(loss) )

			# construct MPI wrapper
			def compute_grad(NN_params,batch,params_dict):
				# compute adam gradients
				grads_MPI=self.NN_Tree.ravel( grad_func(NN_params,batch,params_dict,) )
				# sum up MPI processes
				grads=np.zeros_like(grads_MPI)
				self.comm.Allreduce(np.asarray(grads_MPI), grads,  op=MPI.SUM)
				return grads

			self.compute_grad=compute_grad


		elif self.cost=='SR':

			if self.NN_dtype=='real':

				@jit
				def loss_log(NN_params,batch,):
					prediction = NN_evaluate(NN_params,batch,)	
					return jnp.sum(prediction)

				@jit
				def grad_log(NN_params,batch,):

					dlog_s   = vmap(partial(jit(grad(loss_log)),   NN_params))(batch, )
					#dlog_s   = vmap(jit(grad(loss_log)), in_axes=(None, 0))(NN_params, batch)

					dlog = []
					for dlog_W in self.NN_Tree.flatten(dlog_s):
						dlog.append( dlog_W.reshape(batch.shape[0],-1) )

					return jnp.concatenate(dlog, axis=1)


				@jit
				def hessian(NN_params,batch,weights):

					Hess = vmap( jacfwd( grad(loss_log) ), in_axes=(None, 0)) (NN_params, batch)

					ddlog=[]
					for m, dlog_W in enumerate(self.NN_Tree.flatten(Hess)): # loop over m Hessian axis
						dlog=[]
						for n, dlog_W2 in enumerate(self.NN_Tree.flatten(dlog_W)): # loop over n Hessian axis
					
							dlog.append( dlog_W2.reshape(-1,self.NN_Tree.sizes[m], self.NN_Tree.sizes[n]) )
							#dlog.append( jnp.einsum('s,smn->mn',weights,dlog_W2.reshape(-1,self.NN_Tree.sizes[m], self.NN_Tree.sizes[n]) ) )

						ddlog.append(jnp.concatenate(dlog, axis=2))
						#ddlog.append( jnp.concatenate(dlog, axis=1) )

					return jnp.einsum('s,smn->mn',weights, jnp.concatenate(ddlog, axis=1) )
					#return jnp.concatenate(ddlog, axis=0)

					#return jnp.concatenate(ddlog, axis=1)
					

				@jit
				def loss_hessian(NN_params,batch,weights):
					prediction = NN_evaluate(NN_params,batch,)	
					#return jnp.sum(weights*prediction)
					return jnp.dot(weights,prediction)

				@jit
				def hessian2(NN_params,batch,weights,):

					#ti=time.time()

					Hess = jacfwd( grad(loss_hessian) ) (NN_params, batch, weights) # (12.31, 5.51, 27)
					#Hess = jacfwd( jacrev(loss_hessian) ) (NN_params, batch, weights) # (13.92, 5.67, 27)
					#Hess = jacrev( jacfwd(loss_hessian) ) (NN_params, batch, weights) # TOO SLOW

					#tf=time.time()
					#print('hessian took {0:0.3f}'.format(tf-ti) )


					ddlog=[]
					for m, dlog_W in enumerate(self.NN_Tree.flatten(Hess)): # loop over m Hessian axis
						dlog=[]
						for n, dlog_W2 in enumerate(self.NN_Tree.flatten(dlog_W)): # loop over n Hessian axis
							dlog.append( dlog_W2.reshape(-1,self.NN_Tree.sizes[m], self.NN_Tree.sizes[n]) )
						ddlog.append(jnp.concatenate(dlog, axis=2))

					return jnp.concatenate(ddlog, axis=1)




			else:

				@jit
				def loss_log(NN_params,batch,):
					log_psi = NN_evaluate_log(NN_params,batch,)	
					return jnp.sum(log_psi)

				@jit
				def loss_phase(NN_params,batch,):
					phase_psi = NN_evaluate_phase(NN_params,batch,)	
					return jnp.sum(phase_psi)

				@jit
				def grad_log(NN_params,batch,):

					dlog_s     = vmap(partial(jit(grad(loss_log)),   NN_params))(batch, )
					dphase_s   = vmap(partial(jit(grad(loss_phase)), NN_params))(batch, )

					# W_real=NN_params[0][0]
					# W_imag=NN_params[0][1]
					# W=W_real+1j*W_imag

					# s=batch[-1,...]

					# G=s[:,0].dot(np.tanh(s.dot(W[:,0]))) #, axis=0)

					# print(G.imag)
					# print( self.NN_Tree.flatten(dphase_s)[0][-1][0][0] )
					# print()
					# print(G.real)
					# print( self.NN_Tree.flatten(dlog_s)[0][-1][0][0] )				
						
					# #print( self.NN_Tree.flatten(dlog_s)[0][-1][0][0] +1j* self.NN_Tree.flatten(dphase_s)[0][-1][0][0] )
					
					# print(s.shape, W.shape, G.shape)
					
					# exit()
					
					dlog = []
					for dlog_W, dphase_W in zip(self.NN_Tree.flatten(dlog_s), self.NN_Tree.flatten(dphase_s)):
						dlog.append( (dlog_W + 1j*dphase_W).reshape(self.N_batch,-1) )

					return jnp.concatenate(dlog, axis=1)

			self.NG=natural_gradient(self.comm, grad_log, self.NN_dtype, self.NN_Tree, TDVP_opt, mode=self.mode, RK=self.RK, adaptive_SR_cutoff=self.adaptive_SR_cutoff, hessian=hessian, hessian2=hessian2 )
			self.NG.init_global_variables(self.N_MC_points,self.N_batch,self.N_varl_params,self.n_iter,self.N_minibatches,)
			
			self.compute_grad=self.NG.compute

		self._init_optimizer(reestimate_local_energy, )


	def return_grad(self, iteration, NN_params, batch, params_dict, ):

		print("\n"+self.label+":")

		# compute gradients
		if self.opt=='RK':
			grads=self.Runge_Kutta.run(NN_params,batch,params_dict.copy(),)
			NN_params_new=self.NN_Tree.unravel( self.NN_Tree.ravel(NN_params) + grads)

			if self.cost=='SR':
				self.NG.update_NG_params(grads,self_time=self.Runge_Kutta.time)
				self.is_finite = np.isfinite(self.NG.S_matrix).all() and np.isfinite(self.NG.F_vector).all()
				
				S_str="norm(S)={0:0.14f}, norm(F)={1:0.14f}, S_condnum={2:0.14f}".format(self.NG.S_norm, self.NG.F_norm, self.NG.S_logcond) 		
				if self.comm.Get_rank()==0:
					print(S_str)
				#self.logfile.write(S_str)

				# update all data which would've been messed up by the RK iterations
				r2=self.Runge_Kutta.r2
				self.NG.dE=self.Runge_Kutta.dE
				self.NG.S_eigvals[:]=self.Runge_Kutta.S_eigvals
				self.NG.VF_overlap[:]=self.Runge_Kutta.VF_overlap

				self.NG.F_norm=self.Runge_Kutta.F_norm
				self.NG.S_norm=self.Runge_Kutta.S_norm
				self.NG.Flog_norm=self.Runge_Kutta.Flog_norm
				self.NG.Fphase_norm=self.Runge_Kutta.Fphase_norm
				
				self.NG.SNR_exact[:]=self.Runge_Kutta.SNR_exact
				self.NG.SNR_gauss[:]=self.Runge_Kutta.SNR_gauss
				self.NG.SNR_weight_sum_exact=self.Runge_Kutta.SNR_weight_sum_exact
				self.NG.SNR_weight_sum_gauss=self.Runge_Kutta.SNR_weight_sum_gauss

			else:
				r2=0.0

			self.time+=self.Runge_Kutta.step_size
		
		elif self.opt=='sgd':
			grads=self.compute_grad(NN_params,batch,params_dict.copy(),)
			
			if self.cost=='SR':
				self.NG.update_NG_params(grads,) # update NG params
				self.is_finite = np.isfinite(self.NG.S_matrix).all() and np.isfinite(self.NG.F_vector).all()

				S_str=self.label+": norm(S)={0:0.14f}, norm(F)={1:0.14f}, S_condnum={2:0.14f}".format(self.NG.S_norm, self.NG.F_norm, self.NG.S_logcond) 		
				if self.comm.Get_rank()==0:
					print(S_str)
				#self.logfile.write(S_str)

				r2=self.NG.compute_r2_cost(params_dict.copy())
				self.NG.dE*=self.step_size
			else:
				r2=0.0

			# adaptive step size
			
			# max_grad=np.max(np.abs(grads))
			# if max_grad<0.1: # increase
			# 	print('increasing step size')
			# 	self.step_size*=min(1.01, 1.0/max_grad)
			# elif max_grad>50.0: # decrease step size
			# 	print('decreasing step size')
			# 	self.step_size*=0.9

			# print('step_size', self.step_size, max_grad, self.label)
			
			NN_params_new=self.NN_Tree.unravel(self.NN_Tree.ravel(NN_params) - self.step_size*grads )
			
			self.time+=self.step_size

		elif self.opt=='adam':
			grads=self.compute_grad(NN_params,batch,params_dict,)
			r2=0.0

			self.opt_state = self.opt_update(iteration, self.NN_Tree.unravel(grads) , self.opt_state)
			NN_params_new=self.get_params(self.opt_state)
			
			self.time+=self.step_size

		else:
			raise ValueError("unrecognized optimizer {}!".format(self.opt))
		
		#print(np.abs(grads))

		self.iteration+=1
		return NN_params_new, grads, r2
		







