from jax import jit, grad 
from mpi4py import MPI

symmetrized=True  # False  # 
if symmetrized:
	from models.RBM_real_symmetrized import *
	#from models.RBM_cpx_symmetrized import *
else:
	from models.RBM_real import *

from natural_grad import natural_gradient
from MC_lib import MC_sampler
from energy_lib import Energy_estimator

from jax.config import config
config.update("jax_enable_x64", True)
from jax.experimental import optimizers

from cpp_code import integer_to_spinstate

import numpy as np
from functools import partial

from data_analysis import data

import time
np.set_printoptions(threshold=np.inf)

# mpiexec -n 2 python train_data_lib.py 





class VMC(object):

	def __init__(self):

		# initialize communicator
		self.comm=MPI.COMM_WORLD


		self.L=4 # system size
		self.mode='exact' #'MC'  # 
		self.optimizer='RK' # 'NG' # 'adam'  #

		self.save=False # True #
		load_data=False # True #
		self.plot_data=False
		
		# training params
		self.N_epochs=10 #500 

		### MC sampler
		self.N_MC_points=107 #10000 #
		# number of processors must fix MC sampling ratio
		if self.mode=='exact':
			self.N_batch=self.N_MC_points#
			if self.comm.Get_size()>1:
				print('only one core allowed for "exact" simulation')
				exit()
		else:
			self.N_batch=self.N_MC_points//self.comm.Get_size()

		
		if load_data:
			model_params=dict(model='RBMcpx',
							  mode=self.mode, 
							  symm=int(symmetrized),
							  L=self.L,
							  J2=0.5,
							  opt=self.optimizer,
							  NNstrct=((2,16),(2,16)),
							  epochs=self.N_epochs,
							  MCpts=self.N_MC_points,  
							)
			self._create_data_obj(model_params)
		
		self._create_NN(load_data=load_data)
		self._create_optimizer()
		self._create_energy_estimator()
		self._create_MC_sampler()
		if not load_data:
			self._create_data_obj()
		
		


	def _create_NN(self, load_data=False):
		### Neural network 

		if load_data:

			self.NN_params=jnp.array(self.data_structure.load_weights()[0])
			self.NN_params=[W for W in self.NN_params]


			'''
			self.NN_params=[ jnp.array(
						[[ 0.00580084, -0.07497671, -0.06789109, -0.01204818, -0.01274435, -0.10384455,
						  -0.10245288, -0.04374218, -0.08915268,  0.00177797, -0.04978558, -0.05587124,
						   0.07693333, -0.07984009, -0.01530174, -0.12691828],
						 [-0.21518007, -0.11389974, -0.20263269, -0.12619203, -0.08462431, -0.20707012,
						  -0.17650112, -0.19922243, -0.16967753, -0.18817129, -0.20458634, -0.08676682,
						  -0.11785064, -0.03068135, -0.04197004, -0.17183342]]

					  	),

						jnp.array(
						[[-0.00231961, -0.00895621,  0.06853679,  0.0069357,  -0.02742392,  0.00391063,
						   0.02143496,  0.07258342,  0.05560984, -0.02823264,  0.02835682, -0.00693428,
						   0.01221359,  0.05616201, -0.02058408,  0.0281348 ],
						 [-0.1803375,  -0.12494415, -0.17192932, -0.11660786, -0.12339658, -0.18764674,
						  -0.12699878, -0.17132069, -0.15470591, -0.13720351, -0.18281736, -0.10516087,
						  -0.10887956, -0.07979148, -0.12417864, -0.16585421]]
					  	)
					 ]
			'''


		else:

			self.N_neurons=2
			self.NN_params=create_NN([self.N_neurons,self.L**2])
			

			# N_neurons_fc1=2
			# N_neurons_fc2=4
			# shape=[[N_neurons_fc1,self.L**2],[N_neurons_fc1,N_neurons_fc2],[N_neurons_fc1,N_neurons_fc2]]
			# self.NN_params=create_NN(shape)

		
		self.NN_shapes=np.array([W.shape for W in self.NN_params])
		self.NN_dims=np.array([np.prod(shape) for shape in self.NN_shapes])
		self.N_varl_params=self.NN_dims.sum()

	
		

	def _create_optimizer(self):

		### self.optimizer params
		# initiaize natural gradient class
			
		self.NG=natural_gradient(self.comm,self.N_MC_points,self.N_batch,self.N_varl_params,compute_grad_log_psi,
									reshape_to_gradient_format, reshape_from_gradient_format, self.NN_dims, self.NN_shapes)

		# jax self.optimizer
		if self.optimizer=='NG':
			step_size=1E-2
			self.opt_init, self.opt_update, self.get_params = optimizers.sgd(step_size=step_size)
			self.opt_state = self.opt_init(self.NN_params)

		elif self.optimizer=='adam':
			step_size=1E-3
			self.opt_init, self.opt_update, self.get_params = optimizers.adam(step_size=step_size, b1=0.9, b2=0.99, eps=1e-08)
			if self.mode=='exact':
				self.compute_grad=jit(grad(loss_energy_exact))
			elif self.mode=='MC':
				self.compute_grad=jit(grad(loss_energy_MC))
			self.opt_state = self.opt_init(self.NN_params)

		elif self.optimizer=='RK':
			step_size=1E-6
			self.NG.init_RK_params(step_size)



	

	def _create_energy_estimator(self):
		### Energy estimator
		self.E_est=Energy_estimator(self.comm,self.N_MC_points,self.N_batch,self.L,symmetrized=symmetrized) # contains all of the physics
		self.E_est.init_global_params()
		self.N_features=self.E_est.N_sites*self.E_est.N_symms

	def _create_MC_sampler(self):
		### initialize MC sampler variables
		build_dict_args=(self.E_est.sign,self.L,self.E_est.J2)
		self.MC_tool=MC_sampler(self.comm,build_dict_args=build_dict_args)
		self.MC_tool.init_global_vars(self.L,self.N_batch,self.E_est.N_symms,self.E_est.basis_type)


	def _create_data_obj(self,model_params=None):
		### initialize data class
		if model_params is None:
			self.model_params=dict(model='RBMcpx',
							  mode=self.mode, 
							  symm=int(symmetrized),
							  L=self.L,
							  J2=self.E_est.J2,
							  opt=self.optimizer,
							  NNstrct=tuple(tuple(shape) for shape in self.NN_shapes),
							  epochs=self.N_epochs,
							  MCpts=self.N_MC_points,
							  
							)
		else:
			self.model_params=model_params

		extra_label=''#'-unique_configs'
		self.data_structure=data(self.model_params,self.N_MC_points,self.N_epochs,extra_label=extra_label)



	def train(self):

		if self.mode=='exact':
			self.MC_tool.ints_ket, self.index, self.inv_index, self.count=self.E_est.get_exact_kets()
			integer_to_spinstate(self.MC_tool.ints_ket, self.MC_tool.spinstates_ket, self.N_features)


		for epoch in range(self.N_epochs): 

			
			ti=time.time()

			##### evaluate model
			self.get_training_data(self.NN_params)
			self.get_Stot_data(self.NN_params)


			##### check c++ and python DNN evaluation
			if epoch==0:
				self.MC_tool.check_consistency(evaluate_NN,self.NN_params,symmetrized)

				if self.mode=='exact':
					np.testing.assert_allclose(self.Eloc_mean_g.real, self.E_est.H.expt_value(self.psi[self.inv_index]))

				if self.comm.Get_rank()==0:
					print('cpp/python consistency check passed!\n')

			#####		
			if self.comm.Get_rank()==0:
				print("epoch {0:d}:".format(epoch))
				print("E={0:0.14f} and SS={1:0.14f}.".format(self.Eloc_mean_g.real, self.SdotSloc_mean.real ), self.E_MC_std 	)
				if self.mode=='exact':
					print('overlap', self.params_dict['overlap'] )


			##### combine results from all cores
			self.MC_tool.Allgather()	


			#### update model parameters
			if epoch<self.N_epochs-1:
				loss, r2 = self.update_NN_params(epoch)


			print("process_rank {0:d} calculation took {1:0.4f}secs.\n".format(self.comm.Get_rank(),time.time()-ti) )


			##### store data
			if self.comm.Get_rank()==0:
				self.data_structure.excess_energy[epoch,0]=(self.Eloc_mean_g - self.E_est.E_GS)/self.E_est.N_sites
				self.data_structure.excess_energy[epoch,1]=self.E_MC_std/self.E_est.N_sites
				self.data_structure.SdotS[epoch,0]=self.SdotSloc_mean/(0.5*self.E_est.N_sites*(0.5*self.E_est.N_sites+1.0))
				self.data_structure.SdotS[epoch,1]=self.SdotS_MC_std/(0.5*self.E_est.N_sites*(0.5*self.E_est.N_sites+1.0))
				self.data_structure.loss[epoch,:]=loss
				self.data_structure.r2[epoch]=r2
				self.data_structure.phase_psi[epoch,:]=self.MC_tool.phase_kets_tot
				self.data_structure.mod_psi[epoch,:]=self.MC_tool.mod_kets_tot



		if self.comm.Get_rank()==0:

	
			# save data
			if self.save:
				self.data_structure.save(NN_params=self.NN_params)
				
			# plot data
			if self.plot_data:
				self.data_structure.compute_phase_hist()
				self.data_structure.plot(save=0)

	

	def get_training_data(self,NN_params):

		##### MC sample #####
		if self.mode=='exact':
			self.MC_tool.exact(NN_params, evaluate_NN=evaluate_NN)
		elif self.mode=='MC':
			self.MC_tool.sample(tuple([W._value for W in NN_params]) ,self.N_neurons)

		##### compute local energies #####
		self.E_est.compute_local_energy(evaluate_NN,NN_params,self.MC_tool.ints_ket,self.MC_tool.mod_kets,self.MC_tool.phase_kets,self.MC_tool.log_psi_shift)
			
		if self.mode=='exact':
			#print(self.MC_tool.mod_kets)
			self.psi = self.MC_tool.mod_kets*np.exp(+1j*self.MC_tool.phase_kets)/np.linalg.norm(self.MC_tool.mod_kets[self.inv_index])
			abs_psi_2=self.count*np.abs(self.psi)**2
			#print(abs_psi_2)
			#exit()
			self.params_dict=dict(abs_psi_2=abs_psi_2,)
			overlap=np.abs(self.psi[self.inv_index].dot(self.E_est.psi_GS_exact))**2
			self.params_dict['overlap']=overlap
		elif self.mode=='MC':
			self.params_dict=dict(N_MC_points=self.N_MC_points)
		
		self.Eloc_mean_g, self.Eloc_var_g, E_diff_real, E_diff_imag = self.E_est.process_local_energies(mode=self.mode,params_dict=self.params_dict)
		self.Eloc_std=np.sqrt(self.Eloc_var_g)
		self.E_MC_std=self.Eloc_std/np.sqrt(self.N_MC_points)
		self.params_dict['E_diff']=E_diff_real+1j*E_diff_imag
		self.params_dict['Eloc_mean']=self.Eloc_mean_g
		self.params_dict['Eloc_var']=self.Eloc_var_g



		##### total batch
		self.batch=self.MC_tool.spinstates_ket.reshape(-1,self.E_est.N_symms,self.E_est.N_sites)


		return self.batch, self.params_dict

	def get_Stot_data(self,NN_params): 
		# check SU(2) conservation
		self.E_est.compute_local_energy(evaluate_NN,NN_params,self.MC_tool.ints_ket,self.MC_tool.mod_kets,self.MC_tool.phase_kets,self.MC_tool.log_psi_shift,SdotS=True)
		self.SdotSloc_mean, SdotS_var, SdotS_diff_real, SdotS_diff_imag = self.E_est.process_local_energies(mode=self.mode,params_dict=self.params_dict,SdotS=True)
		self.SdotS_MC_std=np.sqrt(SdotS_var/self.N_MC_points)


	def update_NN_params(self,epoch):

		if self.optimizer=='RK':
			# compute updated NN parameters
			self.NN_params=self.NG.Runge_Kutta_2(self.NN_params,self.batch,self.params_dict,self.mode,self.get_training_data)
			loss=self.NG.max_grads

		else:
			##### compute gradients
			if self.optimizer=='NG':
				# compute enatural gradients
				grads=self.NG.compute(self.NN_params,self.batch,self.params_dict,mode=self.mode)
				loss=self.NG.max_grads
				
			elif self.optimizer=='adam':
				
				# reshape
				if symmetrized:
					self.MC_tool.spinstates_ket_tot=self.MC_tool.spinstates_ket_tot.reshape(-1,self.E_est.N_symms,self.E_est.N_sites)
				else:
					self.MC_tool.spinstates_ket_tot=self.MC_tool.spinstates_ket_tot.reshape(-1,self.E_est.N_sites)
				batch=self.MC_tool.spinstates_ket_tot
				

				grads=self.compute_grad(self.NN_params,batch,self.params_dict)
				loss=[jnp.max([jnp.max(grads[j]) for j in range(self.NN_shapes.shape[0])]),0.0]


			##### apply gradients
			self.opt_state = self.opt_update(epoch, grads, self.opt_state) 
			self.NG.update_params() # update NG params
			self.NN_params=self.get_params(self.opt_state)
			
		##### compute loss
		r2=self.NG.r2_cost

		return loss, r2



DNN_psi=VMC()
DNN_psi.train()



