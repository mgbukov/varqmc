from jax import jit, grad, vmap, random, ops, partial
from jax.config import config
config.update("jax_enable_x64", True)
from jax.experimental import optimizers
import jax.numpy as jnp
import numpy as np

seed=1
np.random.seed(seed)
np.random.RandomState(seed)
rng = random.PRNGKey(seed)

from mpi4py import MPI

symmetrized=True  # False  # 
#from models.RBM_real_symmetrized import *
from cpp_code import Neural_Net
from cpp_code import integer_to_spinstate

from natural_grad import natural_gradient
from MC_lib import MC_sampler
from energy_lib import Energy_estimator
from data_analysis import data


import time
np.set_printoptions(threshold=np.inf)

# mpiexec -n 2 python main.py 




def reshape_to_gradient_format(gradient,NN_dims,NN_shapes):
	NN_params=[]
	Ndims=np.insert(np.cumsum(NN_dims), 0, 0)
	# loop over network architecture
	for j in range(NN_dims.shape[0]): 
		NN_params.append( gradient[Ndims[j]:Ndims[j+1]].reshape(NN_shapes[j]) )
		
	return NN_params
	

def reshape_from_gradient_format(NN_params,NN_dims,NN_shapes):
	return jnp.concatenate([params.ravel() for params in NN_params])




class VMC(object):

	def __init__(self):

		# initialize communicator
		self.comm=MPI.COMM_WORLD


		self.L=4 # system size
		self.mode='MC'  # 'exact' #
		self.optimizer='RK' #'NG' #  'adam'  #

		self.save=False # True #
		load_data=False # True #
		self.plot_data=False
		
		# training params
		self.N_epochs=5 #500 

		### MC sampler
		self.N_MC_points=100 #107 #10000 #
		self.N_MC_chains = 1 # number of MC chains to run in parallel

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
		
		if load_data:
			print('exiting...')
			exit()

			# self.NN_params=jnp.array(
			# 	[ [0.021258599215, -0.0823963887505,
			# 	-0.0764654369726, -0.0628942940286,
			# 	0.00182637347543, -0.00695127347814,
			# 	-0.00525690483403, 0.0481080176332,
			# 	0.000980873958246, -0.00692652171121,
			# 	-0.0169162330746, -0.0106888594278,
			# 	0.00585653048371, 0.0138074306434,
			# 	-0.0478208906869, 0.0548194907154,
			# 	0.00420109717598, -0.0124403232342,
			# 	0.00567369833488, 0.0371481096711,
			# 	0.0306937197576, -0.0224021086711,
			# 	-0.00400841403652, -0.0300551229252,
			# 	0.0732902932636, 0.0458134104469,
			# 	-0.0668617051509, -0.0580895073338,
			# 	0.01761345991, -0.00145492616907,
			# 	0.0320281652266, 0.0310190200109,
			# 	0.0100540410205, 0.00611392133309,
			# 	-0.0253986890351, 0.035396303171,
			# 	0.0140308059331, 0.000374839619012,
			# 	-0.0188215528856, 0.01156958282,
			# 	-0.00130670195036, 0.0498983872269,
			# 	-0.0330274839124, -0.0123636849822,
			# 	0.0225649581684, -0.0167168862779,
			# 	0.0315205951684, -0.0684157061108,
			# 	-0.0428906561036, 0.0754985501526,
			# 	-0.0295073373148, -0.0631896535519,
			# 	-0.09089461634, -0.0290092409887,
			# 	0.00408861284419, 0.00340387882643,
			# 	-0.000359464136101, 0.0218156069361,
			# 	-0.00775859832165, -0.0618378944315,
			# 	0.0478057783988, -0.0637107169233,
			# 	0.0398047959674, -0.0134030969116]
			# 	]
			# 	)


			# self.NN_params=[self.NN_params.reshape(32,2,order='F')[0:32:2,].T,
			# 			  	self.NN_params.reshape(32,2,order='F')[1:32:2,].T
			# 			  	]
		
			self.NN_params=jnp.array(self.data_structure.load_weights()[0])
			self.NN_params=[W for W in self.NN_params]



			'''
			# Stot=0 eigenstate
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



		self.N_neurons=2
		shape=[self.N_neurons,self.L**2]
		#self.NN_params=create_NN(shape)

		### Neural network
		self.DNN=Neural_Net(shape,self.N_MC_chains)

		# jit functions
		self.evaluate_NN=jit(self.DNN.evaluate)
		#self.evaluate_NN=self.DNN.evaluate



	def _create_optimizer(self):


		@jit
		def loss_psi(NN_params,batch,):
			log_psi, phase_psi = self.evaluate_NN(NN_params,batch,)
			return jnp.sum(log_psi), jnp.sum(phase_psi)

		@jit
		def loss_log_psi(NN_params,batch,):
			log_psi, _ = self.evaluate_NN(NN_params,batch,)
			return jnp.sum(log_psi)


		@jit
		def loss_phase_psi(NN_params,batch,):
			_, phase_psi = self.evaluate_NN(NN_params,batch,)	
			return jnp.sum(phase_psi)

		@jit
		def compute_grad_log_psi(NN_params,batch,):

			# dlog_psi_s   = vmap(partial(grad(loss_log_psi),   NN_params))(batch, )
			# dphase_psi_s = vmap(partial(grad(loss_phase_psi), NN_params))(batch, )

			#dlog_psi_s   = vmap(jit(grad(loss_log_psi)),   in_axes=(None,0) )(NN_params,batch, )
			#dphase_psi_s = vmap(jit(grad(loss_phase_psi)), in_axes=(None,0) )(NN_params,batch, )

			dlog_psi_s, dphase_psi_s = vmap(jit(grad(loss_psi)), in_axes=(None,0), out_axes=(None,) )(NN_params,batch, )

	
			N_MC_points=dlog_psi_s[0].shape[0]

			return jnp.concatenate( [(dlog_psi+1j*dphase_psi).reshape(N_MC_points,-1) for (dlog_psi,dphase_psi) in zip(dlog_psi_s,dphase_psi_s)], axis=1  )



		### self.optimizer params
		# initiaize natural gradient class
			
		self.NG=natural_gradient(self.comm,self.N_MC_points,self.N_batch,self.DNN.N_varl_params,compute_grad_log_psi,
									reshape_to_gradient_format, reshape_from_gradient_format, self.DNN.dims, self.DNN.shapes)

		# jax self.optimizer
		if self.optimizer=='NG':
			step_size=5E-3
			self.opt_init, self.opt_update, self.get_params = optimizers.sgd(step_size=step_size)
			#self.opt_state = self.opt_init(self.NN_params)
			self.opt_state = self.opt_init(self.DNN.params)

		elif self.optimizer=='adam':
			step_size=1E-3
			self.opt_init, self.opt_update, self.get_params = optimizers.adam(step_size=step_size, b1=0.9, b2=0.99, eps=1e-08)
			if self.mode=='exact':

				@jit
				def loss_energy_exact(NN_params,batch,params_dict):
					log_psi, phase_psi = self.evaluate_NN(NN_params,batch,)
					return 2.0*jnp.sum(params_dict['abs_psi_2']*(log_psi*params_dict['E_diff'].real + phase_psi*params_dict['E_diff'].imag ))

				self.compute_grad=jit(grad(loss_energy_exact))
			
			elif self.mode=='MC':

				@jit
				def loss_energy_MC(NN_params,batch,params_dict,):
					log_psi, phase_psi = self.evaluate_NN(NN_params,batch,)
					return 2.0*jnp.sum(log_psi*params_dict['E_diff'].real + phase_psi*params_dict['E_diff'].imag)/params_dict['N_MC_points']



				self.compute_grad=jit(grad(loss_energy_MC))
			#self.opt_state = self.opt_init(self.NN_params)
			self.opt_state = self.opt_init(self.DNN.params)

		elif self.optimizer=='RK':
			step_size=1E-6
			self.NG.init_RK_params(step_size)

		self.step_size=step_size



	

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
							  NNstrct=tuple(tuple(shape) for shape in self.DNN.shapes),
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
			#self.get_training_data(self.NN_params)
			#self.get_Stot_data(self.NN_params)

			self.get_training_data(self.DNN.params)
			self.get_Stot_data(self.DNN.params)


			##### check c++ and python DNN evaluation
			if epoch==0:
				#self.MC_tool.check_consistency(self.evaluate_NN,self.NN_params,symmetrized)
				self.MC_tool.check_consistency(self.evaluate_NN,self.DNN.params,symmetrized)

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
				#self.data_structure.save(NN_params=self.NN_params)
				self.data_structure.save(NN_params=self.DNN.params)
				
			# plot data
			if self.plot_data:
				self.data_structure.compute_phase_hist()
				self.data_structure.plot(save=0)

	

	def get_training_data(self,NN_params):

		##### MC sample #####
		if self.mode=='exact':
			self.MC_tool.exact(NN_params, evaluate_NN=self.evaluate_NN)
		elif self.mode=='MC':
			self.MC_tool.sample(tuple([W._value for W in NN_params]) ,self.N_neurons, self.DNN)

		##### compute local energies #####
		self.E_est.compute_local_energy(self.evaluate_NN,NN_params,self.MC_tool.ints_ket,self.MC_tool.mod_kets,self.MC_tool.phase_kets,self.MC_tool.log_psi_shift)
			
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
		self.batch=self.MC_tool.spinstates_ket.reshape(-1,self.E_est.N_symms,self.E_est.N_sites)#.T


		return self.batch, self.params_dict

	def get_Stot_data(self,NN_params): 
		# check SU(2) conservation
		self.E_est.compute_local_energy(self.evaluate_NN,NN_params,self.MC_tool.ints_ket,self.MC_tool.mod_kets,self.MC_tool.phase_kets,self.MC_tool.log_psi_shift,SdotS=True)
		self.SdotSloc_mean, SdotS_var, SdotS_diff_real, SdotS_diff_imag = self.E_est.process_local_energies(mode=self.mode,params_dict=self.params_dict,SdotS=True)
		self.SdotS_MC_std=np.sqrt(SdotS_var/self.N_MC_points)


	def update_NN_params(self,epoch):

		if self.optimizer=='RK':
			# compute updated NN parameters
			#self.NN_params=self.NG.Runge_Kutta(self.NN_params,self.batch,self.params_dict,self.mode,self.get_training_data)
			self.DNN.update_params(self.NG.Runge_Kutta(self.DNN.params,self.batch,self.params_dict,self.mode,self.get_training_data))
			loss=self.NG.max_grads

		else:
			##### compute gradients
			if self.optimizer=='NG':
				# compute enatural gradients
				#grads=self.NG.compute(self.NN_params,self.batch,self.params_dict,mode=self.mode)
				grads=self.NG.compute(self.DNN.params,self.batch,self.params_dict,mode=self.mode)
				loss=self.NG.max_grads

			elif self.optimizer=='adam':
				
				# reshape
				if symmetrized:
					self.MC_tool.spinstates_ket_tot=self.MC_tool.spinstates_ket_tot.reshape(-1,self.E_est.N_symms,self.E_est.N_sites)
				else:
					self.MC_tool.spinstates_ket_tot=self.MC_tool.spinstates_ket_tot.reshape(-1,self.E_est.N_sites)
				batch=self.MC_tool.spinstates_ket_tot
				

				#grads=self.compute_grad(self.NN_params,batch,self.params_dict)
				grads=self.compute_grad(self.DNN.params,batch,self.params_dict)
				loss=[jnp.max([jnp.max(grads[j]) for j in range(self.DNN.shapes.shape[0])]).block_until_ready(),0.0]


			##### apply gradients
			self.opt_state = self.opt_update(epoch, grads, self.opt_state) 
			self.NG.update_params() # update NG params
			#self.NN_params=self.get_params(self.opt_state)
			self.DNN.update_params(self.get_params(self.opt_state))
			
		##### compute loss
		r2=self.NG.r2_cost

		return loss, r2



DNN_psi=VMC()
DNN_psi.train()



