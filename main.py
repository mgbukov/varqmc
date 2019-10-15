import sys,os

os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel

#quspin_path = os.path.join(os.path.expanduser('~'),"quspin/QuSpin/")
quspin_path = os.path.join(os.path.expanduser('~'),"quspin/QuSpin_dev/")
sys.path.insert(0,quspin_path)


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


from cpp_code import Neural_Net
from cpp_code import integer_to_spinstate

from natural_grad import natural_gradient
from MC_lib import MC_sampler
from energy_lib import Energy_estimator
from data_analysis import data


import time
np.set_printoptions(threshold=np.inf)

# export KMP_DUPLICATE_LIB_OK=TRUE
# export OMP_NUM_THREADS=4
# mpiexec -n 2 python main.py 






class VMC(object):

	def __init__(self):

		# initialize communicator
		self.comm=MPI.COMM_WORLD


		self.L=4 # system size
		self.mode='exact'  #'MC'  #
		self.optimizer='RK' # 'NG' #'adam' # 
		self.NN_type='DNN' # 'CNN' #
		 

		self.save=False # True #
		load_data=False # True # 
		self.plot_data=False #True # 
		
		# training params
		self.N_epochs=5 #500 

		### MC sampler
		self.N_MC_points=107 #10000 #
		self.N_MC_chains = 1 # number of MC chains to run in parallel
		os.environ['OMP_NUM_THREADS']='{0:d}'.format(self.N_MC_chains) # set number of OpenMP threads to run in parallel


		# number of processors must fix MC sampling ratio
		if self.mode=='exact':
			self.N_batch=self.N_MC_points#
			if self.comm.Get_size()>1:
				print('only one core allowed for "exact" simulation')
				exit()
		else:
			if self.N_MC_points//self.N_MC_chains != self.N_MC_points/self.N_MC_chains:
				print('number of MC chains incompatible with the total number of points.')
				exit()
			self.N_batch=self.N_MC_points//self.comm.Get_size()

		
		if load_data:
			model_params=dict(model='RBMcpx',
							  mode=self.mode,
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

		N_neurons=4
		shapes=([N_neurons,self.L**2], )
		
		### Neural network
		self.DNN=Neural_Net(shapes, self.N_MC_chains, self.NN_type)
		
		if load_data:
			#print('exiting...')
			#exit()

			'''
			NN_params=jnp.array(
				[ [0.021258599215, -0.0823963887505,
				-0.0764654369726, -0.0628942940286,
				0.00182637347543, -0.00695127347814,
				-0.00525690483403, 0.0481080176332,
				0.000980873958246, -0.00692652171121,
				-0.0169162330746, -0.0106888594278,
				0.00585653048371, 0.0138074306434,
				-0.0478208906869, 0.0548194907154,
				0.00420109717598, -0.0124403232342,
				0.00567369833488, 0.0371481096711,
				0.0306937197576, -0.0224021086711,
				-0.00400841403652, -0.0300551229252,
				0.0732902932636, 0.0458134104469,
				-0.0668617051509, -0.0580895073338,
				0.01761345991, -0.00145492616907,
				0.0320281652266, 0.0310190200109,
				0.0100540410205, 0.00611392133309,
				-0.0253986890351, 0.035396303171,
				0.0140308059331, 0.000374839619012,
				-0.0188215528856, 0.01156958282,
				-0.00130670195036, 0.0498983872269,
				-0.0330274839124, -0.0123636849822,
				0.0225649581684, -0.0167168862779,
				0.0315205951684, -0.0684157061108,
				-0.0428906561036, 0.0754985501526,
				-0.0295073373148, -0.0631896535519,
				-0.09089461634, -0.0290092409887,
				0.00408861284419, 0.00340387882643,
				-0.000359464136101, 0.0218156069361,
				-0.00775859832165, -0.0618378944315,
				0.0478057783988, -0.0637107169233,
				0.0398047959674, -0.0134030969116]
				]
				)


			NN_params=[NN_params.reshape(32,2,order='F')[0:32:2,].T,
					   NN_params.reshape(32,2,order='F')[1:32:2,].T
						]
			'''

			# NN_params=jnp.array(self.data_structure.load_weights()[0])
			# NN_params=[W for W in self.NN_params]


			# CNN
			NN_params=[ ( 
						    (	jnp.array([ [0.021258599215,-0.0764654369726   ],
							 	 			[0.00182637347543,-0.00525690483403]
											]).reshape(1, 1, 2, 2),
						    ),

							(   jnp.array(
											[ [-0.0823963887505,-0.0628942940286],
											  [-0.00695127347814,0.0481080176332]
											]).reshape(1, 1, 2, 2),
							),
						),


							# jnp.array([ [0.000980873958246, -0.0169162330746],
							#   			[0.00585653048371, -0.0478208906869]
							# 			]).reshape(1, 1, 2, 2),

							# jnp.array(
							# 			[ [-0.00692652171121, -0.0106888594278],
							# 			  [0.0138074306434, 0.0548194907154]
							# 			]).reshape(1, 1, 2, 2),
						]


			self.DNN.update_params(NN_params)
		

		# jit functions
		self.evaluate_NN=jit(self.DNN.evaluate)
		#self.evaluate_NN=self.DNN.evaluate


	def _create_optimizer(self):

		@jit
		def loss_log_psi(NN_params,batch,):
			log_psi = self.DNN.evaluate_log(NN_params,batch,)
			return jnp.sum(log_psi)
			

		@jit
		def loss_phase_psi(NN_params,batch,):
			phase_psi = self.DNN.evaluate_phase(NN_params,batch,)	
			return jnp.sum(phase_psi)


		@jit
		def compute_grad_log_psi(NN_params,batch,):

			dlog_psi_s   = vmap(partial(jit(grad(loss_log_psi)),   NN_params))(batch, )
			dphase_psi_s = vmap(partial(jit(grad(loss_phase_psi)), NN_params))(batch, )

			# dlog_psi_s   = vmap(jit(grad(loss_log_psi)),   in_axes=(None,0,) )(NN_params,batch, )
			# dphase_psi_s = vmap(jit(grad(loss_phase_psi)), in_axes=(None,0,) )(NN_params,batch, )
			
			
			N_MC_points=batch.shape[0]

			dlog_psi = []
			for (dlog_psi_layer,dphase_psi_layer) in zip(dlog_psi_s,dphase_psi_s): # loop over layers
				for (dlog_psi_vals,dphase_psi_vals) in zip(dlog_psi_layer,dphase_psi_layer): # cpx vs real network
					for (dlog_psi_W,dphase_psi_W) in zip(dlog_psi_vals,dphase_psi_vals): # W, b
						dlog_psi.append( (dlog_psi_W+1j*dphase_psi_W).reshape(N_MC_points,-1) )

			return jnp.concatenate(dlog_psi, axis=1 )
			#return jnp.concatenate( [(dlog_psi+1j*dphase_psi).reshape(N_MC_points,-1) for (dlog_psi,dphase_psi) in zip(dlog_psi_s,dphase_psi_s)], axis=1  )



		### self.optimizer params
		# initiaize natural gradient class
			
		self.NG=natural_gradient(self.comm,self.N_MC_points,self.N_batch,self.DNN.N_varl_params,compute_grad_log_psi, self.DNN.Reshape )

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
			step_size=1E-4
			self.NG.init_RK_params(step_size)

		self.step_size=step_size



	

	def _create_energy_estimator(self):
		### Energy estimator
		self.E_est=Energy_estimator(self.comm,self.N_MC_points,self.N_batch,self.L,self.DNN.N_symm,self.DNN.NN_type) # contains all of the physics
		self.E_est.init_global_params()
		self.N_features=self.DNN.N_sites*self.DNN.N_symm

	def _create_MC_sampler(self):
		### initialize MC sampler variables
		build_dict_args=(self.E_est.sign,self.L,self.E_est.J2)
		self.MC_tool=MC_sampler(self.comm,build_dict_args=build_dict_args)
		self.MC_tool.init_global_vars(self.L,self.N_batch,self.DNN.N_symm,self.E_est.basis_type)
		self.input_shape=(-1,self.DNN.N_symm,self.E_est.N_sites)


	def _create_data_obj(self,model_params=None):
		### initialize data class
		if model_params is None:
			self.model_params=dict(model='RBMcpx',
							  mode=self.mode,
							  L=self.L,
							  J2=self.E_est.J2,
							  opt=self.optimizer,
							  NNstrct=tuple(tuple(shape) for shape in self.DNN.Reshape.shapes),
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
			#exit()
			
			integer_to_spinstate(self.MC_tool.ints_ket, self.MC_tool.spinstates_ket, self.N_features, NN_type=self.DNN.NN_type)


		for epoch in range(self.N_epochs): 

			
			ti=time.time()

			##### evaluate model
			self.get_training_data(self.DNN.params)
			self.get_Stot_data(self.DNN.params)


			##### check c++ and python DNN evaluation
			if epoch==0:
				self.MC_tool.check_consistency(self.evaluate_NN,self.DNN.params)

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

			#exit()

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
			self.MC_tool.sample(self.DNN)


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
		self.batch=self.MC_tool.spinstates_ket.reshape(self.input_shape)#.T


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
				self.MC_tool.spinstates_ket_tot=self.MC_tool.spinstates_ket_tot.reshape(self.input_shape)
				batch=self.MC_tool.spinstates_ket_tot
				

				#grads=self.compute_grad(self.NN_params,batch,self.params_dict)
				grads=self.compute_grad(self.DNN.params,batch,self.params_dict)
				
				#loss=[np.max([np.max(grads[j]) for j in range(self.DNN.shapes.shape[0])]),0.0]
				loss=[np.max(grads),0.0]


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



