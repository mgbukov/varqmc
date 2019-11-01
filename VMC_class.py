import sys,os

os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel

#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="0"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

#quspin_path = os.path.join(os.path.expanduser('~'),"quspin/QuSpin/")
quspin_path = os.path.join(os.path.expanduser('~'),"quspin/QuSpin_dev/")
sys.path.insert(0,quspin_path)

import jax
print('local devices:', jax.local_devices() )


from jax import jit, grad, vmap, random, ops, partial
from jax.config import config
config.update("jax_enable_x64", True)
from jax.experimental import optimizers

import jax.numpy as jnp
import numpy as np

import yaml
import pickle

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


import time
np.set_printoptions(threshold=np.inf)

# import weights
#from MC_weights import *



class VMC(object):

	def __init__(self,params_dict):


		# initialize communicator
		self.comm=MPI.COMM_WORLD
		self.seed = params_dict['seed']


		self.L=params_dict['L'] # system size
		self.J2 = params_dict['J2']

		self.mode=params_dict['mode'] # exact or MC simulation
		self.optimizer=params_dict['optimizer']
		self.NN_type=params_dict['NN_type'] # DNN vs CNN
		self.NN_dtype=params_dict['NN_dtype'] # 'real' # # cpx vs real network parameters
		 
		self.save_data=params_dict['save_data']
		self.load_data=params_dict['load_data']
		

		# training params
		self.N_iterations=params_dict['N_iterations']
		self.start_iter=params_dict['start_iter']

		### MC sampler
		self.N_MC_points=params_dict['N_MC_points']
		self.N_MC_chains = params_dict['N_MC_chains'] # number of MC chains to run in parallel

		os.environ['OMP_NUM_THREADS']='{0:d}'.format(self.N_MC_chains) # set number of OpenMP threads to run in parallel
		os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="0" # memory per process, interferes with mpi
		os.environ["CUDA_VISIBLE_DEVICES"]="0" # device number



		# number of processors must fix MC sampling ratio
		if self.mode=='exact':
			assert(self.L==4)
			assert(self.N_MC_points==107) # 107 states in the symmetry reduced sector for L=4

			self.N_batch=self.N_MC_points#
			if self.comm.Get_size()>1:
				print('only one MPI process allowed for "exact" simulation.')
				exit()
		else:
			self.N_batch=self.N_MC_points//self.comm.Get_size()
			if self.N_batch//self.N_MC_chains != self.N_batch/self.N_MC_chains:
				print('number of MC chains incompatible with the total number of points:', self.N_batch//self.N_MC_chains, self.N_batch/self.N_MC_chains)
				exit()
			

		
		if self.load_data:

			self.start_iter=params_dict['stop_iter']
			
			model_params=dict(model=self.NN_type+self.NN_dtype,
							  mode=self.mode,
							  L=self.L,
							  J2=self.J2,
							  opt=self.optimizer,
							  NNstrct=params_dict['NN_shape_str'],
							  MCpts=self.N_MC_points,  
							)
			self._create_file_name(model_params)
			self.load_dir=os.getcwd()+'/data/data_files/'
			

		
		
		self._create_NN(load_data=self.load_data)
		self._create_optimizer()
		self._create_energy_estimator()
		self._create_MC_sampler()

		
		if not self.load_data:
			model_params=dict(model=self.NN_type+self.NN_dtype,
							  mode=self.mode,
							  L=self.L,
							  J2=self.J2,
							  opt=self.optimizer,
							  NNstrct=self.NN_shape_str,
							  MCpts=self.N_MC_points, 
							)
			self._create_file_name(model_params)

		# create log file and directory
		self._create_logs()


		# add variables to yaml file
		if self.comm.Get_rank()==0:
			
			config_params_yaml = open('config_params.yaml', 'w')
			
			params_dict['N_batch']=self.N_batch
			params_dict['NN_shape_str']=self.NN_shape_str
			
			self.params_dict=params_dict
			yaml.dump(self.params_dict, config_params_yaml)

			config_params_yaml.close()

			
		# train net
		self.train(self.start_iter)
		
		


	def _create_NN(self, load_data=False):

		
		if self.NN_type == 'DNN':
			shapes=dict(layer_1 = [self.L**2, 6], 
					#	layer_2 = [4,12], 
						)
			self.NN_shape_str='{0:d}'.format(self.L**2) + ''.join( '--{0:d}'.format(value[1]) for value in shapes.values() )

		elif self.NN_type == 'CNN':
			shapes=dict( layer_1 = dict(out_chan=1, filter_shape=(2,2), strides=(1,1), ),
					#	 layer_2 = dict(out_chan=1, filter_shape=(2,2), strides=(1,1), ),
						)
			self.NN_shape_str='{0:d}'.format(self.L**2) + ''.join( '--{0:d}-{1:d}-{2:d}'.format(value['out_chan'],value['filter_shape'][0],value['strides'][0]) for value in shapes.values() )

		
		### create Neural network
		self.DNN=Neural_Net(self.comm.Get_rank(), shapes, self.N_MC_chains, self.NN_type, self.NN_dtype, seed=self.seed+self.comm.Get_rank() )

		#exit()
		
		if load_data:

			
			file_name='NNparams'+'--iter_{0:05d}--'.format(self.start_iter-1) + self.file_name
			with open(self.load_dir+file_name+'.pkl', 'rb') as handle:
				NN_params = pickle.load(handle)

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

			self.opt_state = self.opt_init(self.DNN.params)

		elif self.optimizer=='RK':
			step_size=1E-4
			self.NG.init_RK_params(step_size)

		self.step_size=step_size



	

	def _create_energy_estimator(self):
		### Energy estimator
		self.E_estimator=Energy_estimator(self.comm,self.J2,self.N_MC_points,self.N_batch,self.L,self.DNN.N_symm,self.DNN.NN_type) # contains all of the physics
		self.E_estimator.init_global_params()
		self.N_features=self.DNN.N_sites*self.DNN.N_symm

	def _create_MC_sampler(self):
		### initialize MC sampler variables
		self.MC_tool=MC_sampler(self.comm,self.N_MC_chains)
		self.MC_tool.init_global_vars(self.L,self.N_batch,self.DNN.N_symm,self.E_estimator.basis_type)
		self.input_shape=(-1,self.DNN.N_symm,self.DNN.N_sites)
		
		


	def _create_file_name(self,model_params,extra_label=''):
		file_name = ''
		for key,value in model_params.items():
			file_name += ( key+'_{}'.format(value)+'-' )
		file_name=file_name[:-1]
		self.file_name=file_name+extra_label



	def _create_logs(self):

		logfile_dir=os.getcwd()+'/data/log_files/'
		if not os.path.exists(logfile_dir):
		    os.makedirs(logfile_dir)

		self.savefile_dir=os.getcwd()+'/data/data_files/'
		if not os.path.exists(self.savefile_dir):
		    os.makedirs(self.savefile_dir)
		
		
		def create_open_file(file_name):
			# open log_file
			if os.path.exists(file_name):
				if self.load_data:
				    append_write = 'a' # append if already exists
				else:
					append_write = 'w' # make a new file if not
			else:
				append_write = 'w+' # append if already exists

			return open(file_name, append_write)

		# logfile name
		logfile_name= 'LOGFILE--MPIprss_{0:d}--'.format(self.comm.Get_rank()) + self.file_name + '.txt'
		self.logfile = create_open_file(logfile_dir+logfile_name)

		if self.save_data:
			# data files
			common_str =  self.file_name + '.txt'

			self.file_energy= create_open_file(self.savefile_dir+'energy--'+common_str)
			self.file_energy_std= create_open_file(self.savefile_dir+'energy_std--'+common_str)
			self.file_loss= create_open_file(self.savefile_dir+'loss--'+common_str)
			self.file_r2= create_open_file(self.savefile_dir+'r2--'+common_str)
			self.file_phase_hist=create_open_file(self.savefile_dir+'phases_histogram--'+common_str)

			self.file_MC_data= create_open_file(self.savefile_dir+'MC_data--'+common_str)


		


	def check_point(self, iteration, loss, r2, phases, amplds):
			

		# NN parameters
		file_name='NNparams'+'--iter_{0:05d}--'.format(iteration) + self.file_name
		with open(self.savefile_dir+file_name+'.pkl', 'wb') as handle:
			#pickle.dump([self.DNN.params,], handle, protocol=pickle.HIGHEST_PROTOCOL)
			pickle.dump(self.DNN.params, handle, protocol=pickle.HIGHEST_PROTOCOL)


		# data
		self.file_energy.write("{0:d} : {1:0.14f} : {2:0.14f}\n".format(iteration, self.Eloc_mean_g.real , self.Eloc_mean_g.imag))
		self.file_energy_std.write("{0:d} : {1:0.14f}\n".format(iteration, self.E_MC_std_g))
		
		self.file_loss.write("{0:d} : {1:0.14f} : {2:0.14f}\n".format(iteration, loss[0], loss[1]))
		self.file_r2.write("{0:d} : {1:0.14f}\n".format(iteration, r2))

		self.file_MC_data.write("{0:d} : {1:0.4f} : ".format(iteration, self.MC_tool.acceptance_ratio[0]) + ' '.join(str(s) for s in self.MC_tool.s0_tot) +"\n") #
		

		# compute histogram
		n_bins=40
		#binned_phases=np.linspace(-np.pi,np.pi, n_bins, endpoint=True)

		# shift phases
		phases = (phases+np.pi)%(2*np.pi) - np.pi
		hist, bin_edges = np.histogram(phases ,bins=n_bins,range=(-np.pi,np.pi), density=True, weights=amplds**2)
		phase_hist = hist*np.diff(bin_edges)

		self.file_phase_hist.write("{0:d} : ".format(iteration) + ''.join("{0:0.6f}, ".format(value) for value in phase_hist) + '\n' )


		# record current iteration number
		self.params_dict['stop_iter']=iteration+1
		
		config_params_yaml = open('config_params.yaml', 'w')
		yaml.dump(self.params_dict, config_params_yaml)
		config_params_yaml.close()



	def train(self, start=0):

		# set timer
		t_start=time.time()

		


		if self.mode=='exact':
			self.MC_tool.ints_ket, self.index, self.inv_index, self.count=self.E_estimator.get_exact_kets()
			integer_to_spinstate(self.MC_tool.ints_ket, self.MC_tool.spinstates_ket, self.N_features, NN_type=self.DNN.NN_type)


		for iteration in range(start,self.N_iterations, 1): 

			
			#self.comm.Barrier()
			ti=time.time()

			self.logfile.write("iteration {0:d}, process_rank {1:d}:\n".format(iteration, self.comm.Get_rank()))


			##### evaluate model
			self.get_training_data(self.DNN.params)
			#self.get_Stot_data(self.DNN.params)

			
			##### check c++ and python DNN evaluation
			if iteration==0:
				self.MC_tool.check_consistency(self.evaluate_NN,self.DNN.params)
				if self.mode=='exact':
					np.testing.assert_allclose(self.Eloc_mean_g.real, self.E_estimator.H.expt_value(self.psi[self.inv_index]))


			#####		
			if self.comm.Get_rank()==0:
				print("iteration {0:d}:".format(iteration))
				print("E={0:0.14f}, E_std={1:0.14f}.\n".format(self.Eloc_mean_g.real, self.E_MC_std_g ) )
				
			self.logfile.write("E={0:0.14f}, E_std={1:0.14f}.\n".format(self.Eloc_mean_g.real, self.E_MC_std_g ) 	)

			if self.mode=='exact':
				self.logfile.write('overlap = {0:0.4f}.\n\n'.format(self.Eloc_params_dict['overlap']) )

			#exit()

			#### update model parameters
			if iteration<self.N_iterations-1:
				loss, r2 = self.update_NN_params(iteration)

				##### store data
				
				#combine results from all cores
				self.MC_tool.Allgather()	

				if self.comm.Get_rank()==0 and self.save_data:
					self.check_point(iteration,loss,r2,self.MC_tool.phase_kets_tot,self.MC_tool.mod_kets_tot)

			

			self.logfile.write("iteration {0:d}: total calculation took {1:0.4f} secs.\n\n".format(iteration, time.time()-ti) )
			print("iteration {0:d}, process_rank {1:d} total calculation took {2:0.4f} secs.\n".format(iteration, self.comm.Get_rank(),time.time()-ti) )
			self.logfile.flush()
			os.fsync(self.logfile.fileno())


			# synch 
			self.comm.Barrier()

			

		print('\nprocess_rank {0:d}, total calculation time: {1:0.4f} secs.\n'.format(self.comm.Get_rank(),time.time()-t_start))
		self.logfile.write('\nprocess_rank {0:d}, total calculation time: {1:0.4f} secs.\n\n\n'.format(self.comm.Get_rank(),time.time()-t_start) )
		

		# close files
		self.logfile.close()
		self.file_energy.close()
		self.file_energy_std.close()
		self.file_loss.close()
		self.file_r2.close()
		self.file_phase_hist.close()


		

	def get_training_data(self,NN_params):

		##### get spin configs #####
		if self.mode=='exact':
			self.MC_tool.exact(NN_params, evaluate_NN=self.evaluate_NN)
			
		elif self.mode=='MC':
			ti=time.time()
			
			# sample
			acceptance_ratio = self.MC_tool.sample(self.DNN)
			
			self.logfile.write("MC with acceptance ratio={0:.4f}: took {1:.4f} secs.\n".format(acceptance_ratio[0],time.time()-ti))
			
			if self.comm.Get_rank()==0:
				print("MC acceptance ratio={0:.4f}: took {1:.4f} secs.\n".format(acceptance_ratio[0],time.time()-ti))
		


		##### compute local energies #####
		ti=time.time()
		self.E_estimator.compute_local_energy(self.evaluate_NN,NN_params,self.MC_tool.ints_ket,self.MC_tool.mod_kets,self.MC_tool.phase_kets,self.MC_tool.log_psi_shift)
		self.logfile.write("total local energy calculation took {0:.4f} secs.\n".format(time.time()-ti))
		if self.comm.Get_rank()==0:
			print("total local energy calculation took {0:.4f} secs.\n".format(time.time()-ti))

		if self.mode=='exact':
			#print(self.MC_tool.mod_kets)
			self.psi = self.MC_tool.mod_kets*np.exp(+1j*self.MC_tool.phase_kets)/np.linalg.norm(self.MC_tool.mod_kets[self.inv_index])
			abs_psi_2=self.count*np.abs(self.psi)**2
			#print(abs_psi_2)
			#exit()
			self.Eloc_params_dict=dict(abs_psi_2=abs_psi_2,)
			overlap=np.abs(self.psi[self.inv_index].dot(self.E_estimator.psi_GS_exact))**2
			self.Eloc_params_dict['overlap']=overlap
		
		elif self.mode=='MC':
			self.Eloc_params_dict=dict(N_MC_points=self.N_MC_points)	

		
		self.Eloc_mean_g, self.Eloc_var_g, E_diff_real, E_diff_imag = self.E_estimator.process_local_energies(mode=self.mode,Eloc_params_dict=self.Eloc_params_dict)
		self.Eloc_std_g=np.sqrt(self.Eloc_var_g)
		self.E_MC_std_g=self.Eloc_std_g/np.sqrt(self.N_MC_points)
		

		self.Eloc_params_dict['E_diff']=E_diff_real+1j*E_diff_imag
		self.Eloc_params_dict['Eloc_mean']=self.Eloc_mean_g
		self.Eloc_params_dict['Eloc_var']=self.Eloc_var_g

		##### total batch
		self.batch=self.MC_tool.spinstates_ket.reshape(self.input_shape)

		return self.batch, self.Eloc_params_dict
		

	def get_Stot_data(self,NN_params): 
		# check SU(2) conservation
		self.E_estimator.compute_local_energy(self.evaluate_NN,NN_params,self.MC_tool.ints_ket,self.MC_tool.mod_kets,self.MC_tool.phase_kets,self.MC_tool.log_psi_shift,SdotS=True)
		self.SdotSloc_mean, SdotS_var, SdotS_diff_real, SdotS_diff_imag = self.E_estimator.process_local_energies(mode=self.mode,Eloc_params_dict=self.Eloc_params_dict,SdotS=True)
		self.SdotS_MC_std=np.sqrt(SdotS_var/self.N_MC_points)


	def update_NN_params(self,iteration):

		

		if self.optimizer=='RK':
			# compute updated NN parameters
			self.DNN.update_params(self.NG.Runge_Kutta(self.DNN.params,self.batch,self.Eloc_params_dict,self.mode,self.get_training_data))
			loss=self.NG.max_grads

		else:
			##### compute gradients
			if self.optimizer=='NG':
				# compute enatural gradients
				grads=self.NG.compute(self.DNN.params,self.batch,self.Eloc_params_dict,mode=self.mode)
				loss=self.NG.max_grads

			elif self.optimizer=='adam':
				# compute adam gradients
				grads_MPI=self.DNN.Reshape.from_gradient_format( self.compute_grad(self.DNN.params,self.batch,self.Eloc_params_dict) )
				
				# sum up MPI processes
				grads=np.zeros_like(grads_MPI)
				self.comm.Allreduce(grads_MPI._value, grads,  op=MPI.SUM) 
				grads = self.DNN.Reshape.to_gradient_format( grads )

				loss=[np.max(grads),0.0]


			##### apply gradients
			self.opt_state = self.opt_update(iteration, grads, self.opt_state) 
			self.NG.update_params() # update NG params
			self.DNN.update_params(self.get_params(self.opt_state))
			
		##### compute loss
		r2=self.NG.r2_cost

		return loss, r2


