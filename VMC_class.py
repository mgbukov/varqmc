import sys,os,warnings
from mpi4py import MPI

os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'

os.environ["NUM_INTER_THREADS"]="1"
os.environ["NUM_INTRA_THREADS"]="1"

# set XLA threads and parallelism
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="0"
#os.environ["CUDA_VISIBLE_DEVICES"]="{0:d}".format(MPI.COMM_WORLD.Get_rank()) # device number
#print("process {0:d} runs on GPU device {1:d}".format(MPI.COMM_WORLD.Get_rank(),int(os.environ["CUDA_VISIBLE_DEVICES"])))


#quspin_path = os.path.join(os.path.expanduser('~'),"quspin/QuSpin/")
quspin_path = os.path.join(os.path.expanduser('~'),"quspin/QuSpin_dev/")
sys.path.insert(0,quspin_path)

import jax
#print('local devices:', jax.local_devices() )

from jax.lib import xla_bridge
from jax import jit, grad, vmap, random, ops, partial
from jax.config import config
config.update("jax_enable_x64", True)
from jax.experimental import optimizers
#from optimizers import sgd, adam

import jax.numpy as jnp
import numpy as np
from scipy.linalg import sqrtm, pinvh

import yaml
import pickle

from cpp_code import Neural_Net
from cpp_code import integer_to_spinstate
from cpp_code import scale_cpx

from natural_grad import natural_gradient
from MC_lib import MC_sampler
from energy_lib import Energy_estimator


import datetime
import time
np.set_printoptions(threshold=np.inf)


#from misc.MC_weights import *



def read_str(NN_shape_str):

	shape_tuple=()

	NN_shape_str=NN_shape_str.replace('(','')
	NN_shape_str=NN_shape_str.replace(')','')
	NN_shape_str=NN_shape_str.split(',')


	for NN_str in NN_shape_str:
		shape_tuple+=(NN_str,)

	return shape_tuple, len(NN_shape_str)



class VMC(object):

	def __init__(self,data_dir,params_dict=None,train=True,):

		if params_dict is None:
			self.data_dir=data_dir
			params_dict = yaml.load(open(self.data_dir+'/config_params_init.yaml'),Loader=yaml.FullLoader)
		else:
			self.data_dir=data_dir

		self.params_dict=params_dict
			

		# initialize communicator
		self.comm=MPI.COMM_WORLD
		self.platform=xla_bridge.get_backend().platform
		self.seed = params_dict['seed']

		np.random.seed(self.seed)
		np.random.RandomState(self.seed)
		#rng = random.PRNGKey(self.seed)


		self.L=params_dict['L'] # system size
		self.J2 = params_dict['J2']
		self.sign = params_dict['sign'] # -1: Marshal rule is on; +1 Marshal rule is off

		self.mode=params_dict['mode'] # exact or MC simulation
		self.optimizer=params_dict['optimizer']
		self.grad_update_mode=params_dict['grad_update_mode']
		

		self.NN_type=params_dict['NN_type'] # DNN vs CNN
		self.NN_dtype=params_dict['NN_dtype'] # 'real' # # cpx vs real network parameters
		self.NN_shape_str=params_dict['NN_shape_str']
		 
		self.save_data=params_dict['save_data']
		self.load_data=params_dict['load_data']
		self.batchnorm=params_dict['batchnorm']


		# training params
		self.N_iterations=params_dict['N_iterations']
		self.start_iter=params_dict['start_iter']

		### MC sampler
		self.thermal=params_dict['MC_thermal']
		self.N_MC_points=params_dict['N_MC_points']
		self.N_MC_chains = params_dict['N_MC_chains'] # number of MC chains to run in parallel
		self.minibatch_size=params_dict['minibatch_size'] # define batch size for GPU evaluation of local energy


		os.environ['OMP_NUM_THREADS']='{0:d}'.format(self.N_MC_chains) # set number of OpenMP threads to run in parallel
		

		# number of processors must fix MC sampling ratio
		if self.mode=='exact':
			assert(self.L==4)
			self.N_batch=self.N_MC_points#
			if self.comm.Get_size()>1:
				print('only one MPI process allowed for "exact" simulation.')
				exit()
		elif self.mode=='MC':
			self.N_batch=self.N_MC_points//self.comm.Get_size()
			if self.comm.Get_rank() < self.N_MC_points%self.comm.Get_size():
				self.N_batch+=1
		else:
			print('unrecognized operation mode!')
			

		
		
	
		model_params=dict(model=self.NN_type+self.NN_dtype,
						  mode=self.mode,
						  L=self.L,
						  J2=self.J2,
						  opt=self.optimizer,
						  NNstrct=self.NN_shape_str,
						  MCpts=self.N_MC_points,
						  Nprss=self.comm.Get_size(),
						  NMCchains=self.N_MC_chains,
						)

		
		self.n_iter=10 # define number of iterations to store for debugging purposes
		
		self._create_file_name(model_params)
		self._create_NN(load_data=self.load_data)
		self._create_optimizer()
		self._create_energy_estimator()
		self._create_MC_sampler()

		

		# create log file and directory
		if self.save_data:
			self._create_logs()

			if self.load_data:
				self._load_data()


		# add variables to yaml file
		elif self.comm.Get_rank()==0 and self.save_data:
			
			config_params_yaml = open(self.data_dir + '/config_params.yaml', 'w')
			
			self.params_dict['N_batch']=self.N_batch
			self.params_dict['NN_shape_str']=self.NN_shape_str
			
			yaml.dump(self.params_dict, config_params_yaml)

			config_params_yaml.close()

			
		# train net
		if train:
			self.train(self.start_iter)
		

	def _load_data(self):


		start_iter=self.start_iter

		### load MC data
		
		with open(self.file_MC_data.name) as file:	
			for i in range(start_iter):				
				MC_data_str = file.readline().rstrip().split(' : ')

		it_MC, acceptance_ratio_g, acceptance_ratios, s0_g, sf_g =  MC_data_str

		self.MC_tool.acceptance_ratio_g[0]=np.float64(acceptance_ratio_g)
		self.MC_tool.s0_g=np.array([self.E_estimator.basis_type(s0) for s0 in s0_g.split(' ')] )
		self.MC_tool.sf_g=np.array([self.E_estimator.basis_type(sf) for sf in sf_g.split(' ')] )

		self.DNN._init_MC_data(s0_vec=self.MC_tool.s0_g, sf_vec=self.MC_tool.sf_g, )

		### load DNN params

		file_name='/NN_params/NNparams'+'--iter_{0:05d}--'.format(self.start_iter) + self.file_name

		with open(self.data_dir+file_name+'.pkl', 'rb') as handle:
			self.DNN.params,self.DNN.apply_fun_args,_ = pickle.load(handle)
			self.DNN.apply_fun_args_dyn=self.DNN.apply_fun_args

		self.opt_state = self.opt_init(self.DNN.params)

		file_name='/NN_params/NNparams'+'--iter_{0:05d}--'.format(self.start_iter-1) + self.file_name
		with open(self.data_dir+file_name+'.pkl', 'rb') as handle:
			DNN_params_old,_,_ = pickle.load(handle)

		
		self.NG.nat_grad_guess[:]=self.DNN.NN_Tree.ravel(DNN_params_old)-self.DNN.NN_Tree.ravel(self.DNN.params)


		### load NG data

		with open(self.file_opt_data.name) as file:	
			for i in range(start_iter):				
				opt_data_str = file.readline().rstrip().split(' : ')

		
		self.NG.iteration=int(opt_data_str[0])+1
		self.NG.counter=int(opt_data_str[1])
		self.NG.RK_step_size=np.float64(opt_data_str[2])
		self.NG.RK_time=np.float64(opt_data_str[3])
		self.NG.delta=np.float64(opt_data_str[4])
		self.NG.tol=np.float64(opt_data_str[5])


		with open(self.file_loss.name) as file:	
			for i in range(start_iter+1):				
				loss_data_str = file.readline().rstrip().split(' : ')


		#####
		
		assert(int(it_MC)+1==self.NG.iteration)



	def update_batchnorm_params(self,layers, set_fixpoint_iter=True,):
		layers_type=list(layers.keys())
		for j, layer_type in enumerate(layers_type):
			if 'batch_norm' in layer_type:
				self.DNN.apply_fun_args_dyn[j]['fixpoint_iter']=set_fixpoint_iter
				#print(self.DNN.apply_fun_args_dyn[j]['mean'])

				




	def _create_NN(self, load_data=False):

		
		if ',' in self.NN_shape_str:

			NN_shape_str, M=read_str(self.NN_shape_str)
			

			self.shapes=tuple({} for _ in range(M) )

			neurons=tuple([] for _ in range(M))
			for j in range(M):
				for neuron in NN_shape_str[j].split('--'):
					neurons[j].append(int(neuron))

				assert(neurons[j][0]==self.L**2)	
		
		else:
			self.shapes={}
			neurons=[]
			for neuron in self.NN_shape_str.split('--'):
				neurons.append(int(neuron))

			assert(neurons[0]==self.L**2)

			
	
		if self.NN_type == 'DNN':

			if ',' in self.NN_shape_str:

				for j in range(M):
					for i in range(len(neurons[j])-1):
						self.shapes[j]['layer_{0:d}'.format(i+1)]=[neurons[j][i],neurons[j][i+1]]

			else:
				for i in range(len(neurons)-1):
					self.shapes['layer_{0:d}'.format(i+1)]=[neurons[i],neurons[i+1]]
		
			# self.shapes=dict(layer_1 = [self.L**2, 8], 
			# 			#	 layer_2 = [12       ,  6],
			# 			#	 layer_3 = [4       ,  2], 
			# 			)
			# self.NN_shape_str='{0:d}'.format(self.L**2) + ''.join( '--{0:d}'.format(value[1]) for value in self.shapes.values() )


		elif self.NN_type == 'CNN':
			self.shapes=dict( layer_1 = dict(out_chan=1, filter_shape=(2,2), strides=(1,1), ),
					#	 layer_2 = dict(out_chan=1, filter_shape=(2,2), strides=(1,1), ),
						)
			self.NN_shape_str='{0:d}'.format(self.L**2) + ''.join( '--{0:d}-{1:d}-{2:d}'.format(value['out_chan'],value['filter_shape'][0],value['strides'][0]) for value in self.shapes.values() )



		### create Neural network
		self.DNN=Neural_Net(self.comm, self.shapes, self.N_MC_chains, self.NN_type, self.NN_dtype, seed=self.seed )
		#self.DNN.update_params(load_params())
	

		# jit functions
		self.evaluate_NN_dyn=self.DNN.evaluate_dyn
		
		self.evaluate_NN=jit(self.DNN.evaluate)
		#self.evaluate_NN=self.DNN.evaluate
		#print("\n\nNN evaluation NOT JITTED !!!\n\n")
		
		#self.evaluate_NN=partial(jit(self.DNN.evaluate,static_argnums=2),)
		


	def _create_optimizer(self):

		@jit
		def loss_log_mod_psi(NN_params,batch,):
			log_psi = self.DNN.evaluate_log(NN_params,batch,)
			return jnp.sum(log_psi)
			

		@jit
		def loss_phase_psi(NN_params,batch,):
			phase_psi = self.DNN.evaluate_phase(NN_params,batch,)	
			return jnp.sum(phase_psi)

		@jit
		def grad_log_psi(NN_params,batch,):

			# dlog_psi_s   = vmap(partial(grad(loss_log_mod_psi),   NN_params))(batch, )
			# dphase_psi_s = vmap(partial(grad(loss_phase_psi), NN_params))(batch, )


			# dlog_psi_s   = vmap(jit(grad(loss_log_mod_psi)),   in_axes=(None,0,) )(NN_params,batch, )
			# dphase_psi_s = vmap(jit(grad(loss_phase_psi)), in_axes=(None,0,) )(NN_params,batch, )

			
			dlog_psi_s   = vmap(partial(jit(grad(loss_log_mod_psi)),   NN_params))(batch, )
			dphase_psi_s = vmap(partial(jit(grad(loss_phase_psi)), NN_params))(batch, )


			dlog_psi = []

			for (dlog_psi_W,dphase_psi_W) in zip(self.DNN.NN_Tree.flatten(dlog_psi_s),self.DNN.NN_Tree.flatten(dphase_psi_s)):
				dlog_psi.append( (dlog_psi_W+1j*dphase_psi_W).reshape(self.N_batch,-1) )

			# for (dlog_psi_layer,dphase_psi_layer) in zip(dlog_psi_s,dphase_psi_s): # loop over layers
			# 	for (dlog_psi_W,dphase_psi_W) in zip(dlog_psi_layer,dphase_psi_layer): # loop over NN params
			# 		dlog_psi.append( (dlog_psi_W+1j*dphase_psi_W).reshape(self.N_batch,-1) )


			# for (dlog_psi_tower,dphase_psi_tower) in zip(dlog_psi_s,dphase_psi_s): # loop over layers
			# 	for (dlog_psi_layer,dphase_psi_layer) in zip(dlog_psi_tower,dphase_psi_tower): # loop over layers
			# 		for (dlog_psi_W,dphase_psi_W) in zip(dlog_psi_layer,dphase_psi_layer): # loop over NN params
			# 			dlog_psi.append( (dlog_psi_W+1j*dphase_psi_W).reshape(self.N_batch,-1) )

	
			return jnp.concatenate(dlog_psi, axis=1)


		@jit
		def grad_log_mod_psi(NN_params,batch,):

			dlog_psi_s   = vmap(partial(jit(grad(loss_log_mod_psi)),   NN_params))(batch, )
			
			dlog_psi = []
			for dlog_psi_W in self.DNN.NN_Tree.flatten(dlog_psi_s):
				dlog_psi.append( dlog_psi_W.reshape(self.N_batch,-1) )

			# for dlog_psi_layer in dlog_psi_s: # loop over layers
			# 	for dlog_psi_W in dlog_psi_layer: # loop over NN params
			# 		dlog_psi.append( dlog_psi_W.reshape(self.N_batch,-1) )

			return jnp.concatenate(dlog_psi, axis=1)


		@jit
		def grad_log_phase(NN_params,batch,):

			dphase_psi_s = vmap(partial(jit(grad(loss_phase_psi)), NN_params))(batch, )
	
			dlog_psi = []
			for dphase_psi_W in self.DNN.NN_Tree.flatten(dphase_psi_s):
				dlog_psi.append( 1j*dphase_psi_W.reshape(self.N_batch,-1) )


			# for dphase_psi_layer in dphase_psi_s: # loop over layers
			# 	for dphase_psi_W in dphase_psi_layer: # loop over NN params
			# 		dlog_psi.append( 1j*dphase_psi_W.reshape(self.N_batch,-1) )

			
			return jnp.concatenate(dlog_psi, axis=1)



		def compute_grad_log_psi(NN_params,batch,iteration=0):

			if self.grad_update_mode=='normal':
				dlog_psi=grad_log_psi(NN_params,batch)

			elif self.grad_update_mode=='alternating':
				if iteration%2==0: # phase grads
					dlog_psi=grad_log_phase(NN_params,batch)

				else: # log_mod_psi grads
					dlog_psi=grad_log_mod_psi(NN_params,batch)

			elif self.grad_update_mode=='phase':
				dlog_psi=grad_log_phase(NN_params,batch)

			elif self.grad_update_mode=='log_mod':
				dlog_psi=grad_log_mod_psi(NN_params,batch)
				

			return dlog_psi


	
		self.NG=natural_gradient(self.comm,self.N_MC_points,self.N_batch, self.DNN.N_varl_params_vec, compute_grad_log_psi, self.DNN.NN_Tree, self.NN_dtype, self.grad_update_mode, start_iter=self.start_iter )
		self.NG.init_global_variables(self.n_iter)
		self.NG.run_debug_helper=self.run_debug_helper


		# jax self.optimizer
		if self.optimizer=='NG':
			self.learning_rates=[1E-2,1E-2]
			self.opt_init, self.opt_update, self.get_params = optimizers.sgd(step_size=1.0)
			self.opt_state = self.opt_init(self.DNN.params)

		elif self.optimizer=='adam':

			if self.load_data:
				raise("Loading data not supported for adam yet")
			
			step_size=1E-3
			self.opt_init, self.opt_update, self.get_params = optimizers.adam(step_size=step_size, b1=0.9, b2=0.99, eps=1e-08)
			self.opt_state = self.opt_init(self.DNN.params)

			# Energy cost function
			if self.mode=='exact':

				@jax.partial(jit, static_argnums=(2,3))
				def loss_energy_exact(NN_params,batch,params_dict,iteration):
					if self.grad_update_mode=='normal':
						log_psi, phase_psi = self.DNN.evaluate_dyn(NN_params,batch,)			
						Energy = 2.0*jnp.sum(params_dict['abs_psi_2']*(log_psi*params_dict['E_diff'].real + phase_psi*params_dict['E_diff'].imag ))

					elif self.grad_update_mode=='alternating':
						if iteration%2==0: # phase grads
							phase_psi = self.DNN.evaluate_phase(NN_params,batch,)
							Energy = 2.0*jnp.sum(params_dict['abs_psi_2']*(phase_psi*params_dict['E_diff'].imag ))
						else:
							log_psi = self.DNN.evaluate_log(NN_params,batch,)
							Energy = 2.0*jnp.sum(params_dict['abs_psi_2']*(log_psi*params_dict['E_diff'].real ))

					elif self.grad_update_mode=='log_mod':
						log_psi = self.DNN.evaluate_log(NN_params,batch,)
						Energy = 2.0*jnp.sum(params_dict['abs_psi_2']*(log_psi*params_dict['E_diff'].real ))

					elif self.grad_update_mode=='phase':
						phase_psi = self.DNN.evaluate_phase(NN_params,batch,)
						Energy = 2.0*jnp.sum(params_dict['abs_psi_2']*(phase_psi*params_dict['E_diff'].imag ))

					return Energy

				self.compute_grad=jit(grad(loss_energy_exact), static_argnums=(2,3))
			
			elif self.mode=='MC':

				@jax.partial(jit, static_argnums=(2,3))
				def loss_energy_MC(NN_params,batch,params_dict,iteration):
					if self.grad_update_mode=='normal':
						log_psi, phase_psi = self.DNN.evaluate_dyn(NN_params,batch,)
						Energy=2.0*jnp.sum(log_psi*params_dict['E_diff'].real + phase_psi*params_dict['E_diff'].imag)/params_dict['N_MC_points']

					elif self.grad_update_mode=='alternating':
						if iteration%2==0: # phase grads
							phase_psi = self.DNN.evaluate_phase(NN_params,batch,)
							Energy=2.0*jnp.sum(phase_psi*params_dict['E_diff'].imag)/params_dict['N_MC_points']
						else:
							log_psi = self.DNN.evaluate_log(NN_params,batch,)
							Energy=2.0*jnp.sum(log_psi*params_dict['E_diff'].real)/params_dict['N_MC_points']

					elif self.grad_update_mode=='log_mod':
						log_psi = self.DNN.evaluate_log(NN_params,batch,)
						Energy=2.0*jnp.sum(log_psi*params_dict['E_diff'].real)/params_dict['N_MC_points']

					elif self.grad_update_mode=='phase':
						phase_psi = self.DNN.evaluate_phase(NN_params,batch,)
						Energy=2.0*jnp.sum(phase_psi*params_dict['E_diff'].imag)/params_dict['N_MC_points']

					return Energy

				self.compute_grad=jit(grad(loss_energy_MC), static_argnums=(2,3))

			
		elif self.optimizer=='RK':

			print("\n\nsingle solver currently available\n\n")
			exit()

			step_size=1E-4
			self.NG.init_RK_params(step_size)

		#self.step_size=step_size

		# define variable to keep track of the DNN params update
		n_iter=6
		if self.comm.Get_rank()==0:
			self.params_update=np.zeros((n_iter,self.DNN.N_varl_params),dtype=np.float64)
		else:
			self.params_update=np.array([[None],[None]])



	

	def _create_energy_estimator(self):
		### Energy estimator
		self.E_estimator=Energy_estimator(self.comm,self.J2,self.N_MC_points,self.N_batch,self.L,self.DNN.N_symm,self.DNN.NN_type,self.sign,) # contains all of the physics
		self.E_estimator.init_global_params(self.N_MC_points,self.n_iter)
		
	def _create_MC_sampler(self, ):
		### initialize MC sampler variables
		self.MC_tool=MC_sampler(self.comm,self.N_MC_chains)
		self.MC_tool.init_global_vars(self.L,self.N_MC_points,self.N_batch,self.DNN.N_symm,self.E_estimator.basis_type,self.E_estimator.MPI_basis_dtype,self.n_iter)
		self.input_shape=(-1,self.DNN.N_symm,self.DNN.N_sites)
		
		


	def _create_file_name(self,model_params,extra_label=''):
		file_name = ''
		for key,value in model_params.items():
			file_name += ( key+'_{}'.format(value)+'-' )
		file_name=file_name[:-1]
		self.file_name=file_name+extra_label



	def _create_logs(self):

		# sys_data=''
		
		# if self.comm.Get_rank()==0:
		# 	sys_time=datetime.datetime.now()
		# 	sys_data="{0:d}-{1:02d}-{2:02d}_{3:02d}:{4:02d}:{5:02d}_".format(sys_time.year, sys_time.month, sys_time.day, sys_time.hour, sys_time.minute, sys_time.second)
		# 	#sys_data="{0:d}-{1:02d}-{2:02d}_".format(sys_time.year,sys_time.month,sys_time.day,)

		# # broadcast sys_data
		# sys_data = self.comm.bcast(sys_data, root=0)


		# self.sys_time=sys_data + self.optimizer

		# self.data_dir=os.getcwd()+'/data/'+self.sys_time

		logfile_dir=self.data_dir+'/log_files/'
		self.savefile_dir=self.data_dir+'/data_files/'
		self.savefile_dir_NN=self.data_dir+'/NN_params/'	
		self.savefile_dir_debug=self.data_dir+'/debug_files/'

		if self.comm.Get_rank()==0:

			if not os.path.exists(logfile_dir):
			    os.makedirs(logfile_dir)

			if not os.path.exists(self.savefile_dir):
			    os.makedirs(self.savefile_dir)

			if not os.path.exists(self.savefile_dir_NN):
			    os.makedirs(self.savefile_dir_NN)

			if not os.path.exists(self.savefile_dir_debug):
			    os.makedirs(self.savefile_dir_debug)

		# wait for process 0 to check if directories exist
		self.comm.Barrier()


		def create_open_file(file_name):
			# open log_file
			if os.path.exists(file_name):
				if self.load_data:
				    append_write = 'a+' # append if already exists
				else:
					append_write = 'w' # make a new file if not
			else:
				append_write = 'w+' # append if already exists

			return open(file_name, append_write)

		# logfile name
		logfile_name= 'LOGFILE--MPIprss_{0:d}--'.format(self.comm.Get_rank()) + self.file_name + '.txt'
		self.logfile = create_open_file(logfile_dir+logfile_name)
		self.E_estimator.logfile=self.logfile
		self.NG.logfile=self.logfile

		# redircet warnings to log
		def customwarn(message, category, filename, lineno, file=None, line=None):
			self.logfile.write('\n'+warnings.formatwarning(message, category, filename, lineno)+'\n')
		warnings.showwarning = customwarn

		
		self.debug_file_SF=self.savefile_dir_debug + 'debug-SF_data'+'--' + self.file_name
		self.debug_file_logpsi=self.savefile_dir_debug + 'debug-logpsi_data'+'--' + self.file_name
		self.debug_file_phasepsi=self.savefile_dir_debug + 'debug-phasepsi_data'+'--' + self.file_name
		self.debug_file_intkets=self.savefile_dir_debug + 'debug-intkets_data'+'--' + self.file_name
		self.debug_file_Eloc=self.savefile_dir_debug + 'debug-Eloc_data'+'--' + self.file_name
		self.debug_file_params_update=self.savefile_dir_debug + 'debug-params_update_data'+'--' + self.file_name
		

		if self.save_data:
			# data files
			common_str =  self.file_name + '.txt'

			self.file_energy= create_open_file(self.savefile_dir+'energy--'+common_str)
			#self.file_energy_std= create_open_file(self.savefile_dir+'energy_std--'+common_str)
			self.file_loss= create_open_file(self.savefile_dir+'loss--'+common_str)
			#self.file_r2= create_open_file(self.savefile_dir+'r2--'+common_str)
			self.file_phase_hist=create_open_file(self.savefile_dir+'phases_histogram--'+common_str)

			self.file_MC_data= create_open_file(self.savefile_dir+'MC_data--'+common_str)
			self.file_opt_data= create_open_file(self.savefile_dir+'opt_data--'+common_str)


		### timing vector
		self.timing_vec=np.zeros((self.N_iterations+1,),dtype=np.float64)
		

	def _compute_phase_hist(self, phases, weigths):
								
		# compute histogram
		n_bins=40
	
		#binned_phases=np.linspace(-np.pi,np.pi, n_bins, endpoint=True)

		# shift phases to (-pi,pi)
		phases = (phases+np.pi)%(2*np.pi) - np.pi
		#
		# density=False: normalization for MC happens after MPI gathers all data
		if self.mode=='exact':
			phase_hist, bin_edges = np.histogram(phases ,bins=n_bins,range=(-np.pi,np.pi), density=False, weights=weigths)
		elif self.mode=='MC':
			phase_hist, bin_edges = np.histogram(phases ,bins=n_bins,range=(-np.pi,np.pi), density=False, )
			#phase_hist = phase_hist*np.diff(bin_edges)


		phase_hist_tot=np.zeros_like(phase_hist).astype(np.float64)
		self.comm.Allreduce(phase_hist.astype(np.float64), phase_hist_tot, op=MPI.SUM)
		phase_hist_tot/=phase_hist_tot.sum() # normalize histogram
		
		return phase_hist_tot


	def check_point(self, iteration,):
			
		# NN parameters
		file_name='NNparams'+'--iter_{0:05d}--'.format(iteration) + self.file_name
		with open(self.savefile_dir_NN+file_name+'.pkl', 'wb') as handle:
			pickle.dump([self.DNN.params,self.DNN.apply_fun_args,self.MC_tool.log_psi_shift,], handle, protocol=pickle.HIGHEST_PROTOCOL)

	def save_sim_data(self, iteration, loss, r2, phase_hist):

		# data
		self.file_energy.write("{0:d} : {1:0.14f} : {2:0.14f} : {3:0.14f}\n".format(iteration, self.Eloc_mean_g.real , self.Eloc_mean_g.imag, self.E_MC_std_g))
		#self.file_energy_std.write("{0:d} : {1:0.14f}\n".format(iteration, self.E_MC_std_g))
		
		self.file_loss.write("{0:d} : {1:0.14f} : {2:0.14f} : {3:0.14f} : {4:0.10f} : {5:0.10f} : {6:0.10f} : {7:0.10f} : {8:0.10f}\n".format(iteration, r2, self.NG.S_norm, self.NG.F_norm, self.NG.F_log_norm, self.NG.F_phase_norm, self.NG.S_logcond, loss[0], loss[1], ))
		

		MC_data_1="{0:d} : {1:0.4f} : ".format(iteration, self.MC_tool.acceptance_ratio_g[0])
		MC_data_2=' '.join('{0:0.4f}'.format(r) for r in self.MC_tool.acceptance_ratio)+" : "
		MC_data_3=' '.join(str(s) for s in self.MC_tool.s0_g)+" : "
		MC_data_4=' '.join(str(s) for s in self.MC_tool.sf_g)
		self.file_MC_data.write(MC_data_1  +  MC_data_2  +  MC_data_3 +  MC_data_4 + "\n") #		
		

		self.file_opt_data.write("{0:d} : {1:05d} : {2:0.10f} : {3:0.10f} : {4:0.14f} : {5:0.10f}\n".format(iteration, self.NG.counter, self.NG.RK_step_size, self.NG.RK_time, self.NG.delta, self.NG.tol, ))


		
		self.file_phase_hist.write("{0:d} : ".format(iteration) + ''.join("{0:0.6f}, ".format(value) for value in phase_hist) + '\n' )


		# record current iteration number
		self.params_dict['stop_iter']=iteration+1
		
		# update file in data dir
		config_params_yaml = open(self.data_dir + '/config_params.yaml', 'w')
		yaml.dump(self.params_dict, config_params_yaml)
		config_params_yaml.close()

		# flush data files
		self.file_energy.flush()
		self.file_loss.flush()
		self.file_MC_data.flush()
		self.file_opt_data.flush()
		self.file_phase_hist.flush()



	def debug_helper(self,):

		# record DNN params update
		if self.comm.Get_rank()==0:
			self.params_update[:-1,...]=self.params_update[1:,...]
			self.params_update[-1,...]*=0.0


	def run_debug_helper(self, run=False, ):


		#
		##### store data
		# 
		if self.comm.Get_rank()==0:
	
			# check for nans and infs
			if run or ((not np.isfinite(self.NG.S_matrix).all() ) and (not np.isfinite(self.NG.F_vector).all() )):
				
				with open(self.debug_file_SF+'.pkl', 'wb') as handle:

					pickle.dump([self.NG.S_lastiters, self.NG.F_lastiters, self.NG.delta,], 
									handle, protocol=pickle.HIGHEST_PROTOCOL
								)

				with open(self.debug_file_logpsi+'.pkl', 'wb') as handle:

					pickle.dump([self.MC_tool.log_mod_kets_g, self.MC_tool.log_psi_shift_g], 
									handle, protocol=pickle.HIGHEST_PROTOCOL
								)

				with open(self.debug_file_phasepsi+'.pkl', 'wb') as handle:

					pickle.dump([self.MC_tool.phase_kets_g, ], 
									handle, protocol=pickle.HIGHEST_PROTOCOL
								)

				with open(self.debug_file_intkets+'.pkl', 'wb') as handle:

					pickle.dump([self.MC_tool.ints_ket_g, ], 
									handle, protocol=pickle.HIGHEST_PROTOCOL
								)

				with open(self.debug_file_Eloc+'.pkl', 'wb') as handle:

					pickle.dump([self.E_estimator.Eloc_real_g, self.E_estimator.Eloc_imag_g, ], 
									handle, protocol=pickle.HIGHEST_PROTOCOL
								)

				with open(self.debug_file_params_update+'.pkl', 'wb') as handle:

					pickle.dump([self.params_update, ], 
									handle, protocol=pickle.HIGHEST_PROTOCOL
								)

				
		self.comm.Barrier()


	

	def discard_outliars(self,):

		inds=self.E_estimator.inds_outliers

		self.MC_tool.spinstates_ket[inds,...]=0
		self.MC_tool.log_mod_kets[inds]=0.0
		self.MC_tool.phase_kets[inds]=0.0

		self.E_estimator.Eloc_real[inds]=0.0
		self.E_estimator.Eloc_imag[inds]=0.0


	def train(self, start_iter=0):


		# set timer
		t_start=time.time()


		if self.mode=='exact':
			assert(self.N_MC_points==107) # 107 states in the symmetry reduced sector for L=4

			self.MC_tool.ints_ket, self.index, self.inv_index, self.count=self.E_estimator.get_exact_kets()
			integer_to_spinstate(self.MC_tool.ints_ket, self.MC_tool.spinstates_ket, self.DNN.N_features, NN_type=self.DNN.NN_type)



		for iteration in range(start_iter,start_iter+self.N_iterations, 1): 

			#self.comm.Barrier()
			ti=time.time()

			# shift params_update
			self.debug_helper()

			init_iter_str="\n\nITERATION {0:d}, PROCESS_RANK {1:d}:\n\n".format(iteration, self.comm.Get_rank())
			if self.comm.Get_rank()==0:
				print(init_iter_str)
			self.logfile.write(init_iter_str)


			##### determine batchnorm mean and variance
			if self.batchnorm==1:
				self.compute_batchnorm_params(self.DNN.params,len(self.shapes)+1) #



			##### evaluate model
			self.get_training_data(iteration,self.DNN.params)
			#self.get_Stot_data(self.DNN.params)

			#####
			E_str=self.mode + ": E={0:0.14f}, E_std={1:0.14f}.\n".format(self.Eloc_mean_g.real, self.E_MC_std_g, ) 		
			if self.comm.Get_rank()==0:
				E_str+="	with {0:d} unique spin configs.\n".format(np.unique(self.MC_tool.ints_ket_g[-1,...]).shape[0] )
				print(E_str)
			self.logfile.write(E_str)
			
			#exit()

			if self.mode=='exact':
				self.logfile.write('overlap = {0:0.10f}.\n\n'.format(self.Eloc_params_dict['overlap']) )

			
			
			if iteration<self.N_iterations+start_iter:

				#### check point DNN parameters
				if self.comm.Get_rank()==0 and self.save_data:
					self.check_point(iteration)


				#### update DNN parameters
				loss, r2 = self.update_NN_params(iteration)


				##### store simulation data
				if self.mode=='exact':		
					phase_hist = self._compute_phase_hist(self.MC_tool.phase_kets,self.Eloc_params_dict['abs_psi_2'])
				else:
					mod_psi_2=np.exp(2.0*(self.MC_tool.log_mod_kets-self.MC_tool.log_psi_shift))
					phase_hist = self._compute_phase_hist(self.MC_tool.phase_kets,mod_psi_2)


				if self.comm.Get_rank()==0 and self.save_data:
					#if not(self.load_data and (start_iter==iteration)):
					self.save_sim_data(iteration,loss,r2,phase_hist)

			
			prss_time=time.time()-ti
			fin_iter_str="PROCESS_RANK {0:d}, iteration step {1:d} took {2:0.4f} secs.\n".format(self.comm.Get_rank(), iteration, prss_time)
			self.logfile.write(fin_iter_str)
			print(fin_iter_str)
			

			self.timing_vec[iteration-start_iter]=prss_time
			
			self.logfile.flush()
			os.fsync(self.logfile.fileno())


			# synch 
			self.comm.Barrier()

		
		prss_tot_time=time.time()-t_start
		final_str='\n\nPROCESS_RANK {0:d}, total calculation time: {1:0.4f} secs.\n\n\n'.format(self.comm.Get_rank(),prss_tot_time)
		print(final_str)
		self.logfile.write(final_str)
		self.timing_vec[iteration+1-start_iter]=prss_tot_time


		timing_matrix=np.zeros((self.comm.Get_size(),self.N_iterations+1),dtype=np.float64)
		self.comm.Allgather(self.timing_vec, timing_matrix)


		if self.comm.Get_rank()==0 and self.save_data:
			timing_matrix_filename = '/simulation_time--start_iter_{0:d}--'.format(start_iter) + self.file_name + '.txt'
			np.savetxt(self.data_dir+timing_matrix_filename,timing_matrix.T,delimiter=',')
			
		
		# close files
		self.logfile.close()
		self.file_energy.close()
		#self.file_energy_std.close()
		self.file_loss.close()
		#self.file_r2.close()
		self.file_phase_hist.close()


		# store data from last 6 iterations
		self.run_debug_helper(run=True,)



	def compute_batchnorm_params(self,NN_params,N_iter):
	
		ti=time.time()

		#print(self.DNN.apply_fun_args_dyn[2]['mean'])
		for i in range(N_iter):
			
			# draw MC sample
			acceptance_ratio_g = self.MC_tool.sample(self.DNN, compute_phases=False)
			
			self.update_batchnorm_params(self.DNN.NN_architecture, set_fixpoint_iter=True)
			log_psi, phase_psi = self.evaluate_NN_dyn(self.DNN.params, self.MC_tool.spinstates_ket.reshape(self.MC_tool.N_batch,self.MC_tool.N_symm,self.MC_tool.N_sites), )
			self.update_batchnorm_params(self.DNN.NN_architecture, set_fixpoint_iter=False)

			#norm_str="iter: {0:d}, min(log_psi)={1:0.8f}, max(log_psi)={2:0.8f}.".format( i, np.min(np.abs(log_psi)), np.max(np.abs(log_psi)) )
			psi_str="log_psi_bras: min={0:0.8f}, max={1:0.8f}, mean={2:0.8f}; std={3:0.8f}, diff={4:0.8f}.\n".format(np.min(log_psi_bras), np.max(log_psi_bras), np.mean(log_psi_bras), np.std(log_psi_bras), np.max(log_psi_bras)-np.min(log_psi_bras) )
		

			self.logfile.write(psi_str)
			if self.comm.Get_rank()==0:
				print(psi_str)
		#print(self.DNN.apply_fun_args_dyn[2]['mean'], self.DNN.apply_fun_args[2]['mean'])

								
		MC_str="\nweight normalization with final MC acceptance ratio={0:.4f}: took {1:.4f} secs.\n".format(acceptance_ratio_g[0],time.time()-ti)
		self.logfile.write(MC_str)
		if self.comm.Get_rank()==0:
			print(MC_str)


			

	def get_training_data(self,iteration,NN_params):

		##### get spin configs #####
		if self.mode=='exact':
			self.MC_tool.exact(self.evaluate_NN,self.DNN, self.E_estimator)
			
		elif self.mode=='MC':
			ti=time.time()
			
			# sample
			acceptance_ratio_g = self.MC_tool.sample(self.DNN)
			
			MC_str="MC with acceptance ratio={0:.4f}: took {1:.4f} secs.\n".format(acceptance_ratio_g[0],time.time()-ti)
			self.logfile.write(MC_str)
			if self.comm.Get_rank()==0:
				print(MC_str)

			if iteration==0:
				self.MC_tool.thermal=self.thermal # set MC sampler to re-use initial state
		
		# get log_psi statistics
		data_tuple=np.min(self.MC_tool.log_mod_kets), np.max(self.MC_tool.log_mod_kets), np.mean(self.MC_tool.log_mod_kets), np.std(self.MC_tool.log_mod_kets), np.max(self.MC_tool.log_mod_kets)-np.min(self.MC_tool.log_mod_kets)
		psi_str="log_|psi|_kets: min={0:0.8f}, max={1:0.8f}, mean={2:0.8f}; std={3:0.8f}, diff={4:0.8f}.\n".format(*data_tuple )
		self.logfile.write(psi_str)
		print(psi_str)


		##### compute local energies #####
		ti=time.time()
		self.E_estimator.compute_local_energy(self.evaluate_NN,self.DNN,NN_params,self.MC_tool.ints_ket,self.MC_tool.log_mod_kets,self.MC_tool.phase_kets,self.MC_tool.log_psi_shift,self.minibatch_size)
		
		Eloc_str="total local energy calculation took {0:.4f} secs.\n".format(time.time()-ti)
		self.logfile.write(Eloc_str)
		if self.comm.Get_rank()==0:
			print(Eloc_str)


		if self.mode=='exact':
			mod_kets=np.exp(self.MC_tool.log_mod_kets)
			self.psi = mod_kets*np.exp(+1j*self.MC_tool.phase_kets)/np.linalg.norm(mod_kets[self.inv_index])
			abs_psi_2=self.count*np.abs(self.psi)**2

			# print('abs_psi_2')
			# print(mod_kets)
			# print(abs_psi_2)
			# print(np.sum(abs_psi_2), np.linalg.norm(mod_kets[self.inv_index]), self.MC_tool.log_psi_shift, np.mean(self.MC_tool.log_mod_kets), np.max(self.MC_tool.log_mod_kets), np.min(self.MC_tool.log_mod_kets))

			self.Eloc_params_dict=dict(abs_psi_2=abs_psi_2,)
			overlap=np.abs(self.psi[self.inv_index].dot(self.E_estimator.psi_GS_exact))**2
			self.Eloc_params_dict['overlap']=overlap


		
		elif self.mode=='MC':
			self.Eloc_params_dict=dict(N_MC_points=self.N_MC_points)


		#discard outliers
		# if len(self.E_estimator.inds_outliers)>0:
		# 	print('discarding outliars...\n')
		# 	self.discard_outliars()	

		
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
		self.E_estimator.compute_local_energy(self.evaluate_NN,NN_params,self.MC_tool.ints_ket,self.MC_tool.log_mod_kets,self.MC_tool.phase_kets,self.MC_tool.log_psi_shift,SdotS=True)
		self.SdotSloc_mean, SdotS_var, SdotS_diff_real, SdotS_diff_imag = self.E_estimator.process_local_energies(mode=self.mode,Eloc_params_dict=self.Eloc_params_dict,SdotS=True)
		self.SdotS_MC_std=np.sqrt(SdotS_var/self.N_MC_points)


	def update_NN_params(self,iteration):

		ti=time.time()

		if self.optimizer=='RK':
			# compute updated NN parameters
			self.DNN.update_params(self.NG.Runge_Kutta(self.DNN.params,self.batch,self.Eloc_params_dict,self.mode,self.get_training_data))
			loss=self.NG.max_grads
			grads=self.NG.dy_star - 1.0/6.0*(self.NG.dy-self.NG.dy_star)

		else:
			##### compute gradients
			if self.optimizer=='NG':
				# compute enatural gradients
				grads=self.NG.compute(self.DNN.params,self.batch,self.Eloc_params_dict,mode=self.mode)
				loss=self.NG.max_grads
				self.NG.update_params() # update NG params

				S_str="NG: norm(S)={0:0.14f}, norm(F)={1:0.14f}, norm(F_log)={2:0.14f}, norm(F_phase)={3:0.14f}, S_condnum={4:0.14f}\n".format(self.NG.S_norm, self.NG.F_norm, self.NG.F_log_norm, self.NG.F_phase_norm, self.NG.S_logcond) 		
				if self.comm.Get_rank()==0:
					print(S_str)
				self.logfile.write(S_str)


			elif self.optimizer=='adam':
				# compute adam gradients
				grads_MPI=self.DNN.NN_Tree.ravel( self.compute_grad(self.DNN.params,self.batch,self.Eloc_params_dict, iteration) )
				
				# sum up MPI processes
				grads=np.zeros_like(grads_MPI)
				self.comm.Allreduce(grads_MPI._value, grads,  op=MPI.SUM)
				loss=[np.max(grads),0.0]
				
				
			##### apply gradients

			left_ind = 0
			for j, right_ind in enumerate(self.DNN.N_varl_params_vec):
				grads[left_ind:left_ind+right_ind]*=self.learning_rates[j]
				left_ind+=right_ind

				
			self.opt_state = self.opt_update(iteration, self.DNN.NN_Tree.unravel(grads), self.opt_state) 
			self.DNN.update_params(self.get_params(self.opt_state))



			
		##### compute loss
		r2=self.NG.r2_cost
		

		# record gradients
		if self.comm.Get_rank()==0:
			self.params_update[-1,...]=grads





		

		print('GRADS', r2, loss, self.NG.delta)
		#exit()

		# b_str="reg layer params: {}\n".format( self.DNN.params[0][-1] )
		# self.logfile.write(b_str)
		# if self.comm.Get_rank()==0:
		# 	print(b_str)

		grad_str="total gradients/NG calculation took {0:.4f} secs.\n".format(time.time()-ti)
		self.logfile.write(grad_str)
		if self.comm.Get_rank()==0:
			print(grad_str)
		
		return loss, r2


