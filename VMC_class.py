import sys,os,warnings
#sys.stdout.reconfigure(encoding='utf-8')

if sys.platform == "linux" or sys.platform == "linux2": # linux
    path_to_data="../../ED_data/"

elif sys.platform == "darwin": # OS X
	path_to_data=os.path.expanduser('~') + '/Google_Drive/frustration_from_RBM/ED_data/'


from mpi4py import MPI

# python variable scripts
os.environ['LANG']='en_US.UTF-8'
os.environ['LC_ALL']='en_US.UTF-8'
os.environ['PYTHONIOENCODING']='UTF-8'

# multi threading
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
#from scipy.linalg import sqrtm, pinvh

import yaml
import pickle

from cpp_code import Log_Net, Phase_Net
from cpp_code import integer_to_spinstate

from MC_lib import MC_sampler
from energy_lib import Energy_estimator
from optimizer import optimizer
from data_lib import *

from copy import copy
import datetime
import time
np.set_printoptions(threshold=np.inf)

# from cpx_test_weights import *		


class VMC(object):

	def __init__(self,data_dir,params_dict=None,train=True,):

		if params_dict is None:
			self.data_dir=data_dir
			params_dict = yaml.load(open(self.data_dir+'/config_params_init.yaml'),Loader=yaml.FullLoader)
		else:
			self.data_dir=None #data_dir

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
		
		if 'semi_exact' in self.params_dict.keys():
			self.semi_exact=list(int(j) for j in read_str(params_dict['semi_exact'])[0])
		else:
			self.semi_exact=[0,0]


		self.opt=read_str(params_dict['opt'])[0]

		step_size_str=read_str(params_dict['step_size'])[0]
		self.step_sizes=[np.float(step_size_str[0]), np.float(step_size_str[1])]

		self.cost=read_str(params_dict['cost'])[0]
		self.TDVP_opt=read_str(params_dict['TDVP_opt'])[0]
		self.adaptive_step=params_dict['adaptive_step']

		self.grad_update_mode=params_dict['grad_update_mode']
		self.alt_iters=params_dict['alt_iter'] # only effective in real-decoupled mode
		

		self.NN_type=params_dict['NN_type'] # DNN vs CNN
		self.NN_dtype=params_dict['NN_dtype'] # 'real' # # cpx vs real network parameters
		self.NN_shape_str=params_dict['NN_shape_str']
		 
		self.save_data=params_dict['save_data']
		self.load_data=params_dict['load_data']
		self.batchnorm=params_dict['batchnorm']
		self.adaptive_SR_cutoff=params_dict['adaptive_SR_cutoff']


		self.print=params_dict['print']


		# training params
		self.N_iterations=params_dict['N_iterations']
		self.start_iter=params_dict['start_iter']

		### MC sampler
		self.MC_prop_threshold=params_dict['MC_prop_threshold']
		self.thermal=params_dict['MC_thermal']
		self.N_MC_points=params_dict['N_MC_points']
		self.N_MC_chains = params_dict['N_MC_chains'] # number of MC chains to run in parallel
		self.minibatch_size=params_dict['minibatch_size'] # define batch size for GPU evaluation of local energy


		os.environ['OMP_NUM_THREADS']='{0:d}'.format(self.N_MC_chains) # set number of OpenMP threads to run in parallel
		
		if self.NN_dtype=='real-decoupled':
			self.NN_dtype='real'
		

		if self.NN_dtype!='real' and self.NN_dtype!='cpx':
			raise ValueError('Invalid input {} for variable NN_dtype; valid values are real and cpx.'.format(self.NN_dtype))


		# number of processors must fix MC sampling ratio
		if self.mode=='exact':
			assert(self.L==4)
			self.N_batch=self.N_MC_points#
			if self.comm.Get_size()>1:
				print('only one MPI process allowed for "exact" simulation.')
				exit()

			self.N_minibatches=1

		elif self.mode=='MC':
			self.N_batch=self.N_MC_points//self.comm.Get_size()
			if self.comm.Get_rank() < self.N_MC_points%self.comm.Get_size():
				self.N_batch+=1

			self.N_minibatches=1


		elif self.mode=='ED':

			if self.L==4:
				self.N_MC_points=107
				self.N_minibatches=4
			elif self.L==6:
				self.N_MC_points=15804956
				self.N_minibatches=2000
			else:
				raise ValueError('cannot do a full-basis simulation for L>6.')

			self.N_batch=self.N_MC_points//self.comm.Get_size()
			if self.comm.Get_rank() < self.N_MC_points%self.comm.Get_size():
				self.N_batch+=1
		
			# compute index in data file of process
			self.N_batches=np.zeros(self.comm.Get_size(),dtype=np.int)
			self.comm.Allgather([np.array(self.N_batch),  MPI.INT], [self.N_batches, MPI.INT], )
			self.prss_index=np.insert(np.cumsum(self.N_batches),0,0)[self.comm.Get_rank()]


		else:
			print('unrecognized operation mode!')
			

	
		model_params=dict(model=self.NN_type+self.NN_dtype,
						  mode=self.mode,
						  L=self.L,
						  J2=self.J2,
						  opt=self.opt,
						  NNstrct=self.NN_shape_str,
						  MCpts=self.N_MC_points,
						  Nprss=self.comm.Get_size(),
						  NMCchains=self.N_MC_chains,
						)

		
		self.n_iter=10 # define number of iterations to store for debugging purposes
		self._create_file_name(model_params)
		self._create_NN(load_data=self.load_data)	
		self._create_energy_estimator()
		self._create_MC_sampler()
		self._create_optimizer()

		
	
		# load exact data
		ED_data_file  ="data-GS_J1-J2_Lx={0:d}_Ly={1:d}_J1=1.0000_J2={2:0.4f}.txt".format(self.L,self.L,self.J2)
		self.load_file=path_to_data+ED_data_file
		if np.sum(self.semi_exact)>1E-6:
			print('\nstart loading exact data')
			self.load_exact_data()


		# create log file and directory


		# auxiliary variable
		self.prev_it_data=np.zeros(5) 

		if self.save_data:
			if data_dir is None:
				raise ValueError("data_dir cannot be None-type for save_data=True.")

			self._copy_code()
			self._create_logs()


			if self.load_data:
				self._load_data(self.start_iter, truncate_files=True)


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
		

	def _copy_code(self):

		code_dir=self.data_dir+'/code_files/'

		files=['../VMC_class.py', '../MC_lib.py', '../natural_grad.py', '../energy_lib.py', '../cpp_code/sampling_lib.pyx', '../cpp_code/DNN_architectures*', '../cpp_code/reshape_class.py', '../cpp_code/_cpp_funcs',]

		if self.comm.Get_rank()==0:

			if not os.path.exists(code_dir):
				os.makedirs(code_dir)

				for file_or_dir in files:
					os.system('cp -r ' + file_or_dir + ' ' + code_dir)



	def _load_data(self, start_iter, truncate_files=True, repeat=False):
		
		### load MC 
		print(self.comm.Get_rank(),"loading iteration {0:d}".format(start_iter), truncate_files, repeat)
		
		self.comm.Barrier()
		
		if self.mode=='MC':
			with open(self.file_MC_data.name, 'rb') as file: 
				for i in range(start_iter):
					MC_data_str = file.readline()

			MC_data_str=MC_data_str.decode('utf8').rstrip().split(' : ') 

			it_MC, acceptance_ratio_g, acceptance_ratios, s0_g, sf_g =  MC_data_str

			self.MC_tool.acceptance_ratio_g[0]=np.float64(acceptance_ratio_g)
			self.MC_tool.s0_g=np.array([self.E_estimator.basis_type(s0) for s0 in s0_g.split(' ')] )
			self.MC_tool.sf_g=np.array([self.E_estimator.basis_type(sf) for sf in sf_g.split(' ')] )

			m_l=self.N_MC_chains*self.comm.Get_rank()
			m_r=self.N_MC_chains*(self.comm.Get_rank()+1)

			if self.NN_dtype=='real':
				self.DNN_log._init_MC_data(s0_vec=self.MC_tool.s0_g[m_l:m_r], sf_vec=self.MC_tool.sf_g[m_l:m_r], )
			else:
				self.DNN._init_MC_data(s0_vec=self.MC_tool.s0_g[m_l:m_r], sf_vec=self.MC_tool.sf_g[m_l:m_r], )



		### load DNN params
		file_name='/NN_params/NNparams'+'--iter_{0:05d}'.format(start_iter)

		if self.NN_dtype=='real':

			with open(self.data_dir+file_name+'.pkl', 'rb') as handle:
				self.DNN_log.params, self.DNN_phase.params, \
				self.DNN_log.apply_fun_args, self.DNN_phase.apply_fun_args, \
				self.MC_tool.log_psi_shift = pickle.load(handle)
				
			self.opt_log.init_opt_state(self.DNN_log.params)
			self.opt_phase.init_opt_state(self.DNN_phase.params)


			file_name='/NN_params/NNparams'+'--iter_{0:05d}'.format(start_iter-1) 
			with open(self.data_dir+file_name+'.pkl', 'rb') as handle:
				DNN_params_log_old, DNN_params_phase_old, _ ,_ , _ = pickle.load(handle)

			
			if self.opt_log.cost=='SR':
				self.opt_log.NG.nat_grad_guess[:]  = (self.DNN_log.NN_Tree.ravel(DNN_params_log_old)-self.DNN_log.NN_Tree.ravel(self.DNN_log.params) )/self.opt_log.step_size

			if self.opt_phase.cost=='SR':
				self.opt_phase.NG.nat_grad_guess[:]= (self.DNN_phase.NN_Tree.ravel(DNN_params_phase_old)-self.DNN_phase.NN_Tree.ravel(self.DNN_phase.params) )/self.opt_phase.step_size


			### load opt data
			load_opt_data(self.opt_log, self.file_opt_data_log.name, start_iter)
			load_opt_data(self.opt_phase, self.file_opt_data_phase.name, start_iter)

		else:

			with open(self.data_dir+file_name+'.pkl', 'rb') as handle:
				self.DNN.params, _, \
				self.DNN.apply_fun_args, _, \
				self.MC_tool.log_psi_shift = pickle.load(handle)
				
			self.opt.init_opt_state(self.DNN.params)
		
			file_name='/NN_params/NNparams'+'--iter_{0:05d}'.format(start_iter-1) 
			with open(self.data_dir+file_name+'.pkl', 'rb') as handle:
				DNN_params_old, _, _ ,_ , _ = pickle.load(handle)

			
			if self.opt.cost=='SR':
				self.opt.NG.nat_grad_guess[:]  = (self.DNN.NN_Tree.ravel(DNN_params_old)-self.DNN.NN_Tree.ravel(self.DNN.params) )/self.opt.step_size


			### load opt data
			load_opt_data(self.opt, self.file_opt_data.name, start_iter)
			

		### load energy

		with open(self.file_energy.name,'rb') as file:
			for i in range(start_iter):
					energy_data_str = file.readline()

		energy_data_str=energy_data_str.decode('utf8').rstrip().split(' : ')				

		it_E, Eloc_mean_g_real , Eloc_mean_g_imag, Eloc_std_g, E_MC_std_g = energy_data_str
		self.prev_it_data[0], self.prev_it_data[1], self.prev_it_data[2]=np.float64(Eloc_mean_g_real), np.float(Eloc_mean_g_imag), np.float64(E_MC_std_g)
			


		# truncate remaining files
		self.comm.Barrier()
		if truncate_files and self.comm.Get_rank()==0:

			if repeat:
				start_iter+=1
		
			for file in self.all_data_files:
				truncate_file(file, start_iter)

			self._create_open_data_files(load_data=True)
		
		self.comm.Barrier()

		#####
		if self.mode=='MC':
			if self.NN_dtype=='real':
				assert(int(it_MC)+1==self.opt_log.iteration)
				assert(int(it_MC)+1==self.opt_phase.iteration)
			else:
				assert(int(it_MC)+1==self.opt.iteration)
			assert(int(it_MC)==int(it_E))


	def load_exact_data(self,):

		from pandas import read_csv


		if self.semi_exact[0]==1 and self.semi_exact[1]==0:
			assert(self.grad_update_mode=='phase')
		elif self.semi_exact[0]==0 and self.semi_exact[1]==1:
			assert(self.grad_update_mode=='log_mod')


		ints_ket_exact=read_csv(self.load_file, header=None, dtype=self.E_estimator.basis_type, delimiter=' ',usecols=[0,]).to_numpy().squeeze()
		log_psi_exact =read_csv(self.load_file, header=None, dtype=np.float64, delimiter=' ',usecols=[1,]).to_numpy().squeeze()

		if self.sign>0:
			phase_psi_exact=read_csv(self.load_file, header=None, dtype=np.float64, delimiter=' ',usecols=[4,]).to_numpy().squeeze()
		else:
			phase_psi_exact=read_csv(self.load_file, header=None, dtype=np.float64, delimiter=' ',usecols=[3,]).to_numpy().squeeze() 
		phase_psi_exact=np.pi*(1.0-phase_psi_exact)/2.0


		if self.NN_dtype=='real':
			if self.semi_exact[0]:
				self.DNN_log.load_exact_data(ints_ket_exact,log_psi_exact=log_psi_exact,phase_psi_exact=None)
			if self.semi_exact[1]:
				self.DNN_phase.load_exact_data(ints_ket_exact,log_psi_exact=None,phase_psi_exact=phase_psi_exact)
		else:
			if self.semi_exact[0]:
				self.DNN.load_exact_data(ints_ket_exact,log_psi_exact=log_psi_exact,phase_psi_exact=phase_psi_exact)


		# clear memory
		del ints_ket_exact, log_psi_exact, phase_psi_exact

		print('finished loading exact data\n')
		#exit()



	def _create_NN(self, load_data=False):

		NN_shape_str, M = read_str(self.NN_shape_str)
		self.shapes=tuple({} for _ in range(M) )

		
		if self.NN_type == 'DNN':
			neurons=tuple([] for _ in range(M))
			for j in range(M):
				for neuron in NN_shape_str[j].split('--'):
					neurons[j].append(int(neuron))

				assert(neurons[j][0]==self.L**2)	

			for j in range(M):
				for i in range(len(neurons[j])-1):
					self.shapes[j]['layer_{0:d}'.format(i+1)]=[neurons[j][i],neurons[j][i+1]]

		elif self.NN_type == 'CNN':

			filters=tuple([] for _ in range(M))
			out_chans=tuple([] for _ in range(M))

			for j in range(M):
				for layer in NN_shape_str[j].split('--'):
					layer_filter, output_channel = layer.split('-')	
					filters[j].append(tuple(np.array(layer_filter.split('x'),dtype=int)))
					out_chans[j].append(int(output_channel))

			for j in range(M):
				for i in range(len(filters[j])):

					self.shapes[j]['layer_{0:d}'.format(i+1)]=[filters[j][i], out_chans[j][i],]


		### create Neural network
		if self.NN_dtype=='real':
			self.DNN_log=Log_Net(self.comm, self.shapes[0], self.N_MC_chains, self.NN_type, self.NN_dtype, seed=self.seed, prop_threshold=self.MC_prop_threshold )
			self.DNN_phase=Phase_Net(self.comm, self.shapes[1], self.N_MC_chains, self.NN_type, self.NN_dtype, seed=self.seed, prop_threshold=self.MC_prop_threshold )
			self.DNN=None
			self.N_symm = self.DNN_log.N_symm
		
			params_str="\nlog-net params:   {0:d}\nphase-net params: {1:d}\n".format(self.DNN_log.N_varl_params, self.DNN_phase.N_varl_params)

		else:
			self.DNN=Log_Net(self.comm, self.shapes[0], self.N_MC_chains, self.NN_type, self.NN_dtype, seed=self.seed, prop_threshold=self.MC_prop_threshold )
			self.DNN_log, self.DNN_phase = None, None
			self.N_symm = self.DNN.N_symm

			params_str="\ncpx-net params: {0:d}\n".format(self.DNN.N_varl_params)

		print(params_str)
		


		# print(self.DNN_log.params[2])
		# exit()

		#self.DNN.params=[(W_real, W_imag), (), ()]

		self.N_features=self.N_symm * self.L**2
		

	def _create_optimizer(self):

		if self.NN_dtype=='real':

			# log net
			self.opt_log   = optimizer(self.comm, self.opt[0], self.cost[0], self.mode, self.NN_dtype, self.DNN_log.NN_Tree, label='LOG',  step_size=self.step_sizes[0], adaptive_step=self.adaptive_step, adaptive_SR_cutoff=self.adaptive_SR_cutoff )
			self.opt_log.init_global_variables(self.N_MC_points, self.N_batch, self.DNN_log.N_varl_params, self.n_iter, self.N_minibatches, )
			self.opt_log.define_grad_func(NN_evaluate=self.DNN_log.evaluate, TDVP_opt=self.TDVP_opt[0], reestimate_local_energy=self.reestimate_local_energy ) # reestimate_local_energy_log
			self.opt_log.init_opt_state(self.DNN_log.params)
			# phase net
			self.opt_phase = optimizer(self.comm, self.opt[1], self.cost[1], self.mode, self.NN_dtype, self.DNN_phase.NN_Tree, label='PHASE', step_size=self.step_sizes[1], adaptive_step=self.adaptive_step, adaptive_SR_cutoff=self.adaptive_SR_cutoff )
			self.opt_phase.init_global_variables(self.N_MC_points, self.N_batch, self.DNN_phase.N_varl_params, self.n_iter, self.N_minibatches,)
			self.opt_phase.define_grad_func(NN_evaluate=self.DNN_phase.evaluate, TDVP_opt=self.TDVP_opt[1], reestimate_local_energy=self.E_estimator.reestimate_local_energy_phase )
			self.opt_phase.init_opt_state(self.DNN_phase.params)


			# define variable to keep track of the DNN params update
			if self.comm.Get_rank()==0:
				self.params_log_update_lastiters=np.zeros((self.n_iter,self.DNN_log.N_varl_params),dtype=np.float64)
				self.params_phase_update_lastiters=np.zeros((self.n_iter,self.DNN_phase.N_varl_params),dtype=np.float64)
			else:
				self.params_log_update_lastiters=np.array([[None],[None]])
				self.params_phase_update_lastiters=np.array([[None],[None]])
		


		else:

			self.opt = optimizer(self.comm, self.opt[0], self.cost[0], self.mode, self.NN_dtype, self.DNN.NN_Tree, label='CPX',  step_size=self.step_sizes[0], adaptive_step=self.adaptive_step, adaptive_SR_cutoff=self.adaptive_SR_cutoff )
			self.opt.init_global_variables(self.N_MC_points, self.N_batch, self.DNN.N_varl_params, self.n_iter, self.N_minibatches,)
			self.opt.define_grad_func(NN_evaluate=self.DNN.evaluate, NN_evaluate_log=self.DNN.evaluate_log, NN_evaluate_phase=self.DNN.evaluate_phase, TDVP_opt=self.TDVP_opt[0], reestimate_local_energy=self.reestimate_local_energy ) #reestimate_local_energy_log
			self.opt.init_opt_state(self.DNN.params)


			# define variable to keep track of the DNN params update
			if self.comm.Get_rank()==0:
				self.params_update_lastiters=np.zeros((self.n_iter,self.DNN.N_varl_params),dtype=np.float64)
			else:
				self.params_update_lastiters=np.array([[None],[None]])
				

		self.r2=np.zeros(2)

	

	def _create_energy_estimator(self):
		### Energy estimator

		if self.mode=='ED':
			if self.NN_dtype=='real':
				self.E_estimator	=Energy_estimator(self.comm,self.DNN_log,self.DNN_phase,self.mode,self.J2,self.N_MC_points,self.N_batch,self.L,self.N_symm,self.sign, self.minibatch_size, self.semi_exact) # contains all of the physics
				#self.E_estimator_log=Energy_estimator(self.comm,self.DNN_log,self.DNN_phase,self.mode,self.J2,self.N_MC_points,self.N_batch,self.L,self.N_symm,self.sign, self.minibatch_size, self.semi_exact) # contains all of the physics
			else:
				self.E_estimator	=Energy_estimator(self.comm,self.DNN,None,self.mode,self.J2,self.N_MC_points,self.N_batch,self.L,self.N_symm,self.sign, self.minibatch_size, self.semi_exact) # contains all of the physics
				#self.E_estimator_log=Energy_estimator(self.comm,self.DNN,None,self.mode,self.J2,self.N_MC_points,self.N_batch,self.L,self.N_symm,self.sign, self.minibatch_size, self.semi_exact) # contains all of the physics
			
		else:
			if self.NN_dtype=='real':
				self.E_estimator	=Energy_estimator(self.comm,self.DNN_log,self.DNN_phase,self.mode,self.J2,self.N_MC_points,self.N_batch,self.L,self.N_symm,self.sign, self.minibatch_size, self.semi_exact) # contains all of the physics
				#self.E_estimator_log=Energy_estimator(self.comm,self.DNN_log,self.DNN_phase,self.mode,self.J2,self.N_MC_points,self.N_batch,self.L,self.N_symm,self.sign, self.minibatch_size, self.semi_exact) # contains all of the physics
			else:
				self.E_estimator	=Energy_estimator(self.comm,self.DNN,None,self.mode,self.J2,self.N_MC_points,self.N_batch,self.L,self.N_symm,self.sign, self.minibatch_size, self.semi_exact) # contains all of the physics
				#self.E_estimator_log=Energy_estimator(self.comm,self.DNN,None,self.mode,self.J2,self.N_MC_points,self.N_batch,self.L,self.N_symm,self.sign, self.minibatch_size, self.semi_exact) # contains all of the physics
			

		self.E_estimator.init_global_params(self.N_MC_points,self.n_iter)
		#self.E_estimator_log.init_global_params(self.N_MC_points,self.n_iter)
		

		
	def _create_MC_sampler(self, ):
		### initialize MC sampler variables
		self.MC_tool=MC_sampler(self.comm,self.N_MC_chains)
		self.MC_tool.init_global_vars(self.L,self.N_MC_points,self.N_batch,self.N_minibatches,self.N_symm,self.NN_type,self.E_estimator.basis_type,self.E_estimator.MPI_basis_dtype,self.n_iter,self.mode)
		
		if self.NN_type=='DNN':
			self.input_shape=(-1,self.N_symm,self.L**2)
		elif self.NN_type=='CNN':
			if self.cost[0]=='SR':
				self.input_shape=(-1,self.N_symm,1,self.L,self.L)
			else:
				self.input_shape=(-1,1,self.L,self.L)

		
		#self.MC_tool_log=MC_sampler(self.comm,self.N_MC_chains)
		#self.MC_tool_log.init_global_vars(self.L,self.N_MC_points,self.N_batch,self.N_minibatches,self.N_symm,self.NN_type,self.E_estimator_log.basis_type,self.E_estimator_log.MPI_basis_dtype,self.n_iter,self.mode)
		


	def _create_file_name(self,model_params,extra_label=''):
		file_name = ''
		for key,value in model_params.items():
			file_name += ( key+'_{}'.format(value)+'-' )
		file_name=file_name[:-1]
		self.file_name=file_name+extra_label

	def _create_open_data_files(self, load_data):

		self.all_data_files=[]

		self.file_energy= create_open_file(self.savefile_dir+'energy'+self.common_str,load_data)
		self.file_phase_hist=create_open_file(self.savefile_dir+'phases_histogram'+self.common_str,load_data)
		self.file_MC_data= create_open_file(self.savefile_dir+'MC_data'+self.common_str,load_data)
		
		self.all_data_files.extend([self.file_energy,self.file_phase_hist,self.file_MC_data,])


		if self.NN_dtype=='real':

			self.file_loss_log= create_open_file(self.savefile_dir+'loss_log'+self.common_str,load_data)
			self.file_loss_phase= create_open_file(self.savefile_dir+'loss_phase'+self.common_str,load_data)

			self.file_opt_data_log= create_open_file(self.savefile_dir+'opt_data_log'+self.common_str,load_data)
			self.file_opt_data_phase= create_open_file(self.savefile_dir+'opt_data_phase'+self.common_str,load_data)

			self.all_data_files.extend([self.file_loss_log,self.file_opt_data_log, 
										self.file_loss_phase,self.file_opt_data_phase ])


		
			self.file_S_eigvals_log=create_open_file(self.savefile_dir+'eigvals_S_matrix_log'+self.common_str,load_data)
			self.file_S_eigvals_phase=create_open_file(self.savefile_dir+'eigvals_S_matrix_phase'+self.common_str,load_data)
			
			self.file_VF_overlap_log=create_open_file(self.savefile_dir+'overlap_VF_log'+self.common_str,load_data)
			self.file_VF_overlap_phase=create_open_file(self.savefile_dir+'overlap_VF_phase'+self.common_str,load_data)

			self.file_SNR_exact_log=create_open_file(self.savefile_dir+'SNR_exact_log'+self.common_str,load_data)
			self.file_SNR_exact_phase=create_open_file(self.savefile_dir+'SNR_exact_phase'+self.common_str,load_data)

			self.file_SNR_gauss_log=create_open_file(self.savefile_dir+'SNR_gauss_log'+self.common_str,load_data)
			self.file_SNR_gauss_phase=create_open_file(self.savefile_dir+'SNR_gauss_phase'+self.common_str,load_data)
			
			self.all_data_files.extend([self.file_S_eigvals_log,self.file_VF_overlap_log,self.file_SNR_exact_log,self.file_SNR_gauss_log, 
										self.file_S_eigvals_phase,self.file_VF_overlap_phase,self.file_SNR_exact_phase,self.file_SNR_gauss_phase ])

		else:

			self.file_loss= create_open_file(self.savefile_dir+'loss'+self.common_str,load_data)
			self.file_opt_data= create_open_file(self.savefile_dir+'opt_data'+self.common_str,load_data)

			self.all_data_files.extend([self.file_loss,self.file_opt_data])

			
			self.file_S_eigvals=create_open_file(self.savefile_dir+'eigvals_S_matrix'+self.common_str,load_data)
			self.file_VF_overlap=create_open_file(self.savefile_dir+'overlap_VF'+self.common_str,load_data)
			self.file_SNR_exact=create_open_file(self.savefile_dir+'SNR_exact'+self.common_str,load_data)
			self.file_SNR_gauss=create_open_file(self.savefile_dir+'SNR_gauss'+self.common_str,load_data)
			
			self.all_data_files.extend([self.file_S_eigvals,self.file_VF_overlap,self.file_SNR_exact,self.file_SNR_gauss,])
		

	def _create_logs(self):

		# sys_data=''
		
		# if self.comm.Get_rank()==0:
		#	 sys_time=datetime.datetime.now()
		#	 sys_data="{0:d}-{1:02d}-{2:02d}_{3:02d}:{4:02d}:{5:02d}_".format(sys_time.year, sys_time.month, sys_time.day, sys_time.hour, sys_time.minute, sys_time.second)
		#	 #sys_data="{0:d}-{1:02d}-{2:02d}_".format(sys_time.year,sys_time.month,sys_time.day,)

		# # broadcast sys_data
		# sys_data = self.comm.bcast(sys_data, root=0)


		# self.sys_time=sys_data + self.opt

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

		# redirect warnings to log
		def customwarn(message, category, filename, lineno, file=None, line=None):
			s='\n'+warnings.formatwarning(message, category, filename, lineno)+'\n'
			self.logfile.write(s)
		

		# logfile name
		#logfile_name= 'LOGFILE--MPIprss_{0:d}--'.format(self.comm.Get_rank()) + self.file_name + '.txt'
		logfile_name= 'LOGFILE--MPIprss_{0:d}'.format(self.comm.Get_rank()) + '.txt'
		if self.comm.Get_rank()==0:
			self.logfile = create_open_file(logfile_dir+logfile_name,self.load_data,binary=False)
		else:
			self.logfile = None
		
		self.E_estimator.logfile=self.logfile
		#self.E_estimator_log.logfile=self.logfile
			
		if self.NN_dtype=='real':
			self.opt_log.logfile=self.logfile
			self.opt_phase.logfile=self.logfile
		else:
			self.opt.logfile=self.logfile

		

		# redirect std out
		if not self.print:
			warnings.showwarning = customwarn
			sys.stdout = self.logfile
			sys.stderr = self.logfile
			

		
		if self.NN_dtype=='real':
			self.debug_file_SF_log	   =self.savefile_dir_debug + 'debug-SF_data_log'			#+'--' + self.file_name
			self.debug_file_SF_phase	 =self.savefile_dir_debug + 'debug-SF_data_phase'
		else:
			self.debug_file_SF	   =self.savefile_dir_debug + 'debug-SF_data' 
		
		self.debug_file_logpsi	   =self.savefile_dir_debug + 'debug-logpsi_data'		#+'--' + self.file_name
		self.debug_file_phasepsi	 =self.savefile_dir_debug + 'debug-phasepsi_data'	  #+'--' + self.file_name
		self.debug_file_intkets	  =self.savefile_dir_debug + 'debug-intkets_data'	   #+'--' + self.file_name
		self.debug_file_Eloc		 =self.savefile_dir_debug + 'debug-Eloc_data'		  #+'--' + self.file_name
		self.debug_file_params_update=self.savefile_dir_debug + 'debug-params_update_data' #+'--' + self.file_name
		

		if self.save_data:
			# data files
			#common_str = '--'  self.file_name + '.txt'
			self.common_str = '.txt'

			self._create_open_data_files(self.load_data)	

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
		if self.mode=='exact' or self.mode=='ED':
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
		#file_name='NNparams'+'--iter_{0:05d}--'.format(iteration) + self.file_name
		file_name='NNparams'+'--iter_{0:05d}'.format(iteration)

		if self.NN_dtype=='real': 
			with open(self.savefile_dir_NN+file_name+'.pkl', 'wb') as handle:
				pickle.dump([self.DNN_log.params, self.DNN_phase.params, 
							 self.DNN_log.apply_fun_args, self.DNN_phase.apply_fun_args,
							 self.MC_tool.log_psi_shift,
							 ], handle, protocol=pickle.HIGHEST_PROTOCOL)
		else:
			with open(self.savefile_dir_NN+file_name+'.pkl', 'wb') as handle:
				pickle.dump([self.DNN.params, None, 
							 self.DNN.apply_fun_args, None,
							 self.MC_tool.log_psi_shift,
							 ], handle, protocol=pickle.HIGHEST_PROTOCOL)



	def save_sim_data(self, iteration, grads_max, r2, phase_hist):

		# data
		en_data="{0:d} : {1:0.14f} : {2:0.14f} : {3:0.14f} : {4:0.14f}\n".format(iteration, self.Eloc_mean_g.real , self.Eloc_mean_g.imag, self.Eloc_std_g, self.E_MC_std_g)
		self.file_energy.write(en_data.encode('utf8'))

		
		######################################################


		if self.mode=='MC':

			MC_data_1="{0:d} : {1:0.4f} : ".format(iteration, self.MC_tool.acceptance_ratio_g[0])
			MC_data_2=' '.join('{0:0.4f}'.format(r) for r in self.MC_tool.acceptance_ratio)+" : "
			MC_data_3=' '.join(str(s) for s in self.MC_tool.s0_g)+" : "
			MC_data_4=' '.join(str(s) for s in self.MC_tool.sf_g)
			MC_str=MC_data_1  +  MC_data_2  +  MC_data_3 +  MC_data_4 + "\n" 
			self.file_MC_data.write(MC_str.encode('utf8')) #	


		######################################################


		phase_str="{0:d} : ".format(iteration) + ''.join("{0:0.6f}, ".format(value) for value in phase_hist) + '\n' 
		self.file_phase_hist.write(phase_str.encode('utf8'))
	


		######################################################

		if self.NN_dtype=='real':
			store_loss(iteration,r2[0], grads_max[0],self.file_loss_log,self.opt_log)
			store_loss(iteration,r2[1], grads_max[1],self.file_loss_phase,self.opt_phase)

			store_opt_data(iteration,self.file_opt_data_log,self.opt_log)
			store_opt_data(iteration,self.file_opt_data_phase,self.opt_phase)

			if self.opt_log.cost=='SR':
				self.file_S_eigvals_log.write(("{0:d} : ".format(iteration) + ''.join("{0:0.15f}, ".format(value) for value in self.opt_log.NG.S_eigvals) + '\n' ).encode('utf8'))
				self.file_VF_overlap_log.write(("{0:d} : ".format(iteration) + ''.join("{0:0.15f}, ".format(value) for value in self.opt_log.NG.VF_overlap) + '\n' ).encode('utf8'))
				self.file_SNR_gauss_log.write(("{0:d} : ".format(iteration) + ''.join("{0:0.15f}, ".format(value) for value in self.opt_log.NG.SNR_gauss) + '\n' ).encode('utf8'))
				self.file_SNR_exact_log.write(("{0:d} : ".format(iteration) + ''.join("{0:0.15f}, ".format(value) for value in self.opt_log.NG.SNR_exact) + '\n' ).encode('utf8'))
				
			if self.opt_phase.cost=='SR':
				self.file_S_eigvals_phase.write(("{0:d} : ".format(iteration) + ''.join("{0:0.15f}, ".format(value) for value in self.opt_phase.NG.S_eigvals) + '\n' ).encode('utf8'))
				self.file_VF_overlap_phase.write(("{0:d} : ".format(iteration) + ''.join("{0:0.15f}, ".format(value) for value in self.opt_phase.NG.VF_overlap) + '\n' ).encode('utf8'))
				self.file_SNR_exact_phase.write(("{0:d} : ".format(iteration) + ''.join("{0:0.15f}, ".format(value) for value in self.opt_phase.NG.SNR_exact) + '\n' ).encode('utf8'))
				self.file_SNR_gauss_phase.write(("{0:d} : ".format(iteration) + ''.join("{0:0.15f}, ".format(value) for value in self.opt_phase.NG.SNR_gauss) + '\n' ).encode('utf8'))



		else:
			store_loss(iteration,r2[0], grads_max[0],self.file_loss,self.opt)
			store_opt_data(iteration,self.file_opt_data,self.opt)

			if self.opt.cost=='SR':
				self.file_S_eigvals.write(("{0:d} : ".format(iteration) + ''.join("{0:0.15f}, ".format(value) for value in self.opt.NG.S_eigvals) + '\n' ).encode('utf8'))
				self.file_VF_overlap.write(("{0:d} : ".format(iteration) + ''.join("{0:0.15f}, ".format(value) for value in self.opt.NG.VF_overlap) + '\n' ).encode('utf8'))
				self.file_SNR_gauss.write(("{0:d} : ".format(iteration) + ''.join("{0:0.15f}, ".format(value) for value in self.opt.NG.SNR_gauss) + '\n' ).encode('utf8'))
				self.file_SNR_exact.write(("{0:d} : ".format(iteration) + ''.join("{0:0.15f}, ".format(value) for value in self.opt.NG.SNR_exact) + '\n' ).encode('utf8'))
			
	

		######################################################

		# record current iteration number
		self.params_dict['stop_iter']=iteration+1
		
		# update file in data dir
		config_params_yaml = open(self.data_dir + '/config_params.yaml', 'w')
		yaml.dump(self.params_dict, config_params_yaml)
		config_params_yaml.close()


		######################################################

		# flush data files
		flush_all_datafiles(self.all_data_files)



	def save_all_data(self,iteration,start_iter):

		if iteration<self.N_iterations+start_iter:

			#### check point DNN parameters
			if self.comm.Get_rank()==0 and self.save_data:
				self.check_point(iteration)


			#### update DNN parameters
			if self.NN_dtype=='real':
				grads_max = self.update_NN_params_real(iteration)
			else:
				grads_max = self.update_NN_params_cpx(iteration)


			##### store simulation data
			if self.mode in ['ED','exact']:		
				phase_hist = self._compute_phase_hist(self.MC_tool.phase_kets,self.Eloc_params_dict_log['abs_psi_2'])
			else:
				mod_psi_2=np.exp(2.0*(self.MC_tool.log_mod_kets-self.MC_tool.log_psi_shift))
				phase_hist = self._compute_phase_hist(self.MC_tool.phase_kets,mod_psi_2)

			# print(phase_hist)
			# exit()

			if self.comm.Get_rank()==0 and self.save_data:
				#if not(self.load_data and (start_iter==iteration)):
				self.save_sim_data(iteration,grads_max,self.r2,phase_hist)


		# synch repeat_iteration data
		self.prev_it_data[0], self.prev_it_data[1], self.prev_it_data[2]=self.Eloc_mean_g.real, self.Eloc_mean_g.imag, self.E_MC_std_g
			

	def debug_helper(self,):

		# record DNN params update
		if self.comm.Get_rank()==0:
			
			if self.NN_dtype=='real':

				self.params_log_update_lastiters[:-1,...]=self.params_log_update_lastiters[1:,...]
				self.params_log_update_lastiters[-1,...]*=0.0

				self.params_phase_update_lastiters[:-1,...]=self.params_phase_update_lastiters[1:,...]
				self.params_phase_update_lastiters[-1,...]*=0.0

			else:

				self.params_update_lastiters[:-1,...]=self.params_update_lastiters[1:,...]
				self.params_update_lastiters[-1,...]*=0.0



	def run_debug_helper(self, run=False,):

		# set default flag to False
		exit_flag=False 

		if self.NN_dtype=='real':
			opt_flag=(not self.opt_log.is_finite ) or (not self.opt_phase.is_finite )
		else:
			opt_flag=(not self.opt.is_finite )

		#
		##### store data
		# 
		if self.comm.Get_rank()==0:
	
			# check for nans and infs
			if run or opt_flag or (not np.isfinite(self.Eloc_mean_g).all() ):
				

				if self.NN_dtype=='real':
					store_debug_helper_data(self.debug_file_SF_log,self.opt_log)
					store_debug_helper_data(self.debug_file_SF_phase,self.opt_phase)

					with open(self.debug_file_params_update+'.pkl', 'wb') as handle:
						pickle.dump([self.params_log_update_lastiters, self.params_phase_update_lastiters,], 
										handle, protocol=pickle.HIGHEST_PROTOCOL
									)

				else:
					store_debug_helper_data(self.debug_file_SF,self.opt)

					with open(self.debug_file_params_update+'.pkl', 'wb') as handle:
						pickle.dump([self.params_update_lastiters, None,], 
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


				# set exit variable and bcast it to all processes
				if not run: 
					exit_flag=True
		
		exit_flag = self.comm.bcast(exit_flag, root=0)
				
		

		self.comm.Barrier()



		if exit_flag:
			exit_str="\nEncountered nans or infs!\nExiting simulation...\n"
			print(exit_str)
			#self.logfile.write(exit_str)
			exit()


	def repeat_iteration(self,iteration,Eloc_mean_g,E_MC_std_g, go_back_iters=0, load_data=True):

		repeat=False
		if iteration>go_back_iters+1 and self.mode=='MC':

			Eloc_mean_prev=self.prev_it_data[0]
			
			_c1=Eloc_mean_prev-Eloc_mean_g.real
			_c2=np.abs(Eloc_mean_g.imag)
			_c3=6.0*self.prev_it_data[2] - E_MC_std_g 

			_b1=np.abs(_c1) > 2.0 and (_c1<0.0)
			_b2=_c2 > 5.0*E_MC_std_g 
			_b3=_c3 < 0.0

			
			if (_b1 or _b2 or _b3) and (Eloc_mean_prev<0.0): # and Eloc_mean_prev < 0.0: 

				data_tuple=(iteration, Eloc_mean_g.real, Eloc_mean_g.imag, E_MC_std_g,)

				if _b1:
					mssg="!!!  restarting iteration {0:d}: E={1:0.6f}, E_imag={2:0.10f}, E_std={3:0.10f}, E_mean_drop={4:0.10f}  !!!".format( *data_tuple, _c1, )
				elif _b2:
					mssg="!!!  restarting iteration {0:d}: E={1:0.6f}, E_imag={2:0.10f}, E_std={3:0.10f}, E_imag_check={4:0.10f}  !!!".format( *data_tuple, _c2, )
				elif _b3:
					mssg="!!!  restarting iteration {0:d}: E={1:0.6f}, E_imag={2:0.10f}, E_std={3:0.10f}, E_std_check={4:0.10f}  !!!".format(*data_tuple, _c3, )


				if self.comm.Get_rank()==0:
					print(mssg)
				#self.logfile.write(mssg)
				
				# load data
				if load_data:
					self.comm.Barrier()
					self._load_data(iteration-1-go_back_iters, truncate_files=True, repeat=True)
					iteration=iteration-go_back_iters

				repeat=True
		else:
			print('iteration {0:d} sample checks passed.'.format(iteration))

		
		
		return repeat, iteration


	def train(self, start_iter=0):


		# print("remove norm from real net")
		# exit()

		# set timer
		t_start=time.time()
		
		if self.mode=='exact':
			assert(self.N_MC_points==107) # 107 states in the symmetry reduced sector for L=4

			self.MC_tool.ints_ket, self.index, self.inv_index, self.MC_tool.count=self.E_estimator.get_exact_kets()
			integer_to_spinstate(self.MC_tool.ints_ket, self.MC_tool.spinstates_ket, self.N_features, NN_type=self.NN_type)

			# required to train independent real nets with RK
			#self.MC_tool_log.ints_ket, self.index, self.inv_index, self.MC_tool_log.count=self.E_estimator_log.get_exact_kets()
			#integer_to_spinstate(self.MC_tool_log.ints_ket, self.MC_tool_log.spinstates_ket, self.N_features, NN_type=self.NN_type)

		elif self.mode=='ED':


			self.E_estimator.load_exact_basis(self.NN_type,self.MC_tool,self.N_features, self.load_file, self.prss_index)
	
			# required to train independent real nets with RK
			#self.E_estimator_log.load_exact_basis(self.NN_type,self.MC_tool_log,self.N_features, self.load_file, self.prss_index)
	

			data_str="loading data took {0:.4f} secs.\n".format(time.time()-t_start)
			#self.logfile.write(MC_str)
			if self.comm.Get_rank()==0:
				print(data_str)



		iteration=start_iter
		while iteration < start_iter+self.N_iterations:
	
			ti=time.time()

			
			init_iter_str="\n\n\nITERATION {0:d}, PROCESS_RANK {1:d}:\n\n".format(iteration, self.comm.Get_rank())
			if self.comm.Get_rank()==0:
				print(init_iter_str)

			
			##### evaluate model

			if self.logfile is not None:
				self.logfile.flush()

			self.get_Hessian(iteration,)
			

			self.get_training_data(iteration,)

			E_str="total Eloc calcumation took {0:.4f} secs.\n".format(time.time()-ti)
			#self.logfile.write(MC_str)
			if self.comm.Get_rank()==0:
				print(E_str)



			if self.mode=='exact' or self.mode=='ED':
				olap_str='overlap = {0:0.10f}.\n'.format(self.Eloc_params_dict_log['overlap'])
				if self.comm.Get_rank()==0:
					print(olap_str)
			if self.logfile is not None:
				self.logfile.flush()

			print(self.E_estimator.Eloc_real.shape)

			#exit()
		
			#### check energy variance, undo update and restart sampling back 10 iterations
			repeat, iteration = self.repeat_iteration(iteration,self.Eloc_mean_g,self.E_MC_std_g,go_back_iters=1,load_data=True)
			if repeat:
				continue

			##### save data
			self.save_all_data(iteration,start_iter)


			prss_time=time.time()-ti
			fin_iter_str="PROCESS_RANK {0:d}, iteration step {1:d} took {2:0.4f} secs.".format(self.comm.Get_rank(), iteration, prss_time)
			print(fin_iter_str)
			

			self.timing_vec[iteration-start_iter]=prss_time
			
			if self.logfile is not None:
				self.logfile.flush()
				os.fsync(self.logfile.fileno())


			# run debug helper
			if self.mode=='MC':
				self.run_debug_helper()
			

			iteration+=1
			self.comm.Barrier()


			if self.semi_exact[0] and self.semi_exact[1]:
				print('\n\n\nSemi-exact check complete!\nexiting simulation...\n')
				exit()


		iteration-=1


		### store runtime data

		prss_tot_time=time.time()-t_start
		final_str='\n\n\n\nPROCESS_RANK {0:d}, TOTAL calculation time: {1:0.4f} secs.\n\n'.format(self.comm.Get_rank(),prss_tot_time)
		print(final_str)
		self.timing_vec[iteration+1-start_iter]=prss_tot_time

		timing_matrix=np.zeros((self.comm.Get_size(),self.N_iterations+1),dtype=np.float64)
		self.comm.Allgather(self.timing_vec, timing_matrix)

		if self.comm.Get_rank()==0 and self.save_data:
			timing_matrix_filename = '/simulation_time--start_iter_{0:d}'.format(start_iter) + '.txt'
			np.savetxt(self.data_dir+timing_matrix_filename,timing_matrix.T,delimiter=',')
			
		
		# close files
		self.comm.Barrier()

		if self.logfile is not None:
			self.logfile.close()

		close_all_datafiles(self.all_data_files)

		self.comm.Barrier()


		# store data from last n_step iterations
		if self.mode=='MC':
			self.run_debug_helper(run=True,)

	
	def reestimate_local_energy_log(self, iteration, NN_params, batch, params_dict,):

	
		max_attemps=10

		if self.NN_dtype=='real':
			self.DNN_log.params=NN_params
		else:
			self.DNN.params=NN_params

		repeat=True
		counter=0
		while repeat and counter<max_attemps:

			##### get spin configs #####
			if self.mode=='exact' or self.mode=='ED':
				if self.NN_dtype=='real':
					self.MC_tool_log.exact(self.DNN_log, self.DNN_phase, )
				else:
					self.MC_tool_log.exact(self.DNN, None, )

			elif self.mode=='MC':
				ti=time.time()
				# sample
				if self.NN_dtype=='real':
					acceptance_ratio_g = self.MC_tool_log.sample(self.DNN_log, self.DNN_phase, )
				else:
					acceptance_ratio_g = self.MC_tool_log.sample(self.DNN, None, )

				MC_str="MC with acceptance ratio={0:.4f} took {1:.4f} secs.\n".format(acceptance_ratio_g[0],time.time()-ti)
				#self.logfile.write(MC_str)
				if self.comm.Get_rank()==0:
					print(MC_str)
				

			##### compute local energies #####
			if self.NN_dtype=='real':
				self.E_estimator_log.compute_local_energy(NN_params, self.DNN_phase.params, self.MC_tool_log.ints_ket,self.MC_tool_log.log_mod_kets,self.MC_tool_log.phase_kets,self.MC_tool_log.log_psi_shift, verbose=False,)
			else:
				self.E_estimator_log.compute_local_energy(NN_params, None, self.MC_tool_log.ints_ket,self.MC_tool_log.log_mod_kets,self.MC_tool_log.phase_kets,self.MC_tool_log.log_psi_shift, verbose=False,)
			

			if self.mode=='exact':
				mod_kets=np.exp(self.MC_tool_log.log_mod_kets)
				self.MC_tool_log.psi = mod_kets*np.exp(+1j*self.MC_tool_log.phase_kets)/np.linalg.norm(mod_kets[self.inv_index])
				abs_psi_2=self.MC_tool_log.count*np.abs(self.MC_tool_log.psi)**2
				
				params_dict=dict(abs_psi_2=abs_psi_2,)
				overlap=np.abs(self.MC_tool_log.psi[self.inv_index].conj().dot(self.E_estimator.psi_GS_exact))**2
				params_dict['overlap']=overlap

			elif self.mode=='ED':
				abs_psi_2=self.MC_tool_log.count*np.abs(self.MC_tool_log.psi)**2
				params_dict=dict(abs_psi_2=abs_psi_2,)

				# local overlap
				overlap_loc=np.sum(self.MC_tool_log.count * self.MC_tool_log.psi.conj()*self.E_estimator.psi_GS_exact) 
				
				# sum up MPI processes
				overlap=np.zeros(1, dtype=np.complex128)
				self.comm.Allreduce(overlap_loc, overlap,  op=MPI.SUM)
				
				params_dict['overlap']=np.abs(overlap[0])**2

			

			Eloc_mean_g, Eloc_var_g, E_diff_real, E_diff_imag = self.E_estimator_log.process_local_energies(params_dict)
			Eloc_std_g=np.sqrt(Eloc_var_g)
			E_MC_std_g=Eloc_std_g/np.sqrt(self.N_MC_points)

		
			# check quality of sample
			repeat, iteration = self.repeat_iteration(iteration,Eloc_mean_g,E_MC_std_g,go_back_iters=0, load_data=False)
			
			print("{0:d}. log-net sampling: Eloc = {1:14f}; repeat {2:d}".format(counter, Eloc_mean_g, repeat) )
			
			# increment counter
			counter+=1

			if repeat and counter==max_attemps:
				mssg="Failed to draw a good MC sample in {0:d} attempts. Exiting!".format(max_attemps)
				print(mssg)
				exit()
				#self.logfile.write(mssg)

			

		print("accepted log-net sample after {0:d} attempts.".format(counter))

		if self.NN_dtype=='real':
			params_dict['E_diff']=E_diff_real
		else:
			params_dict['E_diff']=E_diff_real+1j*E_diff_imag
		params_dict['Eloc_mean']=Eloc_mean_g
		params_dict['Eloc_var']=Eloc_var_g
		#params_dict['Eloc_mean_part']=Eloc_mean_g.real 
		
			
		##### total batch
		batch=self.MC_tool_log.spinstates_ket.reshape(self.input_shape)#[:self.N_batch]

		return params_dict, batch
	

	def reestimate_local_energy(self, iteration, NN_params, batch, params_dict,):

		max_attemps=10

		if self.NN_dtype=='real':
			self.DNN_log.params=NN_params # overwritten before iteration is over once gradients are updates
		else:
			self.DNN.params=NN_params

		repeat=True
		counter=0
		while repeat and counter<max_attemps:

			##### get spin configs #####
			if self.mode=='exact' or self.mode=='ED':
				if self.NN_dtype=='real':
					self.MC_tool.exact(self.DNN_log, self.DNN_phase, )
				else:
					self.MC_tool.exact(self.DNN, None, )

			elif self.mode=='MC':
				ti=time.time()
				# sample
				if self.NN_dtype=='real':
					acceptance_ratio_g = self.MC_tool.sample(self.DNN_log, self.DNN_phase, )
				else:
					acceptance_ratio_g = self.MC_tool.sample(self.DNN, None, )

				MC_str="MC with acceptance ratio={0:.4f} took {1:.4f} secs.\n".format(acceptance_ratio_g[0],time.time()-ti)
				if self.comm.Get_rank()==0:
					print(MC_str)
				

			##### compute local energies #####
			if self.NN_dtype=='real':
				self.E_estimator.compute_local_energy(NN_params, self.DNN_phase.params, self.MC_tool.ints_ket,self.MC_tool.log_mod_kets,self.MC_tool.phase_kets,self.MC_tool.log_psi_shift, verbose=False,)
			else:
				self.E_estimator.compute_local_energy(NN_params, None, self.MC_tool.ints_ket,self.MC_tool.log_mod_kets,self.MC_tool.phase_kets,self.MC_tool.log_psi_shift, verbose=False,)
			

			if self.mode=='exact':
				mod_kets=np.exp(self.MC_tool.log_mod_kets)
				self.MC_tool.psi = mod_kets*np.exp(+1j*self.MC_tool.phase_kets)/np.linalg.norm(mod_kets[self.inv_index])
				abs_psi_2=self.MC_tool.count*np.abs(self.MC_tool_log.psi)**2
				
				params_dict=dict(abs_psi_2=abs_psi_2,)
				overlap=np.abs(self.MC_tool.psi[self.inv_index].conj().dot(self.E_estimator.psi_GS_exact))**2
				params_dict['overlap']=overlap

			elif self.mode=='ED':
				abs_psi_2=self.MC_tool.count*np.abs(self.MC_tool.psi)**2
				params_dict=dict(abs_psi_2=abs_psi_2,)

				# local overlap
				overlap_loc=np.sum(self.MC_tool.count * self.MC_tool.psi.conj()*self.E_estimator.psi_GS_exact) 
				
				# sum up MPI processes
				overlap=np.zeros(1, dtype=np.complex128)
				self.comm.Allreduce(overlap_loc, overlap,  op=MPI.SUM)
				
				params_dict['overlap']=np.abs(overlap[0])**2

			

			Eloc_mean_g, Eloc_var_g, E_diff_real, E_diff_imag = self.E_estimator.process_local_energies(params_dict)
			Eloc_std_g=np.sqrt(Eloc_var_g)
			E_MC_std_g=Eloc_std_g/np.sqrt(self.N_MC_points)

		
			# check quality of sample
			repeat, iteration = self.repeat_iteration(iteration,Eloc_mean_g,E_MC_std_g,go_back_iters=0, load_data=False)
			
			print("{0:d}. log-net sampling: Eloc = {1:14f}; repeat {2:d}".format(counter, Eloc_mean_g, repeat) )
			
			# increment counter
			counter+=1

			if repeat and counter==max_attemps:
				mssg="Failed to draw a good MC sample in {0:d} attempts. Exiting!".format(max_attemps)
				print(mssg)
				exit()
				#self.logfile.write(mssg)

			

		print("accepted log-net sample after {0:d} attempts.".format(counter))

		if self.NN_dtype=='real':
			params_dict['E_diff']=E_diff_real
		else:
			params_dict['E_diff']=E_diff_real+1j*E_diff_imag
		params_dict['Eloc_mean']=Eloc_mean_g
		params_dict['Eloc_var']=Eloc_var_g
		#params_dict['Eloc_mean_part']=Eloc_mean_g.real 
		
			
		##### total batch
		batch=self.MC_tool.spinstates_ket.reshape(self.input_shape)#[:self.N_batch]

		return params_dict, batch
	

	def get_Hessian(self,iteration,):

		##### get spin configs #####
		ti=time.time()

		if self.NN_dtype=='real':
			self.MC_tool.exact(self.DNN_log, self.DNN_phase, self.logfile )
		else:
			self.MC_tool.exact(self.DNN, None, )

		E_str="network evaluation took {0:.4f} secs.\n".format(time.time()-ti)
		#self.logfile.write(MC_str)
		if self.comm.Get_rank()==0:
			print(E_str)
			
		

		# get log_psi statistics
		data_tuple=np.min(self.MC_tool.log_mod_kets), np.max(self.MC_tool.log_mod_kets), np.mean(self.MC_tool.log_mod_kets), np.std(self.MC_tool.log_mod_kets), np.max(self.MC_tool.log_mod_kets)-np.min(self.MC_tool.log_mod_kets)
		psi_str="log_|psi|_kets: min={0:0.8f}, max={1:0.8f}, mean={2:0.8f}; std={3:0.8f}, diff={4:0.8f}.".format(*data_tuple )
		#self.logfile.write(psi_str)
		print(psi_str)


		if self.logfile is not None:
			self.logfile.flush()
		print("LOCAL ENERGY:")



		##### total batch	
		self.batch=self.MC_tool.spinstates_ket.reshape(self.input_shape)#[:self.N_batch]
		self.MC_tool.spinstates_ket=None

		
		##### compute O
		self.opt_log.NG._compute_grads(self.DNN_log.params,self.batch,)
		self.opt_phase.NG._compute_grads(self.DNN_phase.params,self.batch,)

		self.batch=None
		self.opt_log.NG.dlog_psi_aux=None
		self.opt_log.NG.ddlog_psi_aux=None
		

		# self.MC_tool.dlog_psi_g = np.zeros((self.N_MC_points,self.DNN_log.N_varl_params),  dtype=self.opt_log.NG.dlog_psi.dtype  )
		# self.MC_tool.dphase_psi_g=np.zeros((self.N_MC_points,self.DNN_phase.N_varl_params),dtype=self.opt_phase.NG.dlog_psi.dtype)

		

		# sendcounts_log = np.zeros(self.comm.Get_size(),dtype=int)		
		# self.comm.Allgather([np.prod(self.opt_log.NG.dlog_psi.shape),MPI.INT],[sendcounts_log, MPI.INT])
		
		# sendcounts_phase = np.zeros(self.comm.Get_size(),dtype=int)		
		# self.comm.Allgather([np.prod(self.opt_phase.NG.dlog_psi.shape),MPI.INT],[sendcounts_phase, MPI.INT])
		
				
		# n_batch = self.comm.bcast(self.MC_tool.N_batch, root=0)
		# displacements_log=np.array([j*n_batch*self.DNN_log.N_varl_params for j in range(self.comm.Get_size())])
		# displacements_phase=np.array([j*n_batch*self.DNN_phase.N_varl_params for j in range(self.comm.Get_size())])

		
		# self.comm.Allgatherv([self.opt_log.NG.dlog_psi,	  MPI.DOUBLE], [self.MC_tool.dlog_psi_g,   sendcounts_log,   displacements_log,  MPI.DOUBLE], )
		# self.comm.Allgatherv([self.opt_phase.NG.dlog_psi, MPI.DOUBLE], [self.MC_tool.dphase_psi_g, sendcounts_phase, displacements_phase,  MPI.DOUBLE], )
		
		


		##### compute local energies #####
		N_params=self.DNN_log.N_varl_params+self.DNN_phase.N_varl_params
		HO=np.zeros( (self.N_batch,N_params) , dtype=np.complex128)


		ti=time.time()		
		Eloc, HO[:,:self.DNN_log.N_varl_params], HO[:,self.DNN_log.N_varl_params:] = self.E_estimator.ED_compute_local_energy_hessian(self.DNN_log.params, self.DNN_phase.params, self.MC_tool.ints_ket,self.MC_tool.log_mod_kets,self.MC_tool.phase_kets,self.MC_tool.log_psi_shift,self.opt_log.NG.dlog_psi, self.opt_phase.NG.dlog_psi)
		

		Eloc_str="\nlocal energy calculation took {0:.4f} secs.\n".format(time.time()-ti)
		#self.logfile.write(Eloc_str)
		if self.comm.Get_rank()==0:
			print(Eloc_str)


		##### compute measure
		abs_psi_2=self.MC_tool.count*np.abs(self.MC_tool.psi)**2
		Eloc_params_dict=dict(abs_psi_2=abs_psi_2,)
		
		# local overlap
		overlap_loc=np.sum(self.MC_tool.count * self.MC_tool.psi.conj()*self.E_estimator.psi_GS_exact) 
		# sum up MPI processes
		overlap=np.zeros(1, dtype=np.complex128)
		self.comm.Allreduce(overlap_loc, overlap,  op=MPI.SUM)
		overlap2=np.abs(overlap[0])**2
		overlap_loc=None


		olap_str="overlap^2 = {0:.4f} secs.\n".format(overlap2)
		if self.comm.Get_rank()==0:
			print(olap_str)
		if self.logfile is not None:
			self.logfile.flush()
			
		
	
		
		Eloc_mean_g, Eloc_var_g, E_diff_real, E_diff_imag = self.E_estimator.process_local_energies(Eloc_params_dict)
		Eloc_std_g=np.sqrt(Eloc_var_g)
		E_MC_std_g=Eloc_std_g/np.sqrt(self.N_MC_points)


		self.E_estimator=None
		self.E_estimator_log=None
		self.MC_tool=None
		self.MC_tool_log=None


		#########
		
		dlog_kets  =np.zeros( (self.N_batch,N_params) , dtype=np.float64)
		dlog_kets[:,:self.DNN_log.N_varl_params]=self.opt_log.NG.dlog_psi
		self.opt_log.NG.dlog_psi=None

		print("finished preallocation of dlog_kets\n")
		if self.logfile is not None:
			self.logfile.flush()

		dphase_kets=np.zeros( (self.N_batch,N_params) , dtype=np.float64)
		dphase_kets[:,self.DNN_log.N_varl_params:]=self.opt_phase.NG.dlog_psi
		self.opt_phase.NG.dlog_psi=None

		print("finished preallocation of dphase_kets\n")
		if self.logfile is not None:
			self.logfile.flush()


		print("finished preallocation of all variables\n")
		if self.logfile is not None:
			self.logfile.flush()



		##### compute Hessian

		print('start hessian computation\n')
		if self.logfile is not None:
			self.logfile.flush()

		self.H_shape=(N_params,N_params)
		Hessian=np.zeros(self.H_shape, dtype=np.float64)


		def _compute_dOElocdiff(abs_psi_2, dO, E_diff, shape,):

			# d_m O_n dEloc
			H = np.zeros((shape,shape),dtype=np.float64)
			self.comm.Allreduce( np.einsum('s,smn,s->mn', abs_psi_2, dO, E_diff) , H[...], op=MPI.SUM)
			return 2.0*(H+H.T)


		Hessian[:self.DNN_log.N_varl_params,:self.DNN_log.N_varl_params]+=_compute_dOElocdiff(abs_psi_2, self.opt_log.NG.ddlog_psi,   E_diff_real, self.DNN_log.N_varl_params   )
		Hessian[self.DNN_log.N_varl_params:,self.DNN_log.N_varl_params:]+=_compute_dOElocdiff(abs_psi_2, self.opt_phase.NG.ddlog_psi, E_diff_imag, self.DNN_phase.N_varl_params )



		self.opt_phase.NG=None
		self.opt_log.NG=None

		
		print('computed ddpsi\n')
		if self.logfile is not None:
			self.logfile.flush()


		def _compute_OHO(abs_psi_2, O, HO):
			H = np.zeros(self.H_shape,dtype=np.float64)
			self.comm.Allreduce( np.einsum('s,sn,sm->nm', abs_psi_2, O, HO) , H[...], op=MPI.SUM)
			return 2.0*H


		Hessian+=_compute_OHO(abs_psi_2, dlog_kets, HO.real)
		Hessian+=_compute_OHO(abs_psi_2, dphase_kets, HO.imag)
		HO=None

		print('computed HO\n')
		if self.logfile is not None:
			self.logfile.flush()

		def _compute_OOEdiff(abs_psi_2, O1, O2, E_diff, symmetrize=False):
			H = np.zeros(self.H_shape,dtype=np.float64)
			self.comm.Allreduce( np.einsum('s,sm,sn,s->mn', abs_psi_2, O1, O2, E_diff) , H[...], op=MPI.SUM)
			if symmetrize:
				return 2.0*(H+H.T)
			else:
				return 2.0*H

		# # compute auxiliary variables
		# def _couple_OO(O1, O2):
		# 	return np.einsum('sm,sn->smn', O1, O2)


		# Olog_Olog    =_couple_OO(dlog_kets, dlog_kets)
		# Ophase_Ophase=_couple_OO(dphase_kets, dphase_kets)
		# Olog_Ophase  =_couple_OO(dlog_kets, dphase_kets)

		# print('computed OO\n')
		# if self.logfile is not None:
		# 	self.logfile.flush()


		Hessian+=_compute_OOEdiff(abs_psi_2, dlog_kets, dlog_kets, E_diff_real)
		Hessian-=_compute_OOEdiff(abs_psi_2, dphase_kets, dphase_kets, E_diff_real)
		Hessian+=_compute_OOEdiff(abs_psi_2, dlog_kets, dphase_kets, E_diff_imag, symmetrize=True )


		print('computed ddpsi\n')
		if self.logfile is not None:
			self.logfile.flush()


		def _compute_OO(abs_psi_2, O1, O2,):
			H = np.zeros(self.H_shape,dtype=np.float64)
			self.comm.Allreduce( np.einsum('s,sm,sn->mn', abs_psi_2, O1, O2) , H[...], op=MPI.SUM)
			return 2.0*(H+H.T)

		Hessian+=_compute_OO(abs_psi_2, dlog_kets, dphase_kets)*Eloc_mean_g.real


		# compute last bit
		def _compute_O_OEdiff(abs_psi_2, O1, O2, E_diff, ):
			# d_m O_n dEloc
			O_expt = np.zeros(self.H_shape[0],dtype=np.float64)
			O_Ediff_Expt = np.zeros(self.H_shape[0],dtype=np.float64)

			self.comm.Allreduce( np.einsum('s,sm->m', abs_psi_2, O1) , O_expt[...], op=MPI.SUM)
			self.comm.Allreduce( np.einsum('s,sn,s->n', abs_psi_2, O2, E_diff) , O_Ediff_Expt[...], op=MPI.SUM)

			H=np.outer(O_expt,O_Ediff_Expt)

			return 4.0*(H+H.T)

		Hessian-=_compute_O_OEdiff(abs_psi_2, dlog_kets, dlog_kets, E_diff_real, )
		Hessian-=_compute_O_OEdiff(abs_psi_2, dlog_kets, dphase_kets, E_diff_imag, )



		#######


		check_herm=np.linalg.norm(Hessian - Hessian.T)
		print('hermiticity check value: {0:.18f}'.format(check_herm))

		E = np.linalg.eigvalsh(Hessian)



		file_hessian = self.savefile_dir + 'hessian_evalues_iter={0:d}'.format(iteration)

		
		if self.comm.Get_rank()==0:
			store_hessian_evalues(iteration,file_hessian, E)


			# with open(file_hessian+'.pkl', 'rb') as handle:
			# 	E2 = pickle.load(handle)

		print(E[-1])

		exit()


	



	
		


	def get_training_data(self,iteration,):

		##### get spin configs #####
		if self.mode=='exact' or self.mode=='ED':
			ti=time.time()

			if self.NN_dtype=='real':
				self.MC_tool.exact(self.DNN_log, self.DNN_phase, self.logfile )
			else:
				self.MC_tool.exact(self.DNN, None, )

			E_str="network evaluation took {0:.4f} secs.\n".format(time.time()-ti)
			#self.logfile.write(MC_str)
			if self.comm.Get_rank()==0:
				print(E_str)
			
		elif self.mode=='MC':
			ti=time.time()
			
			# sample
			if self.NN_dtype=='real':
				acceptance_ratio_g = self.MC_tool.sample(self.DNN_log, self.DNN_phase, )
			else:
				acceptance_ratio_g = self.MC_tool.sample(self.DNN, None, )
			
			MC_str="MC with acceptance ratio={0:.4f} took {1:.4f} secs.\n".format(acceptance_ratio_g[0],time.time()-ti)
			#self.logfile.write(MC_str)
			if self.comm.Get_rank()==0:
				print(MC_str)
			#exit()

			if iteration==0:
				self.MC_tool.thermal=self.thermal # set MC sampler to re-use initial state
		


		if self.logfile is not None:
			self.logfile.flush()
		print("LOCAL ENERGY:")

		# get log_psi statistics
		data_tuple=np.min(self.MC_tool.log_mod_kets), np.max(self.MC_tool.log_mod_kets), np.mean(self.MC_tool.log_mod_kets), np.std(self.MC_tool.log_mod_kets), np.max(self.MC_tool.log_mod_kets)-np.min(self.MC_tool.log_mod_kets)
		psi_str="log_|psi|_kets: min={0:0.8f}, max={1:0.8f}, mean={2:0.8f}; std={3:0.8f}, diff={4:0.8f}.".format(*data_tuple )
		#self.logfile.write(psi_str)
		print(psi_str)


		##### compute local energies #####
		ti=time.time()
		if self.NN_dtype=='real':
			self.E_estimator.compute_local_energy(self.DNN_log.params, self.DNN_phase.params, self.MC_tool.ints_ket,self.MC_tool.log_mod_kets,self.MC_tool.phase_kets,self.MC_tool.log_psi_shift,)
		else:
			self.E_estimator.compute_local_energy(self.DNN.params, None, self.MC_tool.ints_ket,self.MC_tool.log_mod_kets,self.MC_tool.phase_kets,self.MC_tool.log_psi_shift,)
		

		Eloc_str="local energy calculation took {0:.4f} secs.".format(time.time()-ti)
		#self.logfile.write(Eloc_str)
		if self.comm.Get_rank()==0:
			print(Eloc_str)


		if self.mode=='exact':
			#mod_kets=np.exp(self.MC_tool.log_mod_kets)
			#self.MC_tool.psi = mod_kets*np.exp(+1j*self.MC_tool.phase_kets)/np.linalg.norm(mod_kets[self.inv_index])
			abs_psi_2=self.MC_tool.count*np.abs(self.MC_tool.psi)**2
			
			Eloc_params_dict=dict(abs_psi_2=abs_psi_2,)
			overlap=np.abs(self.MC_tool.psi[self.inv_index].conj().dot(self.E_estimator.psi_GS_exact))**2
			Eloc_params_dict['overlap']=overlap
			#print(abs_psi_2)

		
		elif self.mode=='ED':
			abs_psi_2=self.MC_tool.count*np.abs(self.MC_tool.psi)**2
			Eloc_params_dict=dict(abs_psi_2=abs_psi_2,)

			
			#print(np.sum(self.E_estimator.count*self.MC_tool.psi.conj()*self.MC_tool.psi) )
			#print(np.sum(self.E_estimator.count*self.E_estimator.psi_GS_exact.conj()*self.E_estimator.psi_GS_exact) )
			
			# local overlap
			overlap_loc=np.sum(self.MC_tool.count * self.MC_tool.psi.conj()*self.E_estimator.psi_GS_exact) 
			
			# sum up MPI processes
			overlap=np.zeros(1, dtype=np.complex128)
			self.comm.Allreduce(overlap_loc, overlap,  op=MPI.SUM)

			Eloc_params_dict['overlap']=np.abs(overlap[0])**2

			
		
		elif self.mode=='MC':
			Eloc_params_dict=dict(N_MC_points=self.N_MC_points)

		
		self.Eloc_mean_g, self.Eloc_var_g, E_diff_real, E_diff_imag = self.E_estimator.process_local_energies(Eloc_params_dict)
		self.Eloc_std_g=np.sqrt(self.Eloc_var_g)
		self.E_MC_std_g=self.Eloc_std_g/np.sqrt(self.N_MC_points)
		

		Eloc_params_dict['Eloc_mean']=self.Eloc_mean_g
		Eloc_params_dict['Eloc_var']=self.Eloc_var_g

		if self.NN_dtype=='real':
			self.Eloc_params_dict_log=Eloc_params_dict.copy()
			self.Eloc_params_dict_phase=Eloc_params_dict.copy()

			self.Eloc_params_dict_log['E_diff']  =E_diff_real
			self.Eloc_params_dict_phase['E_diff']=E_diff_imag

			#self.Eloc_params_dict_log['Eloc_mean_part']  =self.Eloc_mean_g.real 
			#self.Eloc_params_dict_phase['Eloc_mean_part']=self.Eloc_mean_g.imag
		else:
			self.Eloc_params_dict_log=Eloc_params_dict.copy()
			self.Eloc_params_dict_log['E_diff']=E_diff_real+1j*E_diff_imag
		
		
		
		#####
		E_str=self.mode + ": E={0:0.14f}, E_var={1:0.14f}, E_std={2:0.14f}, E_imag={3:0.14f}.".format(self.Eloc_mean_g.real,self.Eloc_var_g, self.E_MC_std_g, self.Eloc_mean_g.imag, )
		if self.comm.Get_rank()==0:
			#E_str+="	with {0:d} unique spin configs.\n".format(np.unique(self.MC_tool.ints_ket_g[-1,...]).shape[0] )
			print(E_str)
		#self.logfile.write(E_str)


		##### total batch	
		self.batch=self.MC_tool.spinstates_ket.reshape(self.input_shape)#[:self.N_batch]

	
		#return self.batch, self.Eloc_params_dict
		

	def update_NN_params_real(self,iteration):

		ti=time.time()

		if self.grad_update_mode=='normal':
			# order is important !!! (energy_lib stores log-values)
			phase_params, phase_params_update, self.r2[1] = self.opt_phase.return_grad(iteration, self.DNN_phase.params, self.batch, self.Eloc_params_dict_phase, )
			log_params,   log_params_update  , self.r2[0] = self.opt_log.return_grad(iteration, self.DNN_log.params, self.batch, self.Eloc_params_dict_log, )
			
		elif self.grad_update_mode=='alternating':
			if (iteration//self.alt_iters)%2==1: # phase grads
				log_params_update*=0.0
				phase_params, phase_params_update, self.r2[1] = self.opt_phase.return_grad(iteration, self.DNN_phase.params, self.batch, self.Eloc_params_dict_phase, )

			else: # log grads
				log_params, log_params_update, self.r2[0] = self.opt_log.return_grad(iteration, self.DNN_log.params, self.batch, self.Eloc_params_dict_log, )
				phase_params_update=0.0

		elif self.grad_update_mode=='phase':
			log_params=self.DNN_log.params
			log_params_update=0.0
			phase_params, phase_params_update, self.r2[1] = self.opt_phase.return_grad(iteration, self.DNN_phase.params, self.batch, self.Eloc_params_dict_phase, )

		elif self.grad_update_mode=='log_mod':
			log_params, log_params_update, self.r2[0] = self.opt_log.return_grad(iteration, self.DNN_log.params, self.batch, self.Eloc_params_dict_log, )
			phase_params=self.DNN_phase.params
			phase_params_update=0.0


		# update params
		self.DNN_phase.params, self.DNN_phase.params_update[:] = phase_params, phase_params_update
		self.DNN_log.params,   self.DNN_log.params_update[:]   = log_params  , log_params_update

		#print("HERE",phase_params_update[-1])
		#exit()

		# print(np.linalg.norm(log_params_update), np.linalg.norm(phase_params_update))
		# print(np.linalg.norm(self.DNN_log.NN_Tree.ravel(log_params)), np.linalg.norm(self.DNN_log.NN_Tree.ravel(phase_params)) )
		# exit()

		#print(self.opt_phase.Runge_Kutta.step_size, self.opt_phase.step_size)

		##### compute max gradients
		grads_max=[np.max(np.abs(self.DNN_log.params_update)),np.max(np.abs(self.DNN_phase.params_update)),]
		
		
		mssg="r2 test: {0:0.14f}.".format(self.r2[0]+self.r2[1]-1.0)
		if self.comm.Get_rank()==0:
			print(mssg)
		
		#exit()

		# record gradients

		if self.comm.Get_rank()==0:
			self.params_log_update_lastiters[-1,...]=self.DNN_log.params_update
			self.params_phase_update_lastiters[-1,...]=self.DNN_phase.params_update


		grad_str="\ntotal gradients/NG calculation took {0:.4f} secs.".format(time.time()-ti)
		#self.logfile.write(grad_str)
		if self.comm.Get_rank()==0:
			print(grad_str)	
		
		return grads_max


	def update_NN_params_cpx(self,iteration):

		assert(self.grad_update_mode=='normal')

		ti=time.time()

		# compute gradients
		params,   params_update  , self.r2[0] = self.opt.return_grad(iteration, self.DNN.params, self.batch, self.Eloc_params_dict_log, )
			
		# update params
		self.DNN.params,   self.DNN.params_update[:]   = params  , params_update

		##### compute max gradients
		grads_max=[np.max(np.abs(self.DNN.params_update)),0.0,]
		
		
		mssg="r2 test: {0:0.14f}.".format(self.r2[0]-1.0)
		if self.comm.Get_rank()==0:
			print(mssg)
		
		# record gradients

		if self.comm.Get_rank()==0:
			self.params_update_lastiters[-1,...]=self.DNN.params_update
			

		grad_str="\ntotal gradients/NG calculation took {0:.4f} secs.".format(time.time()-ti)
		#self.logfile.write(grad_str)
		if self.comm.Get_rank()==0:
			print(grad_str)	


		#exit()
		
		return grads_max

