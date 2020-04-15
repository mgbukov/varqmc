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
#from scipy.linalg import sqrtm, pinvh

import yaml
import pickle

from cpp_code import Log_Net, Phase_Net
from cpp_code import integer_to_spinstate
from cpp_code import scale_cpx

from MC_lib import MC_sampler
from energy_lib import Energy_estimator
from optimizer import optimizer

from copy import copy
import datetime
import time
np.set_printoptions(threshold=np.inf)


#from misc.MC_weights import *


def truncate_file(file_name, start_iter):

	with open(file_name) as file:	
		lines=file.readlines()
		keep_lines=[]
		for i,line in enumerate(lines):
			if i < start_iter:
				keep_lines.append(line)

	with open(file_name, 'w') as file:
		for line in keep_lines:
			file.write(line)


def read_str(tuple_str):

	shape_tuple=()

	tuple_str=tuple_str.replace('(','')
	tuple_str=tuple_str.replace(')','')
	tuple_str=tuple_str.split(',')


	for NN_str in tuple_str:
		shape_tuple+=(NN_str,)

	return shape_tuple, len(tuple_str)


def load_opt_data(opt,file_name,start_iter):

	with open(file_name) as file:
		for i in range(start_iter):
			opt_data_str = file.readline().rstrip().split(' : ')	

	opt.iteration=int(opt_data_str[0])+1
	opt.NG.delta=np.float64(opt_data_str[1])
	opt.NG.tol=np.float64(opt_data_str[2])

	opt.time=np.float64(opt_data_str[5])

	if opt.opt=='RK':
		opt.Runge_Kutta.counter=int(opt_data_str[3])
		opt.Runge_Kutta.step_size=np.float64(opt_data_str[4])
		opt.Runge_Kutta.time=np.float64(opt_data_str[5])
	else:
		opt.step_size=np.float64(opt_data_str[4])


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
		
		self.opt=read_str(params_dict['opt'])[0]

		step_size_str=read_str(params_dict['step_size'])[0]
		self.step_sizes=[np.float(step_size_str[0]), np.float(step_size_str[1])]

		self.cost=read_str(params_dict['cost'])[0]
		self.TDVP_opt=read_str(params_dict['TDVP_opt'])[0]

		self.grad_update_mode=params_dict['grad_update_mode']
		self.alt_iters=params_dict['alt_iter'] # only effective in real-decoupled mode
		

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
		self.MC_prop_threshold=params_dict['MC_prop_threshold']
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


		# create log file and directory
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



	def _load_data(self, start_iter, truncate_files=True):

		### load MC data
		
		with open(self.file_MC_data.name) as file:
			for i in range(start_iter):
					MC_data_str = file.readline().rstrip().split(' : ')				

		it_MC, acceptance_ratio_g, acceptance_ratios, s0_g, sf_g =  MC_data_str

		self.MC_tool.acceptance_ratio_g[0]=np.float64(acceptance_ratio_g)
		self.MC_tool.s0_g=np.array([self.E_estimator.basis_type(s0) for s0 in s0_g.split(' ')] )
		self.MC_tool.sf_g=np.array([self.E_estimator.basis_type(sf) for sf in sf_g.split(' ')] )

		m_l=self.N_MC_chains*self.comm.Get_rank()
		m_r=self.N_MC_chains*(self.comm.Get_rank()+1)

		self.DNN._init_MC_data(s0_vec=self.MC_tool.s0_g[m_l:m_r], sf_vec=self.MC_tool.sf_g[m_l:m_r], )



		### load DNN params
		file_name='/NN_params/NNparams'+'--iter_{0:05d}'.format(start_iter)

		with open(self.data_dir+file_name+'.pkl', 'rb') as handle:
			self.DNN.params_log, self.DNN.params_phase, \
			self.DNN.apply_fun_args_log, self.DNN.apply_fun_args_phase, \
			self.MC_tool.log_psi_shift = pickle.load(handle)
			
		self.opt_log.init_opt_state(self.DNN.params_log)
		self.opt_phase.init_opt_state(self.DNN.params_phase)


		file_name='/NN_params/NNparams'+'--iter_{0:05d}'.format(start_iter-1) 
		with open(self.data_dir+file_name+'.pkl', 'rb') as handle:
			DNN_params_log_old, DNN_params_phase_old, _ ,_ , _ = pickle.load(handle)

		
		if self.opt_log.cost=='SR':
			self.opt_log.NG.nat_grad_guess[:]  = (self.DNN.NN_Tree_log.ravel(DNN_params_log_old)-self.DNN.NN_Tree_log.ravel(self.DNN.params_log) )/self.opt_log.step_size

		if self.opt_phase.cost=='SR':
			self.opt_phase.NG.nat_grad_guess[:]= (self.DNN.NN_Tree_phase.ravel(DNN_params_phase_old)-self.DNN.NN_Tree_phase.ravel(self.DNN.params_phase) )/self.opt_phase.step_size




		### load opt data
		load_opt_data(self.opt_log, self.file_opt_data_log.name, start_iter)
		load_opt_data(self.opt_phase, self.file_opt_data_phase.name, start_iter)
		


		# truncate remaining files
		self.comm.Barrier()
		if truncate_files and self.comm.Get_rank()==0:
			truncate_file(self.file_MC_data.name, start_iter)
			truncate_file(self.file_opt_data_log.name, start_iter)
			truncate_file(self.file_opt_data_phase.name, start_iter)
			# clean rest of files
			truncate_file(self.file_energy.name, start_iter)
			truncate_file(self.file_loss_log.name, start_iter)
			truncate_file(self.file_loss_phase.name, start_iter)
			truncate_file(self.file_phase_hist.name, start_iter)
		self.comm.Barrier()
		#####
		assert(int(it_MC)+1==self.opt_log.iteration)
		assert(int(it_MC)+1==self.opt_phase.iteration)




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
		self.DNN_log=Log_Net(self.comm, self.shapes[0], self.N_MC_chains, self.NN_type, self.NN_dtype, seed=self.seed, prop_threshold=self.MC_prop_threshold )
		self.DNN_phase=Phase_Net(self.comm, self.shapes[1], self.N_MC_chains, self.NN_type, self.NN_dtype, seed=self.seed, prop_threshold=self.MC_prop_threshold )
	
		#print(self.DNN_log.N_varl_params)
		#print(self.DNN_phase.N_varl_params)
		#exit()

		self.N_symm = np.max([self.DNN_log.N_symm,self.DNN_phase.N_symm])


	def _create_optimizer(self):

		# log net
		self.opt_log   = optimizer(self.comm, self.opt[0], self.cost[0], self.mode, self.DNN_log.NN_Tree, label='log',  step_size=self.step_sizes[0], )
		self.opt_log.init_global_variables(self.N_MC_points, self.N_batch, self.DNN_log.N_varl_params, self.n_iter)
		self.opt_log.define_grad_func(self.DNN_log.evaluate, TDVP_opt=self.TDVP_opt[0], reestimate_local_energy=self.reestimate_local_energy_log )
		self.opt_log.init_opt_state(self.DNN_log.params)
		
		# phase net
		self.opt_phase = optimizer(self.comm, self.opt[1], self.cost[1], self.mode, self.DNN_phase.NN_Tree, label='phase', step_size=self.step_sizes[1], )
		self.opt_phase.init_global_variables(self.N_MC_points, self.N_batch, self.DNN_phase.N_varl_params, self.n_iter)
		self.opt_phase.define_grad_func(self.DNN_phase.evaluate, TDVP_opt=self.TDVP_opt[1], reestimate_local_energy=self.E_estimator.reestimate_local_energy_phase )
		self.opt_phase.init_opt_state(self.DNN_phase.params)
		


		# define variable to keep track of the DNN params update
		if self.comm.Get_rank()==0:
			self.params_log_update_lastiters=np.zeros((self.n_iter,self.DNN_log.N_varl_params),dtype=np.float64)
			self.params_phase_update_lastiters=np.zeros((self.n_iter,self.DNN_phase.N_varl_params),dtype=np.float64)
		else:
			self.params_log_update_lastiters=np.array([[None],[None]])
			self.params_phase_update_lastiters=np.array([[None],[None]])


		self.r2=np.zeros(2)

	

	def _create_energy_estimator(self):
		### Energy estimator
		self.E_estimator=Energy_estimator(self.comm,self.DNN_log,self.DNN_phase,self.mode,self.J2,self.N_MC_points,self.N_batch,self.L,self.N_symm,self.sign, self.minibatch_size) # contains all of the physics
		self.E_estimator.init_global_params(self.N_MC_points,self.n_iter)

		self.E_estimator_log=Energy_estimator(self.comm,self.DNN_log,self.DNN_phase,self.mode,self.J2,self.N_MC_points,self.N_batch,self.L,self.N_symm,self.sign, self.minibatch_size) # contains all of the physics
		self.E_estimator_log.init_global_params(self.N_MC_points,self.n_iter)

		
	def _create_MC_sampler(self, ):
		### initialize MC sampler variables
		self.MC_tool=MC_sampler(self.comm,self.N_MC_chains)
		self.MC_tool.init_global_vars(self.L,self.N_MC_points,self.N_batch,self.N_symm,self.NN_type,self.E_estimator.basis_type,self.E_estimator.MPI_basis_dtype,self.n_iter)
		
		if self.NN_type=='DNN':
			self.input_shape=(-1,self.N_symm,self.L**2)
		elif self.NN_type=='CNN':
			self.input_shape=(-1,self.N_symm,1,self.L,self.L)

		self.MC_tool_log=MC_sampler(self.comm,self.N_MC_chains)
		self.MC_tool_log.init_global_vars(self.L,self.N_MC_points,self.N_batch,self.N_symm,self.NN_type,self.E_estimator_log.basis_type,self.E_estimator_log.MPI_basis_dtype,self.n_iter)
		
		
		


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
		#logfile_name= 'LOGFILE--MPIprss_{0:d}--'.format(self.comm.Get_rank()) + self.file_name + '.txt'
		logfile_name= 'LOGFILE--MPIprss_{0:d}'.format(self.comm.Get_rank()) + '.txt'
		self.logfile = create_open_file(logfile_dir+logfile_name)
		self.E_estimator.logfile=self.logfile
		self.E_estimator.logfile=self.logfile
		
		# redircet warnings to log
		def customwarn(message, category, filename, lineno, file=None, line=None):
			self.logfile.write('\n'+warnings.formatwarning(message, category, filename, lineno)+'\n')
		warnings.showwarning = customwarn

		
		self.debug_file_SF_log       =self.savefile_dir_debug + 'debug-SF_data_log'            #+'--' + self.file_name
		self.debug_file_SF_phase     =self.savefile_dir_debug + 'debug-SF_data_phase'
		self.debug_file_logpsi       =self.savefile_dir_debug + 'debug-logpsi_data'        #+'--' + self.file_name
		self.debug_file_phasepsi     =self.savefile_dir_debug + 'debug-phasepsi_data'      #+'--' + self.file_name
		self.debug_file_intkets      =self.savefile_dir_debug + 'debug-intkets_data'       #+'--' + self.file_name
		self.debug_file_Eloc         =self.savefile_dir_debug + 'debug-Eloc_data'          #+'--' + self.file_name
		self.debug_file_params_update=self.savefile_dir_debug + 'debug-params_update_data' #+'--' + self.file_name
		

		if self.save_data:
			# data files
			#common_str = '--'  self.file_name + '.txt'
			common_str = '.txt'

			self.file_energy= create_open_file(self.savefile_dir+'energy'+common_str)
			#self.file_energy_std= create_open_file(self.savefile_dir+'energy_std--'+common_str)
			self.file_loss_log= create_open_file(self.savefile_dir+'loss_log'+common_str)
			self.file_loss_phase= create_open_file(self.savefile_dir+'loss_phase'+common_str)
			#self.file_r2= create_open_file(self.savefile_dir+'r2--'+common_str)
			self.file_phase_hist=create_open_file(self.savefile_dir+'phases_histogram'+common_str)

			self.file_MC_data= create_open_file(self.savefile_dir+'MC_data'+common_str)
			self.file_opt_data_log= create_open_file(self.savefile_dir+'opt_data_log'+common_str)
			self.file_opt_data_phase= create_open_file(self.savefile_dir+'opt_data_phase'+common_str)


		### timing vector
		self.timing_vec=np.zeros((self.N_iterations+1,),dtype=np.float64)
		self.opt_log.logfile=self.logfile
		self.opt_phase.logfile=self.logfile
		

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
		#file_name='NNparams'+'--iter_{0:05d}--'.format(iteration) + self.file_name
		file_name='NNparams'+'--iter_{0:05d}'.format(iteration) 
		with open(self.savefile_dir_NN+file_name+'.pkl', 'wb') as handle:
			pickle.dump([self.DNN_log.params, self.DNN_phase.params, 
						 self.DNN_log.apply_fun_args, self.DNN_phase.apply_fun_args,
						 self.MC_tool.log_psi_shift,
						 ], handle, protocol=pickle.HIGHEST_PROTOCOL)

	def save_sim_data(self, iteration, grads_max, r2, phase_hist):

		# data
		self.file_energy.write("{0:d} : {1:0.14f} : {2:0.14f} : {3:0.14f} : {4:0.14f}\n".format(iteration, self.Eloc_mean_g.real , self.Eloc_mean_g.imag, self.Eloc_std_g, self.E_MC_std_g))
		#self.file_energy_std.write("{0:d} : {1:0.14f}\n".format(iteration, self.E_MC_std_g))
		

		######################################################


		data_tuple=(iteration, r2[0], grads_max[0], )
		if self.opt_log.cost=='SR':
			data_tuple+= (self.opt_log.NG.dE, self.opt_log.NG.curvature, self.opt_log.NG.F_norm, self.opt_log.NG.S_norm, self.opt_log.NG.S_logcond, )
		else:
			data_tuple+= (0.0, 0.0, 0.0, 0.0, 0.0)
		self.file_loss_log.write("{0:d} : {1:0.14f} : {2:0.14f} : {3:0.14f} : {4:0.14f} : {5:0.14f} : {6:0.10f} : {7:0.10f}\n".format(*data_tuple))
		

		data_tuple=(iteration, r2[1], grads_max[1], )
		if self.opt_phase.cost=='SR':
			data_tuple+= (self.opt_phase.NG.dE, self.opt_phase.NG.curvature, self.opt_phase.NG.F_norm, self.opt_phase.NG.S_norm, self.opt_phase.NG.S_logcond, )
		else:
			data_tuple+= (0.0, 0.0, 0.0, 0.0, 0.0)
		self.file_loss_phase.write("{0:d} : {1:0.14f} : {2:0.14f} : {3:0.14f} : {4:0.14f} : {5:0.14f} : {6:0.10f} : {7:0.10f}\n".format(*data_tuple))

		
		######################################################


		MC_data_1="{0:d} : {1:0.4f} : ".format(iteration, self.MC_tool.acceptance_ratio_g[0])
		MC_data_2=' '.join('{0:0.4f}'.format(r) for r in self.MC_tool.acceptance_ratio)+" : "
		MC_data_3=' '.join(str(s) for s in self.MC_tool.s0_g)+" : "
		MC_data_4=' '.join(str(s) for s in self.MC_tool.sf_g)
		self.file_MC_data.write(MC_data_1  +  MC_data_2  +  MC_data_3 +  MC_data_4 + "\n") #		
		

		######################################################


		if self.opt_log.cost=='SR':
			data_cost=(self.opt_log.NG.delta, self.opt_log.NG.tol,) 
		else:
			data_cost=(0.0,0.0,)

		if self.opt_log.opt=='RK':
			data_opt=(self.opt_log.Runge_Kutta.counter, self.opt_log.Runge_Kutta.step_size, self.opt_log.Runge_Kutta.time, )
		else:
			data_opt=(self.opt_log.iteration,self.opt_log.step_size,self.opt_log.time,)

		data_tuple=(iteration,)+data_cost+data_opt
		self.file_opt_data_log.write("{0:d} : {1:0.14f} : {2:0.14f} : {3:d} : {4:0.14f} : {5:0.14f}\n".format(*data_tuple))


		if self.opt_phase.cost=='SR':
			data_cost=(self.opt_phase.NG.delta, self.opt_phase.NG.tol,) 
		else:
			data_cost=(0.0,0.0,)

		if self.opt_phase.opt=='RK':
			data_opt=(self.opt_phase.Runge_Kutta.counter, self.opt_phase.Runge_Kutta.step_size, self.opt_phase.Runge_Kutta.time, )
		else:
			data_opt=(self.opt_phase.iteration,self.opt_phase.step_size,self.opt_phase.time,)

		data_tuple=(iteration,)+data_cost+data_opt
		self.file_opt_data_phase.write("{0:d} : {1:0.14f} : {2:0.14f} : {3:d} : {4:0.14f} : {5:0.14f}\n".format(*data_tuple))


		######################################################

		
		self.file_phase_hist.write("{0:d} : ".format(iteration) + ''.join("{0:0.6f}, ".format(value) for value in phase_hist) + '\n' )


		######################################################

		# record current iteration number
		self.params_dict['stop_iter']=iteration+1
		
		# update file in data dir
		config_params_yaml = open(self.data_dir + '/config_params.yaml', 'w')
		yaml.dump(self.params_dict, config_params_yaml)
		config_params_yaml.close()


		######################################################


		# flush data files
		self.file_energy.flush()
		self.file_loss_log.flush()
		self.file_loss_phase.flush()
		self.file_MC_data.flush()
		self.file_opt_data_log.flush()
		self.file_opt_data_phase.flush()
		self.file_phase_hist.flush()



	def debug_helper(self,):

		# record DNN params update
		if self.comm.Get_rank()==0:
			self.params_log_update_lastiters[:-1,...]=self.params_log_update_lastiters[1:,...]
			self.params_log_update_lastiters[-1,...]*=0.0

			self.params_phase_update_lastiters[:-1,...]=self.params_phase_update_lastiters[1:,...]
			self.params_phase_update_lastiters[-1,...]*=0.0



	def run_debug_helper(self, run=False,):

		# set default flag to False
		exit_flag=False 

		#
		##### store data
		# 
		if self.comm.Get_rank()==0:
	
			# check for nans and infs
			if run or (not self.opt_log.is_finite ) or (not self.opt_phase.is_finite ) or (not np.isfinite(self.Eloc_mean_g).all() ):
				
				if self.opt_log.cost=='SR':
					with open(self.debug_file_SF_log+'.pkl', 'wb') as handle:

						pickle.dump([self.opt_log.NG.S_lastiters,   self.opt_log.NG.F_lastiters,   self.opt_log.NG.delta, ], 
										handle, protocol=pickle.HIGHEST_PROTOCOL
									)

				if self.opt_phase.cost=='SR':
					with open(self.debug_file_SF_phase+'.pkl', 'wb') as handle:

						pickle.dump([self.opt_phase.NG.S_lastiters, self.opt_phase.NG.F_lastiters, self.opt_phase.NG.delta,], 
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

					pickle.dump([self.params_log_update_lastiters, self.params_phase_update_lastiters,], 
									handle, protocol=pickle.HIGHEST_PROTOCOL
								)

				# set exit variable and bcast it to all processes
				if not run: 
					exit_flag=True
		
		exit_flag = self.comm.bcast(exit_flag, root=0)
				
		

		self.comm.Barrier()



		if exit_flag:
			exit_str="\n\nEncountered nans or infs!\nExiting simulation...\n\n"
			print(exit_str)
			self.logfile.write(exit_str)
			exit()




	def save_all_data(self,iteration,start_iter):

		if iteration<self.N_iterations+start_iter:

			#### check point DNN parameters
			if self.comm.Get_rank()==0 and self.save_data:
				self.check_point(iteration)

			#### update DNN parameters
			grads_max = self.update_NN_params(iteration)

			##### store simulation data
			if self.mode=='exact':		
				phase_hist = self._compute_phase_hist(self.MC_tool.phase_kets,self.Eloc_params_dict_log['abs_psi_2'])
			else:
				mod_psi_2=np.exp(2.0*(self.MC_tool.log_mod_kets-self.MC_tool.log_psi_shift))
				phase_hist = self._compute_phase_hist(self.MC_tool.phase_kets,mod_psi_2)

			# print(phase_hist)
			# exit()

			if self.comm.Get_rank()==0 and self.save_data:
				#if not(self.load_data and (start_iter==iteration)):
				self.save_sim_data(iteration,grads_max,self.r2,phase_hist)


	def repeat_iteration(self,iteration,Eloc_mean_g,E_MC_std_g, go_back_iters=0, load_data=True):

		repeat=False
		if iteration>self.start_iter+go_back_iters and self.mode=='MC':

			Eloc_mean_prev=self.prev_it_data[0]
			
			_c1=Eloc_mean_prev-Eloc_mean_g.real
			_c2=np.abs(Eloc_mean_g.imag)
			_c3=6.0*self.prev_it_data[2] - E_MC_std_g 

			_b1=(np.abs(_c1) > 2.0) and (Eloc_mean_g<0.0)
			_b2=_c2 > 3.0*E_MC_std_g
			_b3=_c3 < 0.0

			
			if (_b1 or _b2 or _b3): # and Eloc_mean_prev < 0.0: 

				data_tuple=(iteration, Eloc_mean_g.real, Eloc_mean_g.imag, E_MC_std_g,)

				if _b1:
					mssg="!!!  restarting iteration {0:d}: E={1:0.6f}, E_imag={2:0.10f}, E_std={3:0.10f}, E_mean_check={4:0.10f}  !!!\n".format( *data_tuple, _c1, )
				elif _b2:
					mssg="!!!  restarting iteration {0:d}: E={1:0.6f}, E_imag={2:0.10f}, E_std={3:0.10f}, E_imag_check={4:0.10f}  !!!\n".format( *data_tuple, _c2, )
				elif _b3:
					mssg="!!!  restarting iteration {0:d}: E={1:0.6f}, E_imag={2:0.10f}, E_std={3:0.10f}, E_std_check={4:0.10f}  !!!\n".format(*data_tuple, _c3, )


				if self.comm.Get_rank()==0:
					print(mssg)
				self.logfile.write(mssg)
				
				# load data
				if load_data:
					self.comm.Barrier()
					self._load_data(iteration-1-go_back_iters, truncate_files=False)
					iteration=iteration-go_back_iters

				repeat=True
		
		return repeat, iteration


	def train(self, start_iter=0):

		# set timer
		t_start=time.time()


		if self.mode=='exact':
			assert(self.N_MC_points==107) # 107 states in the symmetry reduced sector for L=4

			self.MC_tool.ints_ket, self.index, self.inv_index, self.count=self.E_estimator.get_exact_kets()
			integer_to_spinstate(self.MC_tool.ints_ket, self.MC_tool.spinstates_ket, self.DNN_log.N_features, NN_type=self.DNN_log.NN_type)

			self.MC_tool_log.ints_ket, self.index, self.inv_index, self.count=self.E_estimator_log.get_exact_kets()
			integer_to_spinstate(self.MC_tool_log.ints_ket, self.MC_tool_log.spinstates_ket, self.DNN_log.N_features, NN_type=self.DNN_log.NN_type)


		# auxiliary variable
		self.prev_it_data=np.zeros(5) # Eloc_real, Eloc_imag, Eloc_std, S_norm, F_norm

		iteration=start_iter
		while iteration < start_iter+self.N_iterations:
		#for iteration in range(start_iter,start_iter+self.N_iterations, 1): 


			#self.comm.Barrier()
			ti=time.time()

			# shift params_update
			self.debug_helper()

			init_iter_str="\n\nITERATION {0:d}, PROCESS_RANK {1:d}:\n\n".format(iteration, self.comm.Get_rank())
			if self.comm.Get_rank()==0:
				print(init_iter_str)
			self.logfile.write(init_iter_str)


			##### determine batchnorm mean and variance
			#if self.batchnorm==1:
			#	self.compute_batchnorm_params(self.DNN.params,len(self.shapes)+1) #



			##### evaluate model
			self.get_training_data(iteration,)


			#####
			E_str=self.mode + ": E={0:0.14f}, E_std={1:0.14f}, E_imag={2:0.14f}.\n".format(self.Eloc_mean_g.real, self.E_MC_std_g, self.Eloc_mean_g.imag, )
			if self.comm.Get_rank()==0:
				#E_str+="	with {0:d} unique spin configs.\n".format(np.unique(self.MC_tool.ints_ket_g[-1,...]).shape[0] )
				print(E_str)
			self.logfile.write(E_str)


			if self.mode=='exact':
				olap_str='overlap = {0:0.10f}.\n\n'.format(self.Eloc_params_dict_log['overlap'])
				if self.comm.Get_rank()==0:
					print(olap_str)
				self.logfile.write(olap_str)


			#exit()


			#### check energy variance, undo update and restart sampling back 10 iterations
			repeat, iteration = self.repeat_iteration(iteration,self.Eloc_mean_g,self.E_MC_std_g,go_back_iters=1)
			if repeat:
				continue

			##### save data
			self.save_all_data(iteration,start_iter)


			prss_time=time.time()-ti
			fin_iter_str="PROCESS_RANK {0:d}, iteration step {1:d} took {2:0.4f} secs.\n".format(self.comm.Get_rank(), iteration, prss_time)
			self.logfile.write(fin_iter_str)
			print(fin_iter_str)
			

			self.timing_vec[iteration-start_iter]=prss_time
			
			self.logfile.flush()
			os.fsync(self.logfile.fileno())


			# synch
			self.prev_it_data[0], self.prev_it_data[1], self.prev_it_data[2]=self.Eloc_mean_g.real, self.Eloc_mean_g.imag, self.E_MC_std_g
			

			# run debug helper
			self.run_debug_helper()
			

			iteration+=1
			self.comm.Barrier()

			#exit()

		iteration-=1

		
		prss_tot_time=time.time()-t_start
		final_str='\n\nPROCESS_RANK {0:d}, total calculation time: {1:0.4f} secs.\n\n\n'.format(self.comm.Get_rank(),prss_tot_time)
		print(final_str)
		self.logfile.write(final_str)
		self.timing_vec[iteration+1-start_iter]=prss_tot_time


		timing_matrix=np.zeros((self.comm.Get_size(),self.N_iterations+1),dtype=np.float64)
		self.comm.Allgather(self.timing_vec, timing_matrix)


		if self.comm.Get_rank()==0 and self.save_data:
			timing_matrix_filename = '/simulation_time--start_iter_{0:d}'.format(start_iter) + '.txt'
			np.savetxt(self.data_dir+timing_matrix_filename,timing_matrix.T,delimiter=',')
			
		
		# close files
		self.logfile.close()
		self.file_energy.close()
		#self.file_energy_std.close()
		self.file_loss_log.close()
		self.file_loss_phase.close()
		self.file_phase_hist.close()


		# store data from last 6 iterations
		self.run_debug_helper(run=True,)


	'''

	def _update_batchnorm_params(self,layers, set_fixpoint_iter=True,):
		layers_type=list(layers.keys())
		for j, layer_type in enumerate(layers_type):
			if 'batch_norm' in layer_type:
				self.DNN.apply_fun_args_dyn[j]['fixpoint_iter']=set_fixpoint_iter
			
	

	def compute_batchnorm_params(self,NN_params,N_iter):
	
		ti=time.time()

		#print(self.DNN.apply_fun_args_dyn[2]['mean'])
		for i in range(N_iter):
			
			# draw MC sample
			acceptance_ratio_g = self.MC_tool.sample(self.DNN, compute_phases=False)
			
			self._update_batchnorm_params(self.DNN.NN_architecture, set_fixpoint_iter=True)
			log_psi, phase_psi = self.evaluate_NN_dyn(self.DNN.params, self.MC_tool.spinstates_ket.reshape(self.MC_tool.N_batch,self.MC_tool.N_symm,self.MC_tool.N_sites), )
			self._update_batchnorm_params(self.DNN.NN_architecture, set_fixpoint_iter=False)

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

	'''
	
	def reestimate_local_energy_log(self, iteration, NN_params_log, batch, params_dict,):


		repeat=True
		counter=0
		while repeat and counter<10:

			##### get spin configs #####
			if self.mode=='exact':
				self.MC_tool_log.exact(self.DNN_log, self.DNN_phase, )

			elif self.mode=='MC':
				# sample
				acceptance_ratio_g = self.MC_tool_log.sample(self.DNN_log, self.DNN_phase, )
				

			##### compute local energies #####
			self.E_estimator_log.compute_local_energy(NN_params_log, self.DNN_phase.params, self.MC_tool_log.ints_ket,self.MC_tool_log.log_mod_kets,self.MC_tool_log.phase_kets,self.MC_tool_log.log_psi_shift, verbose=False,)

			if self.mode=='exact':
				mod_kets=np.exp(self.MC_tool_log.log_mod_kets)
				self.psi = mod_kets*np.exp(+1j*self.MC_tool.phase_kets)/np.linalg.norm(mod_kets[self.inv_index])
				abs_psi_2=self.count*np.abs(self.psi)**2

				params_dict['abs_psi_2']=abs_psi_2
				overlap=np.abs(self.psi[self.inv_index].conj().dot(self.E_estimator_log.psi_GS_exact))**2
				params_dict['overlap']=overlap

			
			Eloc_mean_g, Eloc_var_g, E_diff_real, E_diff_imag = self.E_estimator_log.process_local_energies(params_dict)
			Eloc_std_g=np.sqrt(Eloc_var_g)
			E_MC_std_g=Eloc_std_g/np.sqrt(self.N_MC_points)


			# check quality of sample
			repeat, iteration = self.repeat_iteration(iteration,Eloc_mean_g,E_MC_std_g,go_back_iters=0, load_data=False)
			
			if repeat and counter>=10:
				mssg="Failed to draw a good MC sample in 10 attempts. Exiting!\n"
				if self.comm.Get_rank()==0:
					print(mssg)
				self.logfile.write(mssg)

			counter+=1


		params_dict['E_diff']=E_diff_real
		params_dict['Eloc_mean']=Eloc_mean_g
		params_dict['Eloc_var']=Eloc_var_g

			
		##### total batch
		batch=self.MC_tool_log.spinstates_ket.reshape(self.input_shape)

		return params_dict, batch
	


	def get_training_data(self,iteration,):

		##### get spin configs #####
		if self.mode=='exact':
			self.MC_tool.exact(self.DNN_log, self.DNN_phase, )
			
		elif self.mode=='MC':
			ti=time.time()
			
			# sample
			acceptance_ratio_g = self.MC_tool.sample(self.DNN_log, self.DNN_phase, )
			
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
		self.E_estimator.compute_local_energy(self.DNN_log.params, self.DNN_phase.params, self.MC_tool.ints_ket,self.MC_tool.log_mod_kets,self.MC_tool.phase_kets,self.MC_tool.log_psi_shift,)
		
		Eloc_str="total local energy calculation took {0:.4f} secs.\n".format(time.time()-ti)
		self.logfile.write(Eloc_str)
		if self.comm.Get_rank()==0:
			print(Eloc_str)


		if self.mode=='exact':
			mod_kets=np.exp(self.MC_tool.log_mod_kets)
			self.psi = mod_kets*np.exp(+1j*self.MC_tool.phase_kets)/np.linalg.norm(mod_kets[self.inv_index])
			abs_psi_2=self.count*np.abs(self.psi)**2

			Eloc_params_dict=dict(abs_psi_2=abs_psi_2,)
			overlap=np.abs(self.psi[self.inv_index].conj().dot(self.E_estimator.psi_GS_exact))**2
			Eloc_params_dict['overlap']=overlap

		
		elif self.mode=='MC':
			Eloc_params_dict=dict(N_MC_points=self.N_MC_points)

		
		self.Eloc_mean_g, self.Eloc_var_g, E_diff_real, E_diff_imag = self.E_estimator.process_local_energies(Eloc_params_dict)
		self.Eloc_std_g=np.sqrt(self.Eloc_var_g)
		self.E_MC_std_g=self.Eloc_std_g/np.sqrt(self.N_MC_points)
		

		Eloc_params_dict['Eloc_mean']=self.Eloc_mean_g
		Eloc_params_dict['Eloc_var']=self.Eloc_var_g

		self.Eloc_params_dict_log=Eloc_params_dict.copy()
		self.Eloc_params_dict_phase=Eloc_params_dict.copy()

		self.Eloc_params_dict_log['E_diff']  =E_diff_real
		self.Eloc_params_dict_phase['E_diff']=E_diff_imag
		
		##### total batch
		self.batch=self.MC_tool.spinstates_ket.reshape(self.input_shape)

		#return self.batch, self.Eloc_params_dict
		

	def update_NN_params(self,iteration):

		ti=time.time()

		if self.grad_update_mode=='normal':
			# order is important !!! (energy_lib stores log-values)
			self.DNN_phase.params, self.DNN_phase.params_update[:], self.r2[1] = self.opt_phase.return_grad(iteration, self.DNN_phase.params, self.batch, self.Eloc_params_dict_phase, )
			self.DNN_log.params,   self.DNN_log.params_update[:]  , self.r2[0]   = self.opt_log.return_grad(iteration, self.DNN_log.params, self.batch, self.Eloc_params_dict_log, )
			

		elif self.grad_update_mode=='alternating':
			if (iteration//self.alt_iters)%2==1: # phase grads
				self.DNN_log.params_update*=0.0
				self.DNN_phase.params_phase, self.DNN_phase.params_update[:], self.r2[1] = self.opt_phase.return_grad(iteration, self.DNN_phase.params, self.batch, self.Eloc_params_dict_phase, )

			else: # log grads
				self.DNN_log.params, self.DNN_log.params_update[:], self.r2[0] = self.opt_log.return_grad(iteration, self.DNN_log.params, self.batch, self.Eloc_params_dict_log, )
				self.DNN_phase.params_update*=0.0

		elif self.grad_update_mode=='phase':
			self.DNN_log.params_update*=0.0
			r2_log=0.0
			self.DNN_phase.params, self.DNN_phase.params_update[:], self.r2[1] = self.opt_phase.return_grad(iteration, self.DNN_phase.params, self.batch, self.Eloc_params_dict_phase, )


		elif self.grad_update_mode=='log_mod':
			self.DNN_log.params, self.DNN_log.params_update[:], self.r2[0] = self.opt_log.return_grad(iteration, self.DNN_log.params, self.batch, self.Eloc_params_dict_log, )
			self.DNN_phase.params_update*=0.0
			r2_phase=0.0

		#print(self.opt_phase.Runge_Kutta.step_size, self.opt_phase.step_size)

		##### compute max gradients
		grads_max=[np.max(np.abs(self.DNN_log.params_update)),np.max(np.abs(self.DNN_phase.params_update)),]
		
		
		mssg="total r2 test: {0:0.14f} .\n".format(self.r2[0]+self.r2[1]-1.0)
		if self.comm.Get_rank()==0:
			print(mssg)
		self.logfile.write(mssg)

		# record gradients

		if self.comm.Get_rank()==0:
			self.params_log_update_lastiters[-1,...]=self.DNN_log.params_update
			self.params_phase_update_lastiters[-1,...]=self.DNN_phase.params_update


		grad_str="total gradients/NG calculation took {0:.4f} secs.\n".format(time.time()-ti)
		self.logfile.write(grad_str)
		if self.comm.Get_rank()==0:
			print(grad_str)	
		
		return grads_max


