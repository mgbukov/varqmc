import sys,os
import numpy as np 
import pickle
import jax

path = "../."
sys.path.insert(0,path)
from cpp_code import NN_Tree

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.ticker import LogLocator,LinearLocator

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


matplotlib.use('Qt5Agg')

os.environ["PATH"] += ':/usr/local/texlive/2015/bin/x86_64-darwin'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.tick_params(labelsize=20)


################

from aux_funcs import *

sys.path.append("..")

from cpp_code import integer_to_spinstate
from VMC_class import VMC
import yaml 


n=-3 # steps before final blow-up
max_iter=228 # last iteration with saved E-data + 1
L=6
J2=0.5
opt='NG'
mode='MC'
NN_shape_str='36--8'
N_MC_points=20000
N_prss=130
NMCchains=1
sys_time='2020-02-24_23:19:34'


#### load debug data


data_name = sys_time + '--{0:s}-L_{1:d}-{2:s}/'.format(opt,L,mode)
load_dir='data/' + data_name 
data_params=(mode,L,J2,opt,NN_shape_str,N_MC_points,N_prss,NMCchains,)
params_str='model_DNNcpx-mode_{0:s}-L_{1:d}-J2_{2:0.1f}-opt_{3:s}-NNstrct_{4:s}-MCpts_{5:d}-Nprss_{6:d}-NMCchains_{7:d}'.format(*data_params)



with open(load_dir + 'debug_files/' + 'debug-' + 'intkets_data--' + params_str + '.pkl', 'rb') as handle:
	int_kets, = pickle.load(handle)
	if L==4:
		int_kets.astype(np.uint16)
	else:
		int_kets.astype(np.uint64) 


######################
iteration=max_iter+n+1

file_name='NNparams'+'--iter_{0:05d}--'.format(iteration) + params_str

with open(load_dir + 'NN_params/' +file_name+'.pkl', 'rb') as handle:
	NN_params, apply_fun_args, log_psi_shift = pickle.load(handle)


print('\niteration number: {0:d}.\n'.format(iteration, ))

########

# reconstruct DNN

params = yaml.load(open(load_dir+'config_params.yaml'),Loader=yaml.FullLoader)
params['N_MC_points']=int_kets[n,...].shape[0]
params['save_data']=False

DNN_psi=VMC(load_dir,params_dict=params,train=False)
DNN_psi.DNN.update_params(NN_params)


########

# get spin configs
spinstates_ket=np.zeros((DNN_psi.N_MC_points*DNN_psi.MC_tool.N_features,), dtype=np.int8)
integer_to_spinstate(int_kets[n,:], spinstates_ket, DNN_psi.N_features, NN_type=DNN_psi.DNN.NN_type)


# with jax.disable_jit():
# 	log_psi, phase_psi = DNN_psi.evaluate_NN(NN_params, spinstates_ket.reshape(DNN_psi.N_MC_points*DNN_psi.MC_tool.N_symm,DNN_psi.MC_tool.N_sites), DNN_psi.DNN.apply_fun_args)

# exit()


iteration=0
dlog_psi=DNN_psi.NG.compute_grad_log_psi(NN_params,spinstates_ket.reshape(DNN_psi.input_shape),iteration)._value


print('W-parameter:', np.mean(dlog_psi[...,0]), np.std(dlog_psi[...,0]), np.min(dlog_psi[...,0]), np.max(dlog_psi[...,0]) )

print('b-parameter:', np.mean(dlog_psi[...,-1]), np.std(dlog_psi[...,-1]), np.min(dlog_psi[...,-1]), np.max(dlog_psi[...,-1]) )



