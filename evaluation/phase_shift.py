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


from scipy.linalg import eigh, inv

np.random.seed(0)

matplotlib.use('Qt5Agg')

os.environ["PATH"] += ':/usr/local/texlive/2015/bin/x86_64-darwin'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.tick_params(labelsize=20)


################

from eval_lib import *

sys.path.append("..")

from cpp_code import integer_to_spinstate
from VMC_class import VMC
import yaml 



def wrap_phases(phase_psi,log_psi,):

	# wrap phases
	phase_psi      = (phase_psi+np.pi)%(2*np.pi)      - np.pi
	
	# shift phases
	ind=np.argmax(log_psi)
	a=phase_psi[ind]
	b=log_psi[ind]

	phase_psi=phase_psi-a
	
	log_psi=log_psi-b
	

	return phase_psi,log_psi,


#########################

save= True # False # 

iteration=599 #100,150,200,250,300,350,500,600,
J2=0.5
L=6
opt='CNNreal-RK_RK' # 'sgd_sgd' # 'sgd_sgd' #  
cost='SR_SR' # 'SR_SR' #
mode='ED' # 'MC' # 'exact' #
NN_type='CNNreal'
sys_time='2020_07_24-18_57_48' 



#### load debug data


data_name = sys_time + '--{0:s}-{1:s}-L_{2:d}-{3:s}/'.format(opt,cost,L,mode)
#load_dir='data/' + data_name 
#load_dir='data/paper_data/seeds/' + data_name 
#load_dir='data/paper_data/MC_samples/' + data_name
load_dir='data/paper_data/exact/' + data_name  

#data_params=(NN_dtype,mode,L,J2,opt,NN_shape_str,N_MC_points,N_prss,NMCchains,)
#params_str='--model_DNN{0:s}-mode_{1:s}-L_{2:d}-J2_{3:0.1f}-opt_{4:s}-NNstrct_{5:s}-MCpts_{6:d}-Nprss_{7:d}-NMCchains_{8:d}'.format(*data_params)
params_str=''



# with open(load_dir + 'debug_files/' + 'debug-' + 'Eloc_data--' + params_str + '.pkl', 'rb') as handle:
# 	Eloc_real, Eloc_imag = pickle.load(handle)

######################

#file_name='NNparams'+'--iter_{0:05d}--'.format(iteration) + params_str
file_name='NNparams--iter_{0:05d}'

with open(load_dir + 'NN_params/' +file_name.format(iteration)+'.pkl', 'rb') as handle:
	params_log, params_phase, _, _, _ = pickle.load(handle)


with open(load_dir + 'NN_params/' +file_name.format(iteration-1)+'.pkl', 'rb') as handle:
	params_log_prev, params_phase_prev, _, _, _ = pickle.load(handle)

tree_phase=NN_Tree(params_phase)
tree_log=NN_Tree(params_log)

params_phase_flat=tree_phase.ravel(params_phase)
params_phase_prev_flat=tree_phase.ravel(params_phase_prev)


grad_update = params_phase_flat-params_phase_prev_flat

# print(grad_update[:4])
# exit()

######################

print('\niteration number: {0:d}.\n'.format(iteration,))

############

# print(tree_phase.unravel(grad_update))
# exit()

# print(np.min(np.abs(params_phase_flat)), np.max(np.abs(params_phase_flat)))
# exit()


print('grad update min/max values:', np.min(np.abs(grad_update)), np.max(np.abs(grad_update)) )


# MC points
N_MC_points=10 

#with jax.disable_jit():
MC_tool = MC_sample(load_dir, params_log, N_MC_points=N_MC_points, reps=True)

rep_spin_configs_ints=compute_reps(MC_tool.ints_ket,L)

#print(rep_spin_configs_ints)

################



# log_psi, phase_psi = evaluate_DNN(load_dir, params_log, tree_phase.unravel(grad_update), rep_spin_configs_ints, )

# print(phase_psi)
# exit()


log_psi, phase_psi = evaluate_DNN(load_dir, params_log, params_phase, rep_spin_configs_ints, )
log_psi_prev, phase_psi_prev = evaluate_DNN(load_dir, params_log_prev, params_phase_prev, rep_spin_configs_ints, )


phase_psi_wrapped, log_psi_wrapped = wrap_phases(phase_psi,log_psi,)
phase_psi_wrapped_prev, log_psi_wrapped_prev = wrap_phases(phase_psi_prev,log_psi_wrapped,)

print(np.min(phase_psi), np.max(phase_psi))
#exit()


print(phase_psi - phase_psi_prev)
#exit()

print(phase_psi[[0,4,9]])
print(phase_psi_prev[[0,4,9]])
print()
print(phase_psi_wrapped[[0,4,9]])
print(phase_psi_wrapped_prev[[0,4,9]])
print()



