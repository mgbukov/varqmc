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


matplotlib.use('Qt5Agg')

os.environ["PATH"] += ':/usr/local/texlive/2015/bin/x86_64-darwin'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.tick_params(labelsize=20)


################

from aux import *

sys.path.append("..")

from cpp_code import integer_to_spinstate
from VMC_class import VMC
import yaml 


#########################

# n=-1
# iteration=999+n+1 # last line with saved E-data
# L=6
# J2=0.5
# opt='NG'
# mode='MC'
# NN_dtype='real-decoupled'
# NN_shape_str='(36--12,36--24--12)'
# N_MC_points=200
# N_prss=4
# NMCchains=2
# sys_time='2020_03_15-16_39_47'


# n=-1
# iteration=655+n+1 # last line with saved E-data
# L=6
# J2=0.5
# opt='NG'
# mode='MC'
# NN_dtype='real-decoupled'
# NN_shape_str='({0:d}--12,{0:d}--24--12)'.format(L**2)
# N_MC_points=100000
# N_prss=260
# NMCchains=1
# sys_time= '2020_03_15-19_12_08'


n=-1
iteration=700+n+1 # last line with saved E-data
L=4
J2=0.5
opt='NG'
mode='MC'
NN_dtype='real-decoupled'
NN_shape_str='({0:d}--10,{0:d}--24--12)'.format(L**2)
N_MC_points=80000
N_prss=130
NMCchains=1
sys_time= '2020_03_11-16_22_19'  






#### load debug data


data_name = sys_time + '--{0:s}-L_{1:d}-{2:s}/'.format(opt,L,mode)
load_dir='data/' + data_name 
data_params=(NN_dtype,mode,L,J2,opt,NN_shape_str,N_MC_points,N_prss,NMCchains,)
params_str='model_DNN{0:s}-mode_{1:s}-L_{2:d}-J2_{3:0.1f}-opt_{4:s}-NNstrct_{5:s}-MCpts_{6:d}-Nprss_{7:d}-NMCchains_{8:d}'.format(*data_params)


# with open(load_dir + 'debug_files/' + 'debug-' + 'intkets_data--' + params_str + '.pkl', 'rb') as handle:
# 	int_kets, = pickle.load(handle)
# 	if L==4:
# 		int_kets.astype(np.uint16)
# 	else:
# 		int_kets.astype(np.uint64) 


# with open(load_dir + 'debug_files/' + 'debug-' + 'Eloc_data--' + params_str + '.pkl', 'rb') as handle:
# 	Eloc_real, Eloc_imag = pickle.load(handle)

######################

file_name='NNparams'+'--iter_{0:05d}--'.format(iteration) + params_str

with open(load_dir + 'NN_params/' +file_name+'.pkl', 'rb') as handle:
	NN_params, apply_fun_args, log_psi_shift = pickle.load(handle)

Tree=NN_Tree(NN_params)
NN_params_ravelled=Tree.ravel(NN_params)



######################

print('\niteration number: {0:d}.\n'.format(iteration,))


######################


N_MC_points=1000 # MC points

with jax.disable_jit():
	MC_tool = MC_sample(load_dir, NN_params,N_MC_points=N_MC_points, reps=True)


#rep_spin_configs_ints=compute_reps(int_kets[n,:],L)
rep_spin_configs_ints=compute_reps(MC_tool.ints_ket,L)

#print(rep_spin_configs_ints)
#print(MC_tool.ints_ket)



log_psi, phase_psi = evaluate_DNN(load_dir,NN_params, rep_spin_configs_ints, )
sign_psi = np.exp(-1j*phase_psi)


# collapse sign_psi and re-evaluate the energy



Eloc_Re, Eloc_Im=compute_Eloc(load_dir,NN_params,rep_spin_configs_ints,log_psi,phase_psi,)

# local E for exact states
print(Eloc_Re)



log_psi_ED, sign_psi_ED, mult_ED, p_ED, sign_psi_ED_J2_0 = extract_ED_signs(rep_spin_configs_ints,L,J2)

# get phases
phase_psi_ED = np.pi*0.5*(sign_psi_ED+1.0)
phase_psi_ED_J2_0 = np.pi*0.5*(sign_psi_ED_J2_0+1.0)

#check logs
C_log     = np.mean( np.abs( (log_psi_ED - log_psi_ED[0]) - (log_psi - log_psi[0]) ) )
C_log_max = np.max(  np.abs( (log_psi_ED - log_psi_ED[0]) - (log_psi - log_psi[0]) ) )

# compute sign loss functions

def compute_costs(phase_psi_1, phase_psi_2):
	#C_sign_psi      = 1.0 - np.abs( np.sum(           sign_psi_ED*sign_psi ) )/N_MC_points
	
	hist_1 = phase_histpgram(phase_psi_1)
	hist_2 = phase_histpgram(phase_psi_2)

	hist_1=hist_1/np.linalg.norm(hist_1)
	hist_2=hist_2/np.linalg.norm(hist_2)

	N = len(hist_1)

	C_phase=np.zeros(N)
	for j in range(N):

		C_phase[j]=np.sqrt(hist_1[:].dot( np.roll(hist_2, j) ))

	return C_phase


C_phase_J2_05 = compute_costs(phase_psi, phase_psi_ED)
C_phase_J2_00 = compute_costs(phase_psi, phase_psi_ED_J2_0)
C_phase_mixed = compute_costs(phase_psi_ED, phase_psi_ED_J2_0)

#C_sign=np.abs(np.sum(sign_psi_ED*sign_psi_ED_J2_0))/N_MC_points
#C_sign=sign_psi_ED.dot(sign_psi_ED_J2_0) - (np.sum(sign_psi_ED)*np.sum(sign_psi_ED_J2_0))

print('costs:\n')
print(C_log,C_log_max)
print()
print(C_phase_J2_05)
print()
print(C_phase_J2_00)
print()
print(C_phase_mixed)
print()


print('E={0:0.8f}, E_std={1:0.8f}'.format(np.mean(Eloc_Re), np.std(Eloc_Re)/np.sqrt(N_MC_points)) )

print()
print(phase_histpgram(phase_psi))
print()
print(phase_histpgram(phase_psi_ED))
print()
print(phase_histpgram(phase_psi_ED_J2_0))





