import sys,os
import numpy as np 
import pickle
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

################

# n=-3 #-3 # steps before final blow-up
# max_iter=313 #313 # last iteration with saved data
# data_name='2020-02-04_19:20:06_NG/'


###################
# discard_outliears: FALSE
# MC_thermal: TRUE

# n=-4 # steps before final blow-up
# max_iter=329 # last iteration with saved data
# data_name='2020-02-07_18:29:39_NG/' # 


###################
# discard_outliears: TRUE
# MC_thermal: TRUE

n=-2 # steps before final blow-up
max_iter=338 # last iteration with saved data
data_name='2020-02-08_05:53:59_NG/' # 


###################


#### load debug data


load_dir='data/' + data_name 
params_str='model_DNNcpx-mode_MC-L_6-J2_0.5-opt_NG-NNstrct_36--6-MCpts_20000-Nprss_130-NMCchains_1'



with open(load_dir + 'debug_files/' + 'debug-' + 'Eloc_data--' + params_str + '.pkl', 'rb') as handle:
	Eloc_real, Eloc_imag = pickle.load(handle)

with open(load_dir + 'debug_files/' + 'debug-' + 'SF_data--' + params_str + '.pkl', 'rb') as handle:
	S_lastiters, F_lastiters, delta = pickle.load(handle)

with open(load_dir + 'debug_files/' + 'debug-' + 'modpsi_data--' + params_str + '.pkl', 'rb') as handle:
	modpsi_kets, log_psi_shift = pickle.load(handle)
	#modpsi_kets, = pickle.load(handle)

with open(load_dir + 'debug_files/' + 'debug-' + 'phasepsi_data--' + params_str + '.pkl', 'rb') as handle:
	phasepsi_kets, = pickle.load(handle)

with open(load_dir + 'debug_files/' + 'debug-' + 'intkets_data--' + params_str + '.pkl', 'rb') as handle:
	int_kets, = pickle.load(handle)


######################
iteration=max_iter+n+1

file_name='NNparams'+'--iter_{0:05d}--'.format(iteration) + params_str

with open(load_dir + 'NN_params/' +file_name+'.pkl', 'rb') as handle:
	NN_params = pickle.load(handle)


######################

print('\niteration number: {0:d} with {1:d} unique spin configs.\n'.format(iteration, np.unique(int_kets[n,:]).shape[0]))

uq=np.unique(int_kets[n,:])
print(uq.shape)

print(Eloc_real[n,:20])

print(modpsi_kets[n,:20])

print(phasepsi_kets[n,:20])

exit()




######################

# print(int_kets[n,:10])
# MC_tool = MC_sample(NN_params,N_MC_points=10)
# print(MC_tool.ints_ket)


######################



#inds=np.where(np.abs(Eloc_real[n,:])>32000.0)[0]
inds=np.where(np.abs(Eloc_real[n,:])>50.0)[0]


print(inds)
#print(int_kets[n,inds])
print(Eloc_real[n,inds])
exit()

#inds=np.where(np.logical_and(np.abs(Eloc_real[n,:])>=16.5, np.abs(Eloc_real[n,:])<=16.8))[0]



log_psi_batch, phase_psi_batch = evaluate_DNN(NN_params,int_kets[n,inds], log_psi_shift=0.0)
Eloc_real_batch,Eloc_imag_batch = compute_Eloc(NN_params,int_kets[n,inds], log_psi_batch, phase_psi_batch, log_psi_shift=0.0)

# print(log_psi, phase_psi)
# print(np.log(modpsi_kets[n,inds]), phasepsi_kets[n,inds])
# exit()

print(Eloc_real_batch)
print(Eloc_real[n,inds])

#exit()


################


spin_state_ints=data_ints_ket = np.array([int_kets[n,inds[0]-20],],dtype=np.uint64)

log_psi_batch, phase_psi_batch =  evaluate_DNN(NN_params,spin_state_ints, log_psi_shift=0.0)
Eloc_real_batch,Eloc_imag_batch = compute_Eloc(NN_params,spin_state_ints, log_psi_batch, phase_psi_batch, log_psi_shift=0.0)

print(Eloc_real_batch)


