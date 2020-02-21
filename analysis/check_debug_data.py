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

# n=-2 # steps before final blow-up
# max_iter=338 # last iteration with saved data
# data_name='2020-02-08_05:53:59_NG/' # 


###################


###################
# discard_outliears: TRUE
# MC_thermal: TRUE

# n=-5 # steps before final blow-up
# max_iter=349 # last iteration with saved data
# data_name='2020-02-11_09:19:46_NG/' # 


###################

# discard_outliears: FALSE
# MC_thermal: TRUE


# n=-4 # steps before final blow-up
# max_iter=328 # last iteration with saved data
# data_name='2020-02-14_22:57:49_NG/' # 36x8
# L=6

###################

# discard_outliears: FALSE
# MC_thermal: TRUE


# n=-6 # steps before final blow-up
# max_iter=259 # last iteration with saved data
# data_name='2020-02-18_15:34:03_NG/' # 16x6
# L=4

###################

# discard_outliears: FALSE
# MC_thermal: TRUE


# n=-5 # steps before final blow-up
# max_iter=455 # last iteration with saved data
# data_name='2020-02-19_20:15:07_NG/' # 36x8
# L=6

###################

n=-6 # steps before final blow-up
max_iter=499 # last iteration with saved data
L=4
J2=0.5
opt='NG'
mode='exact'
NN_shape_str='16--8'
N_MC_points=107
N_prss=1
NMCchains=2
data_name='2020-02-21_09:22:35_{0:s}/'.format(opt) 


###################

# n=-6 # steps before final blow-up
# max_iter=499 # last iteration with saved data
# L=4
# J2=0.5
# opt='NG'
# mode='MC'
# NN_shape_str='16--8'
# N_MC_points=200
# N_prss=4
# NMCchains=2
# data_name='2020-02-21_08:51:56_{0:s}/'.format(opt) 


#### load debug data


load_dir='data/' + data_name 
data_params=(mode,L,J2,opt,NN_shape_str,N_MC_points,N_prss,NMCchains,)
params_str='model_DNNcpx-mode_{0:s}-L_{1:d}-J2_{2:0.1f}-opt_{3:s}-NNstrct_{4:s}-MCpts_{5:d}-Nprss_{6:d}-NMCchains_{7:d}'.format(*data_params)




with open(load_dir + 'debug_files/' + 'debug-' + 'Eloc_data--' + params_str + '.pkl', 'rb') as handle:
	Eloc_real, Eloc_imag = pickle.load(handle)

with open(load_dir + 'debug_files/' + 'debug-' + 'SF_data--' + params_str + '.pkl', 'rb') as handle:
	S_lastiters, F_lastiters, delta = pickle.load(handle)

with open(load_dir + 'debug_files/' + 'debug-' + 'logpsi_data--' + params_str + '.pkl', 'rb') as handle:
	logpsi_kets, _ = pickle.load(handle)
	
with open(load_dir + 'debug_files/' + 'debug-' + 'phasepsi_data--' + params_str + '.pkl', 'rb') as handle:
	phasepsi_kets, = pickle.load(handle)

with open(load_dir + 'debug_files/' + 'debug-' + 'params_update_data--' + params_str + '.pkl', 'rb') as handle:
	NN_params_update, = pickle.load(handle)

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


######################

print('\niteration number: {0:d} with {1:d} unique spin configs & {2:d} unique Elocs.\n'.format(iteration, np.unique(int_kets[n,:]).shape[0], np.unique(Eloc_real[n,:].round(decimals=10)).shape[0]))


#exit()


# print("norm_layer", NN_params[-1])



print("F_vector:", np.min(F_lastiters[n,:]), np.max(F_lastiters[n,:]), )
print("S_matrix:", np.min(S_lastiters[n,:]), np.max(S_lastiters[n,:]), ) 
#exit()



# print(Eloc_real[n,:20])
# print(logpsi_kets[n,:20])
# print(phasepsi_kets[n,:20])
# print()

#exit()




######################

# print(int_kets[n,:10])
# MC_tool = MC_sample(load_dir, NN_params,N_MC_points=10)
# print(MC_tool.ints_ket)

# exit()

######################




data=Eloc_real[n,:]

q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
#q25, q75 = np.percentile(data, 0.05), np.percentile(data, 99.95)
#q25, q75 = np.percentile(data, 0.3), np.percentile(data, 99.62)
iqr = q75 - q25
print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
# calculate the outlier cutoff
cut_off = iqr * 1.5
lower, upper = q25 - cut_off, q75 + cut_off
# identify outliers
outliers = [x for x in data if x < lower or x > upper]
reamainers = [x for x in data if x >= lower and x <= upper]

print(np.array(outliers).shape)

print(np.min(outliers), np.max(outliers))

#print(np.sort(outliers))
print(np.mean(reamainers), np.mean(data))


#exit()

#############

# kets=np.array([65280, 65152, 65088, 65040, 65032, 65028, 65025, 64704, 64672, 64656, 64648, 64644,
#  64642, 64641, 64560, 64552, 64548, 64546, 64545, 64524, 64522, 64521, 64515, 64160,
#  64136, 64132, 64130, 64080, 64072, 64068, 64065, 64010, 64005, 63712, 63696, 63684,
#  63682, 63681, 63656, 63652, 63650, 63600, 63592, 63585, 63576, 63570, 61680, 61057,
#  60993, 60945, 60804, 60802, 60801, 60744, 60738, 60737, 60705, 60690, 60577, 60562,
#  60561, 60513, 60498, 60497, 60483, 60468, 60466, 60465, 60453, 60451, 60438, 60435,
#  60290, 60225, 60180, 60097, 60065, 60052, 60050, 60037, 60035, 59992, 59988, 59985,
#  59977, 59973, 59930, 59925, 59812, 59752, 59745, 59730, 59715, 59700, 59685, 59570,
#  59505, 58788, 58785, 58545, 57825, 52275, 51795, 51765, 51510, 50115, 42405,],dtype=np.uint16)


#kets, inds, inv_index, count=np.unique(int_kets[n,:], return_index=True, return_inverse=True, return_counts=True)
#kets=np.array([43940],dtype=np.uint16)

# print(np.where(int_kets[n,:].astype(np.uint16)==kets[0])[0])

# print(Eloc_real[n,np.where(int_kets[n,:].astype(np.uint16)==kets[0])[0]] )
# print(logpsi_kets[n,np.where(int_kets[n,:].astype(np.uint16)==kets[0])[0]])

# int_to_spinconfig(kets[0],L)

# exit()

# log_psi_batch, phase_psi_batch = evaluate_DNN(NN_params,kets, log_psi_shift=log_psi_shift[n], L=L)
# Eloc_real_batch,Eloc_imag_batch = compute_Eloc(NN_params,kets, log_psi_batch, phase_psi_batch, log_psi_shift=log_psi_shift[n], L=L)


##############
#inds=np.where(np.logical_and(np.abs(Eloc_real[n,:])>=16.5, np.abs(Eloc_real[n,:])<=16.8))[0]
inds=[np.where(np.in1d(Eloc_real[n,:], outliers)) [0][-1]]


# print(logpsi_kets[n,inds[0]-70:inds[0]+2])

# print(int_kets[n,inds[0]-70:inds[0]+2])

# int_to_spinconfig(int_kets[n,inds][0],L)

# print(np.max(logpsi_kets[n,:]), np.mean(logpsi_kets[n,:]))

# #print(np.sort(logpsi_kets[n,:])[:-3])

# print(inds[0]+2)

# print(logpsi_kets[n,:])

#exit()

#inds=np.arange(107).astype(np.uint16)


log_psi_batch, phase_psi_batch = evaluate_DNN(load_dir, NN_params,int_kets[n,inds], log_psi_shift=log_psi_shift, )
Eloc_real_batch,Eloc_imag_batch = compute_Eloc(load_dir, NN_params,int_kets[n,inds], log_psi_batch, phase_psi_batch, log_psi_shift=log_psi_shift, )



# print(log_psi_batch, phase_psi_batch)
# print(logpsi_kets[n,inds], phasepsi_kets[n,inds])
# exit()


print(np.min(Eloc_real_batch), np.max(Eloc_real_batch),)

print(Eloc_real_batch[:10])
print(Eloc_real[n,inds][:10])

exit()


################


spin_state_ints=data_ints_ket = np.array([int_kets[n,inds[0]-20],],dtype=np.uint64)

log_psi_batch, phase_psi_batch =  evaluate_DNN(load_dir, NN_params,spin_state_ints, log_psi_shift=log_psi_shift, )
Eloc_real_batch,Eloc_imag_batch = compute_Eloc(load_dir, NN_params,spin_state_ints, log_psi_batch, phase_psi_batch, log_psi_shift=log_psi_shift, )

print(Eloc_real_batch)


