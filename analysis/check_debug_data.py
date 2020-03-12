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

from aux_funcs import *

sys.path.append("..")

from cpp_code import integer_to_spinstate
from VMC_class import VMC
import yaml 


#########################


n=-10 # steps before final blow-up
max_iter=380 # last line with saved E-data
L=4
J2=0.5
opt='NG'
mode='MC'
NN_dtype='real-decoupled'
NN_shape_str='(16--10,16--24--12)'
N_MC_points=20000
N_prss=130
NMCchains=1
sys_time='2020-03-10_22:32:06'


#### load debug data


data_name = sys_time + '--{0:s}-L_{1:d}-{2:s}/'.format(opt,L,mode)
load_dir='data/' + data_name 
data_params=(NN_dtype,mode,L,J2,opt,NN_shape_str,N_MC_points,N_prss,NMCchains,)
params_str='model_DNN{0:s}-mode_{1:s}-L_{2:d}-J2_{3:0.1f}-opt_{4:s}-NNstrct_{5:s}-MCpts_{6:d}-Nprss_{7:d}-NMCchains_{8:d}'.format(*data_params)




with open(load_dir + 'debug_files/' + 'debug-' + 'Eloc_data--' + params_str + '.pkl', 'rb') as handle:
	Eloc_real, Eloc_imag = pickle.load(handle)

with open(load_dir + 'debug_files/' + 'debug-' + 'SF_data--' + params_str + '.pkl', 'rb') as handle:
	S_lastiters, F_lastiters, delta = pickle.load(handle)

with open(load_dir + 'debug_files/' + 'debug-' + 'logpsi_data--' + params_str + '.pkl', 'rb') as handle:
	logpsi_kets, _ = pickle.load(handle)
	
with open(load_dir + 'debug_files/' + 'debug-' + 'phasepsi_data--' + params_str + '.pkl', 'rb') as handle:
	phasepsi_kets, = pickle.load(handle)

# with open(load_dir + 'debug_files/' + 'debug-' + 'params_update_data--' + params_str + '.pkl', 'rb') as handle:
# 	NN_params_update, = pickle.load(handle)
# 	NN_params_update[:-1,...]=NN_params_update[1:,...]
# 	NN_params_update[-1,...]=0.0

with open(load_dir + 'debug_files/' + 'debug-' + 'intkets_data--' + params_str + '.pkl', 'rb') as handle:
	int_kets, = pickle.load(handle)
	if L==4:
		int_kets.astype(np.uint16)
	else:
		int_kets.astype(np.uint64) 


######################
iteration=max_iter+n+2

file_name='NNparams'+'--iter_{0:05d}--'.format(iteration) + params_str

with open(load_dir + 'NN_params/' +file_name+'.pkl', 'rb') as handle:
	NN_params, apply_fun_args, log_psi_shift = pickle.load(handle)

Tree=NN_Tree(NN_params)
NN_params_ravelled=Tree.ravel(NN_params)


######################

print('\niteration number: {0:d} with {1:d} unique spin configs & {2:d} unique Elocs & mean {3:0.4f}.\n'.format(iteration, np.unique(int_kets[n,:]).shape[0], np.unique(Eloc_real[n,:].round(decimals=10)).shape[0], np.mean(Eloc_real[n,:])) )



# print("norm_layer", NN_params[-1])

# print( logpsi_kets[n,:].max()+log_psi_shift )
# exit()


print("F_vector:", np.min(F_lastiters[n,:]), np.max(F_lastiters[n,:]), )
print("S_matrix:", np.min(S_lastiters[n,:]), np.max(S_lastiters[n,:]), )
print() 

E_S, V_S = eigh(S_lastiters[n,:])
nat_grad=inv(S_lastiters[n,:]).dot(F_lastiters[n,:])

#print("NN_params:", NN_params )

#print(NN_params_update[n,:][-1], nat_grad[-1])
#exit()


#print(-1E-2*NN_params_update[n,:][-1], NN_params_ravelled[-1],)
#exit()

print('S_bb:',S_lastiters[n,-1,-1])


# plt.plot(F_lastiters[n,:],'.b')
# plt.plot(inv(S_lastiters[n,...])[0,:],'.r', markersize=1.0)
# #plt.plot(E_S,'.b')
# #plt.yscale('log')
# plt.show()


#exit()


#print(Eloc_real[n,:20])
# print(logpsi_kets[n,:20])
# print(phasepsi_kets[n,:20])
# print()

#print(np.mean(Eloc_real[n,:]))


# print(logpsi_kets[n,:].min()+log_psi_shift, logpsi_kets[n,:].max()+log_psi_shift)
# exit()




######################

# with jax.disable_jit():
# 	MC_tool = MC_sample(load_dir, NN_params,N_MC_points=1)
# print(MC_tool.ints_ket)

# exit()

######################




data=Eloc_real[n,:]

#q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
q25, q75 = np.percentile(data, 0.05), np.percentile(data, 99.95)
#q25, q75 = np.percentile(data, 0.001), np.percentile(data, 99.991)
iqr = q75 - q25
print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
# calculate the outlier cutoff
cut_off = iqr * 1.5
lower, upper = q25 - cut_off, q75 + cut_off
# identify outliers
outliers = np.array([x for x in data if x < lower or x > upper])
reamainers = [x for x in data if x >= lower and x <= upper]



print('outliers shape:', outliers.shape)

print('min/max outliers:', np.min(outliers), np.max(outliers))

#print(np.sort(outliers))
print('mean remainers/data:',np.mean(reamainers), np.mean(data))


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




print( Eloc_real[n, np.where(np.in1d(Eloc_real[n,:], outliers))] )
inds=[np.where(np.in1d(Eloc_real[n,:], outliers)) [0][-1]]
#exit()


# print(logpsi_kets[n,inds[0]-70:inds[0]+2])

# print(int_kets[n,inds[0]-70:inds[0]+2])

# int_to_spinconfig(int_kets[n,inds][0],L)

# print(np.max(logpsi_kets[n,:]), np.mean(logpsi_kets[n,:]))

# #print(np.sort(logpsi_kets[n,:])[:-3])

# print(inds[0]+2)

# print(logpsi_kets[n,:])

#exit()

#inds=np.arange(107).astype(np.uint16)


#int_to_spinconfig(int_kets[n,inds][0],L)



log_psi_batch, phase_psi_batch = evaluate_DNN(load_dir, NN_params,int_kets[n,inds], log_psi_shift=log_psi_shift, )
Eloc_real_batch,Eloc_imag_batch = compute_Eloc(load_dir, NN_params,int_kets[n,inds], log_psi_batch, phase_psi_batch, log_psi_shift=log_psi_shift, )



# print(log_psi_batch, phase_psi_batch)
# print(logpsi_kets[n,inds], phasepsi_kets[n,inds])
np.testing.assert_allclose(log_psi_batch,logpsi_kets[n,inds])
#exit()



print(np.min(Eloc_real_batch), np.max(Eloc_real_batch),)

print(Eloc_real_batch[:10])
print(Eloc_real[n,inds][:10])

exit()


################


spin_state_ints=data_ints_ket = np.array([int_kets[n,inds[0]-20],],dtype=np.uint64)

log_psi_batch, phase_psi_batch =  evaluate_DNN(load_dir, NN_params,spin_state_ints, log_psi_shift=log_psi_shift, )
Eloc_real_batch,Eloc_imag_batch = compute_Eloc(load_dir, NN_params,spin_state_ints, log_psi_batch, phase_psi_batch, log_psi_shift=log_psi_shift, )

print(Eloc_real_batch)


