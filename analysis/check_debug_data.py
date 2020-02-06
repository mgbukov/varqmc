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

sys.path.append("..")

from cpp_code import integer_to_spinstate
from VMC_class import VMC
import yaml 

################


#### load debug data

#data_name='2020-02-05_01:18:37_NG/'
#data_name='2020-02-04_19:20:06_NG/'
data_name='2020-02-06_02:20:06_NG/'

load_dir='data/' + data_name 
params_str='model_DNNcpx-mode_MC-L_6-J2_0.5-opt_NG-NNstrct_36--6-MCpts_20000-Nprss_130-NMCchains_1'



with open(load_dir + 'debug_files/' + 'debug-' + 'Eloc_data--' + params_str + '.pkl', 'rb') as handle:
	Eloc_real, Eloc_imag = pickle.load(handle)

with open(load_dir + 'debug_files/' + 'debug-' + 'SF_data--' + params_str + '.pkl', 'rb') as handle:
	S_lastiters, F_lastiters, delta = pickle.load(handle)

with open(load_dir + 'debug_files/' + 'debug-' + 'modpsi_data--' + params_str + '.pkl', 'rb') as handle:
	modpsi_kets, = pickle.load(handle)

with open(load_dir + 'debug_files/' + 'debug-' + 'phasepsi_data--' + params_str + '.pkl', 'rb') as handle:
	phasepsi_kets, = pickle.load(handle)

with open(load_dir + 'debug_files/' + 'debug-' + 'intkets_data--' + params_str + '.pkl', 'rb') as handle:
	int_kets, = pickle.load(handle)



######################
n=-3
iteration=337+n

file_name='NNparams'+'--iter_{0:05d}--'.format(iteration) + params_str

with open(load_dir + 'NN_params/' +file_name+'.pkl', 'rb') as handle:
	NN_params = pickle.load(handle)



######################


def int_to_spinconfig(s):
	S=np.array(list("{0:036b}".format(s)))
	print(S.reshape(6,6))






#print(np.max(np.abs(Eloc_real[n,:])))

inds=np.where(np.abs(Eloc_real[n,:])>500.0)[0]

#inds=np.where(np.logical_and(np.abs(Eloc_real[n,:])>=16.5, np.abs(Eloc_real[n,:])<=16.8))[0]

#print(inds)
#exit()


print()


# for ind in inds:
# 	s=int_kets[n,:][ind]
# 	print(s, Eloc_real[n,:][ind], modpsi_kets[n,:][ind])
# 	int_to_spinconfig(s)
# 	print()


########

#inds=np.array(inds[0],)

N_MC_points=inds.shape[0]

params=dict(
		J2= 0.5,
		L= 6,
		NN_dtype= 'cpx',
		NN_shape_str= '36--6',
		NN_type= 'DNN',
		N_MC_chains= 1,
		N_MC_points= N_MC_points,
		N_batch= N_MC_points,
		N_iterations= 3,
		batchnorm= False,
		load_data= False,
		minibatch_size= 100,
		mode= 'MC',
		optimizer= 'NG',
		save_data= False,
		seed= 0,
		start_iter= 0,
		stop_iter= 0,
	)
			

DNN_psi=VMC(params,train=False)

# update parameters
DNN_psi.DNN.update_params(NN_params)


# data
data_ints_ket = int_kets[n,inds]
spinstates_ket=np.zeros((DNN_psi.N_MC_points*DNN_psi.MC_tool.N_features,),dtype=np.int8)

integer_to_spinstate(data_ints_ket, spinstates_ket, DNN_psi.N_features, NN_type=DNN_psi.DNN.NN_type)


#spinstates_ket.reshape(DNN_psi.N_MC_points*DNN_psi.MC_tool.N_symm,DNN_psi.MC_tool.N_sites)


log_psi, phase_psi = DNN_psi.evaluate_NN(NN_params, spinstates_ket.reshape(DNN_psi.N_MC_points*DNN_psi.MC_tool.N_symm,DNN_psi.MC_tool.N_sites), DNN_psi.DNN.apply_fun_args)

modpsi=modpsi_kets[n,inds]
log_psi_shift=0.0

#log_psi_shift=np.log(modpsi[0])
#modpsi/=modpsi[0]

print(log_psi)
print(np.log(modpsi))
exit()



DNN_psi.E_estimator.compute_local_energy(DNN_psi.evaluate_NN,DNN_psi.DNN,data_ints_ket,modpsi,phasepsi_kets[n,inds],log_psi_shift,DNN_psi.minibatch_size)
		

print(DNN_psi.E_estimator.Eloc_real)
print(Eloc_real[n,inds])

#print(np.mean(Eloc_real), np.mean(DNN_psi.E_estimator.Eloc_real))


