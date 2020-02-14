import sys,os
import numpy as np 
import pickle, yaml

sys.path.append("..")

from cpp_code import integer_to_spinstate, integer_cyclicity
from VMC_class import VMC


####################


# ED data
ED_data_file="GS-data_J1-J2_Lx=6_Ly=6_J1=1.0000_J2=0.5000.txt"
path_to_data="../../../../../Google Drive/frustration_from_RBM/ED_data/"

N_batch=100#00
N_data=107 #15804956 # 

# NN params
data_name='2020-02-07_18:29:39_NG/'
load_dir='data/' + data_name 
params_str='model_DNNcpx-mode_MC-L_6-J2_0.5-opt_NG-NNstrct_36--6-MCpts_20000-Nprss_130-NMCchains_1'


# create DNN
N_MC_points=1
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
		MC_thermal= True,
		optimizer= 'NG',
		save_data= False,
		seed= 0,
		start_iter= 0,
		stop_iter= 0,
	)	
DNN_psi=VMC(params,train=False)



iteration=1

file_name='NNparams'+'--iter_{0:05d}--'.format(iteration) + params_str

with open(load_dir + 'NN_params/' +file_name+'.pkl', 'rb') as handle:
	NN_params = pickle.load(handle)

DNN_psi.DNN.update_params(NN_params)


# first loop to compute norm and average phase

norm2=0.0
phase_ave=0.0
sign_ave_ED=0.0
norm2_ED=0.0

N_sweeps=N_data//N_batch+1
for j in range(N_sweeps):
	
	spin_configs=np.loadtxt(path_to_data+ED_data_file,dtype=np.uint64,delimiter=' ',usecols=0,max_rows=N_batch,skiprows=N_batch*j)
	log_psi_ED=np.loadtxt(path_to_data+ED_data_file,dtype=np.float64,delimiter=' ',usecols=1,max_rows=N_batch,skiprows=N_batch*j)
	sign_psi_ED=np.loadtxt(path_to_data+ED_data_file,dtype=np.float64,delimiter=' ',usecols=3,max_rows=N_batch,skiprows=N_batch*j)
	mult_ED=np.loadtxt(path_to_data+ED_data_file,dtype=np.float64,delimiter=' ',usecols=6,max_rows=N_batch,skiprows=N_batch*j)

	N_MC_points=spin_configs.shape[0]

	spinstates_ket=np.zeros((N_MC_points*DNN_psi.MC_tool.N_features*2,), dtype=np.int8)
	integer_to_spinstate(spin_configs, spinstates_ket, DNN_psi.N_features, NN_type=DNN_psi.DNN.NN_type)

	
	print(576//cycl_let)
	exit()


	shape_tuple=(N_MC_points*DNN_psi.MC_tool.N_symm, DNN_psi.MC_tool.N_sites)
	log_psi, phase_psi = DNN_psi.evaluate_NN(NN_params, spinstates_ket.reshape(shape_tuple), DNN_psi.DNN.apply_fun_args)

	norm2+=np.sum(mult_ED*np.exp(2.0*log_psi))
	norm2_ED+=np.sum(mult_ED*np.exp(2.0*log_psi_ED))

	phase_ave+=np.sum(phase_psi)
	sign_ave_ED+=np.sum(sign_psi_ED)

	print("finished iteration {0:d}/{1:d}".format(j+1,N_sweeps))


norm=np.sqrt(norm2)
log_shift=np.log(norm)

norm_ED=np.sqrt(norm2_ED)
log_shift_ED=np.log(norm_ED)

print(norm, norm_ED)
exit()

phase_ave/=N_data
sign_ave_ED/=N_data


cross_entropy=0.0
KL_div=0.0

overlap=0.0


mod_psi_L2=0.0
sign_psi_L2=0.0

psi_L1_weighted=0.0
mod_psi_L1_weighted=0.0
sign_psi_L1_weighted=0.0

for j in range(N_sweeps):
	
	spin_configs=np.loadtxt(path_to_data+ED_data_file,dtype=np.uint64,delimiter=' ',usecols=0,max_rows=N_batch,skiprows=N_batch*j)
	log_psi_ED=np.loadtxt(path_to_data+ED_data_file,dtype=np.float64,delimiter=' ',usecols=1,max_rows=N_batch,skiprows=N_batch*j)
	sign_psi_ED=np.loadtxt(path_to_data+ED_data_file,dtype=np.float64,delimiter=' ',usecols=3,max_rows=N_batch,skiprows=N_batch*j)
	mult_ED=np.loadtxt(path_to_data+ED_data_file,dtype=np.float64,delimiter=' ',usecols=6,max_rows=N_batch,skiprows=N_batch*j)

	log_psi_ED-=log_shift_ED

	print(mult_ED*np.sum(np.exp(2.0*log_psi_ED)))
	exit()

	N_MC_points=spin_configs.shape[0]

	spinstates_ket=np.zeros((N_MC_points*DNN_psi.MC_tool.N_features,), dtype=np.int8)
	integer_to_spinstate(spin_configs, spinstates_ket, DNN_psi.N_features, NN_type=DNN_psi.DNN.NN_type)
	shape_tuple=(N_MC_points*DNN_psi.MC_tool.N_symm, DNN_psi.MC_tool.N_sites)

	log_psi, phase_psi = DNN_psi.evaluate_NN(NN_params, spinstates_ket.reshape(shape_tuple), DNN_psi.DNN.apply_fun_args)
	# normalize to prevent overflow
	log_psi-=log_shift
	phase_psi-=phase_ave

	p_ED=np.exp(2.0*log_psi_ED)

	diff=log_psi-log_psi_ED 
	cross_entropy += mult_ED*np.sum(diff)
	KL_div        += mult_ED*np.sum(p_ED * diff)

	overlap += np.sum( mult_ED * sign_psi_ED*np.exp(log_psi_ED + log_psi + 1.0j*phase_psi) )

	mod_psi_L2  += np.sum(mult_ED * np.abs(np.exp(log_psi_ED) - np.exp(log_psi))**2)
	sign_psi_L2 += np.sum(mult_ED * np.abs(sign_psi_ED - np.exp(1j*phase_psi))**2)

	psi_L1_weighted      += np.sum( mult_ED * p_ED*np.abs(sign_psi_ED*np.exp(log_psi_ED) - np.exp(log_psi+1j*phase_psi)) )
	mod_psi_L1_weighted  += np.sum( mult_ED * p_ED*np.abs(            np.exp(log_psi_ED) - np.exp(log_psi)             ) )
	sign_psi_L1_weighted += np.sum( mult_ED * p_ED*np.abs(sign_psi_ED                    - np.exp(       +1j*phase_psi)) )


psi_L2=2.0*(1.0-np.real(overlap))
overlap=1.0-np.abs(overlap)**2


"""
# cost functions

- log(p_ED/p_DNN)

- p_ED log(p_ED/p_DNN) 


1.0 - |<psi_ED|psi_DNN>|^2


|psi_ED - psi_DNN|^2 = 2[1 - Re{<psi_ED|psi_DNN>}] 

|mod_psi_ED - mod_psi_DNN|^2 

|sign_psi_ED - sign_psi_DNN|^2 


p_ED |psi_ED - psi_DNN| 

p_ED |mod_psi_ED - mod_psi_DNN| 

p_ED |sign_psi_ED - sign_psi_DNN|
"""






