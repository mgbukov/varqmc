import sys,os
import numpy as np 
import pickle, yaml

sys.path.append("..")

from cpp_code import integer_to_spinstate, integer_cyclicity
from VMC_class import VMC


####################

L=4
N_batch=10
N_iter=480

# ED data
ED_data_file="data-GS_J1-J2_Lx={0:d}_Ly={1:d}_J1=1.0000_J2=0.5000.txt".format(L,L)
path_to_data="../../../../../Google Drive/frustration_from_RBM/ED_data/"


# NN params
#data_name='2020-02-07_18:29:39_NG/'
data_name='2020-02-14_09:47:56_NG/'
#data_name='2020-02-17_09:15:17_NG/' # ED

load_dir='data/' + data_name 
if L==6:
	N_data=15804956
	basis_dtype=np.uint64 
	params_str='model_DNNcpx-mode_MC-L_6-J2_0.5-opt_NG-NNstrct_36--6-MCpts_20000-Nprss_130-NMCchains_1'
elif L==4:
	N_data=107
	basis_dtype=np.uint16
	params_str='model_DNNcpx-mode_MC-L_4-J2_0.5-opt_NG-NNstrct_16--6-MCpts_200-Nprss_4-NMCchains_2'
	#params_str='model_DNNcpx-mode_exact-L_4-J2_0.5-opt_NG-NNstrct_16--6-MCpts_107-Nprss_1-NMCchains_2' # ED
else:
	'exiting'


file_name=load_dir + 'cost_funcs--' + params_str + '.txt'
file_cost_funcs= open(file_name, 'w')


# create DNN
params=dict(
		J2= 0.5,
		L= L,
		NN_dtype= 'cpx',
		NN_type= 'DNN',
		N_MC_chains= 1,
		N_MC_points= 1,
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
	)	
DNN_psi=VMC(params,train=False)




########################

#pre-allocate data

# C_KL_div=np.zeros(N_iter)
# C_cross_entropy=np.zeros_like(C_KL_div)

# C_overlap=np.zeros_like(C_KL_div)
# C_psi_L2=np.zeros_like(C_KL_div)

# C_log_psi_L2=np.zeros_like(C_KL_div)
# C_sign_psi_L2=np.zeros_like(C_KL_div)

# C_psi_L1_weighted=np.zeros_like(C_KL_div)
# C_log_psi_L1_weighted=np.zeros_like(C_KL_div)
# C_sign_psi_L1_weighted=np.zeros_like(C_KL_div)


print('USE pandas cvs reader isntead !!!')
exit()


for k, iteration in enumerate(range(N_iter)):



	file_name='NNparams'+'--iter_{0:05d}--'.format(iteration) + params_str

	with open(load_dir + 'NN_params/' +file_name+'.pkl', 'rb') as handle:
		NN_params, apply_fun_args, log_psi_shift = pickle.load(handle)

	DNN_psi.DNN.update_params(NN_params)
	DNN_psi.DNN.apply_fun_args=apply_fun_args
	

	# first loop to compute norm of DNN.psi

	norm2=0.0
	N_dim=0
	N_sweeps=N_data//N_batch+1
	for j in range(N_sweeps):
		
		spin_configs=np.loadtxt(path_to_data+ED_data_file,dtype=basis_dtype,delimiter=' ',usecols=0,max_rows=N_batch,skiprows=N_batch*j)
		# log_psi_ED=np.loadtxt(path_to_data+ED_data_file,dtype=np.float64,delimiter=' ',usecols=1,max_rows=N_batch,skiprows=N_batch*j)
		# sign_psi_ED=np.loadtxt(path_to_data+ED_data_file,dtype=np.float64,delimiter=' ',usecols=3,max_rows=N_batch,skiprows=N_batch*j)
		mult_ED=np.loadtxt(path_to_data+ED_data_file,dtype=np.float64,delimiter=' ',usecols=6,max_rows=N_batch,skiprows=N_batch*j)

		N_MC_points=spin_configs.shape[0]

		spinstates_ket=np.zeros((N_MC_points*DNN_psi.MC_tool.N_features,), dtype=np.int8)
		integer_to_spinstate(spin_configs, spinstates_ket, DNN_psi.N_features, NN_type=DNN_psi.DNN.NN_type)
		

		shape_tuple=(N_MC_points*DNN_psi.MC_tool.N_symm, DNN_psi.MC_tool.N_sites)
		log_psi, phase_psi = DNN_psi.evaluate_NN(NN_params, spinstates_ket.reshape(shape_tuple), DNN_psi.DNN.apply_fun_args)
		mod_psi_2=np.exp(2.0*log_psi)

		norm2+=np.sum(mult_ED*mod_psi_2)
		


	norm=np.sqrt(norm2)
	log_shift=np.log(norm)




	cross_entropy=0.0
	KL_div=0.0

	overlap=0.0

	#psi_L2=0.0

	log_psi_L2=0.0
	sign_psi_L2=0.0

	psi_L1_weighted=0.0
	log_psi_L1_weighted=0.0
	sign_psi_L1_weighted=0.0

	for j in range(N_sweeps):
		
		spin_configs=np.loadtxt(path_to_data+ED_data_file,dtype=basis_dtype,delimiter=' ',usecols=0,max_rows=N_batch,skiprows=N_batch*j)
		log_psi_ED=np.loadtxt(path_to_data+ED_data_file,dtype=np.float64,delimiter=' ',usecols=1,max_rows=N_batch,skiprows=N_batch*j)
		sign_psi_ED=np.loadtxt(path_to_data+ED_data_file,dtype=np.float64,delimiter=' ',usecols=3,max_rows=N_batch,skiprows=N_batch*j)
		mult_ED=np.loadtxt(path_to_data+ED_data_file,dtype=np.float64,delimiter=' ',usecols=6,max_rows=N_batch,skiprows=N_batch*j)

		N_dim+=np.sum(mult_ED)


		N_MC_points=spin_configs.shape[0]

		spinstates_ket=np.zeros((N_MC_points*DNN_psi.MC_tool.N_features,), dtype=np.int8)
		integer_to_spinstate(spin_configs, spinstates_ket, DNN_psi.N_features, NN_type=DNN_psi.DNN.NN_type)
		shape_tuple=(N_MC_points*DNN_psi.MC_tool.N_symm, DNN_psi.MC_tool.N_sites)

		log_psi, phase_psi = DNN_psi.evaluate_NN(NN_params, spinstates_ket.reshape(shape_tuple), DNN_psi.DNN.apply_fun_args)
		# normalize to prevent overflow
		log_psi-=log_shift
		

		p_ED=mult_ED*np.exp(2.0*log_psi_ED)

		# evaluate cost functions

		cross_entropy += -np.sum(p_ED * (np.log(mult_ED)+2.0*log_psi))
		KL_div        +=  np.sum(p_ED * 2.0*(log_psi_ED-log_psi))

		overlap += np.sum( mult_ED * sign_psi_ED*np.exp(log_psi_ED + log_psi + 1.0j*phase_psi) )

		#psi_L2  += np.sum( mult_ED * np.abs( sign_psi_ED*np.exp(log_psi_ED) - np.exp(log_psi + 1.0j*phase_psi) )**2  )
		
		log_psi_L2  += np.sum( mult_ED * np.abs(log_psi_ED - log_psi)**2  ) 
		sign_psi_L2 += np.sum( mult_ED * sign_psi_ED*np.exp(1j*phase_psi)) 

		psi_L1_weighted      += np.sum( p_ED * np.abs(sign_psi_ED*np.exp(log_psi_ED) - np.exp(log_psi+1j*phase_psi)) )
		log_psi_L1_weighted  += np.sum( p_ED * np.abs(log_psi_ED - log_psi) )
		sign_psi_L1_weighted += np.sum( p_ED * sign_psi_ED*np.exp(1j*phase_psi))


		print("finished loop {0:d}/{1:d}/{2:d}".format(j+1,N_sweeps,k)) 


	### normalize cost functions

	# C_KL_div[k]=KL_div
	# C_cross_entropy[k]=cross_entropy

	# C_overlap[k]=1.0-np.abs(overlap)**2
	# C_psi_L2[k] = np.sqrt(2.0*(1.0-np.real(overlap)))
	
	# C_log_psi_L2[k]=np.sqrt(log_psi_L2/N_dim)
	# C_sign_psi_L2[k]=1.0-np.abs(sign_psi_L2)/N_dim

	# C_psi_L1_weighted[k]=psi_L1_weighted
	# C_log_psi_L1_weighted[k]=log_psi_L1_weighted
	# C_sign_psi_L1_weighted[k]=1.0-np.abs(sign_psi_L1_weighted)

	#print(np.sqrt(psi_L2), np.sqrt(2.0*(1.0-np.real(overlap))))

	print("\nfinished iteration {0:d}\n".format(k))


	data_tuple=(iteration, KL_div, cross_entropy, 1.0-np.abs(overlap)**2, np.sqrt(2.0*(1.0-np.real(overlap))), np.sqrt(log_psi_L2/N_dim), 1.0-np.abs(sign_psi_L2)/N_dim, psi_L1_weighted, log_psi_L1_weighted, 1.0-np.abs(sign_psi_L1_weighted))
	file_cost_funcs.write("{0:d} : {1:0.14f} : {2:0.14f} : {3:0.14f} : {4:0.14f} : {5:0.14f} : {6:0.14f} : {7:0.14f} : {8:0.14f} : {9:0.14f}\n".format(*data_tuple))
		

file_cost_funcs.close()




