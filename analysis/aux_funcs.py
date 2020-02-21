import sys,os
import numpy as np 
import pickle

sys.path.append("..")

from cpp_code import integer_to_spinstate
from VMC_class import VMC
import yaml 


################


def int_to_spinconfig(s,L):
	S=np.array(list("{0:0{1:d}b}".format(s,L**2)))
	print(S.reshape(L,L))


def MC_sample(NN_params,N_MC_points=10,L=6 ):

	params=dict(
			J2= 0.5,
			L= L,
			NN_dtype= 'cpx',
			NN_shape_str= '36--6',
			NN_type= 'DNN',
			N_MC_chains= 4,
			N_MC_points= N_MC_points,
			N_batch= N_MC_points,
			N_iterations= 3,
			batchnorm= False,
			load_data= False,
			minibatch_size= 100,
			mode= 'MC',
			grad_update_mode= 'normal',
			MC_thermal= True,
			optimizer= 'NG',
			save_data= False,
			seed= 0,
			start_iter= 0,
			stop_iter= 0,
		)			

	DNN_psi_MC=VMC(params,train=False)
	DNN_psi_MC.DNN.update_params(NN_params)

	acceptance_ratio_g = DNN_psi_MC.MC_tool.sample(DNN_psi_MC.DNN)
	print('\nacc ratio: {0:0.8f}.\n'.format(acceptance_ratio_g[0]))

	return DNN_psi_MC.MC_tool


def evaluate_DNN(NN_params, spin_configs, log_psi_shift=0.0,L=6):

	N_MC_points=spin_configs.shape[0]

	params=dict(
		J2= 0.5,
		L= L,
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
		grad_update_mode= 'normal',
		MC_thermal= True,
		optimizer= 'NG',
		save_data= False,
		seed= 0,
		start_iter= 0,
		stop_iter= 0,
	)	

	DNN_psi=VMC(params,train=False)
	DNN_psi.DNN.update_params(NN_params)


	spinstates_ket=np.zeros((DNN_psi.N_MC_points*DNN_psi.MC_tool.N_features,), dtype=np.int8)
	integer_to_spinstate(spin_configs, spinstates_ket, DNN_psi.N_features, NN_type=DNN_psi.DNN.NN_type)

	log_psi, phase_psi = DNN_psi.evaluate_NN(NN_params, spinstates_ket.reshape(DNN_psi.N_MC_points*DNN_psi.MC_tool.N_symm,DNN_psi.MC_tool.N_sites), DNN_psi.DNN.apply_fun_args)


	return log_psi._value - log_psi_shift,   phase_psi._value


def compute_Eloc(NN_params,spin_configs,log_psi,phase_psi,log_psi_shift=0.0,L=6):

	N_MC_points=log_psi.shape[0]

	params=dict(
		J2= 0.5,
		L= L,
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
		grad_update_mode= 'normal',
		optimizer= 'NG',
		save_data= False,
		seed= 0,
		start_iter= 0,
		stop_iter= 0,
	)	

	DNN_psi=VMC(params,train=False)
	DNN_psi.DNN.update_params(NN_params)

	#DNN_psi.E_estimator.get_exact_kets()


	phase_psi=np.array(phase_psi)
	mod_psi=np.exp(np.array(log_psi))



	DNN_psi.E_estimator.compute_local_energy(DNN_psi.evaluate_NN,DNN_psi.DNN,DNN_psi.DNN.params,spin_configs,mod_psi,phase_psi,log_psi_shift,DNN_psi.minibatch_size)

	return DNN_psi.E_estimator.Eloc_real, DNN_psi.E_estimator.Eloc_imag





