import sys,os
import numpy as np 
import pickle

sys.path.append("..")

from cpp_code import integer_to_spinstate, representative
from VMC_class import VMC
import yaml 

from pandas import read_csv


################

def phase_histpgram(phase_psi, n_bins=40):

	
	phases = (phase_psi+np.pi)%(2*np.pi) - np.pi
	phase_hist, bin_edges = np.histogram(phases ,bins=n_bins,range=(-np.pi,np.pi), density=False, )

	return phase_hist


def extract_ED_signs(rep_spin_configs_ints,L,J2):

	
	rep_spin_configs_ints_uq, index, inv_index, count=np.unique(rep_spin_configs_ints, return_index=True, return_inverse=True, return_counts=True)
		
	Ns=rep_spin_configs_ints_uq.shape[0]


	N_batch=10000
	Marshal=False #True # 

	# ED data
	ED_data_file  ="data-GS_J1-J2_Lx={0:d}_Ly={1:d}_J1=1.0000_J2={2:0.4f}.txt".format(L,L,J2)
	ED_data_file_2="data-GS_J1-J2_Lx={0:d}_Ly={1:d}_J1=1.0000_J2={2:0.4f}.txt".format(L,L,0.0)
	path_to_data=os.path.expanduser('~') + '/Google_Drive/frustration_from_RBM/ED_data/'
	

	if L==6:
		N_data=15804956
		basis_dtype=np.uint64 
		
	elif L==4:
		N_data=107
		basis_dtype=np.uint16
	else:
		print('exiting')
		exit()


	N_sweeps=N_data//N_batch+1


	spin_int_states_it=read_csv(path_to_data+ED_data_file, chunksize=N_batch, header=None, dtype=basis_dtype,delimiter=' ',usecols=[0,]) 
	log_psi_ED_it=read_csv(path_to_data+ED_data_file, chunksize=N_batch, header=None, dtype=np.float64,delimiter=' ',usecols=[1,]) 
	sign_psi_ED_it=read_csv(path_to_data+ED_data_file, chunksize=N_batch, header=None, dtype=np.float64,delimiter=' ',usecols=[3,]) 
	mult_ED_it=read_csv(path_to_data+ED_data_file, chunksize=N_batch, header=None, dtype=np.float64,delimiter=' ',usecols=[6,]) 

	sign_psi_ED_it_J2_0=read_csv(path_to_data+ED_data_file_2, chunksize=N_batch, header=None, dtype=np.float64,delimiter=' ',usecols=[3,]) 
	

	log_psi_sample=np.zeros(Ns,)
	sign_psi_sample=np.zeros(Ns,)
	mult_sample=np.zeros(Ns,)
	p_sample=np.zeros(Ns,)
	sign_psi_sample_J2_0=np.zeros(Ns,)

	states=np.zeros(Ns,)

	n=0
	for j, (spin_ints_ED, log_psi_ED, sign_psi_ED, mult_ED,  sign_psi_ED_J2_0) in enumerate(zip(spin_int_states_it, log_psi_ED_it, sign_psi_ED_it, mult_ED_it, sign_psi_ED_it_J2_0)):
    
		p_ED=mult_ED.to_numpy().squeeze()*np.exp(2.0*log_psi_ED).to_numpy().squeeze()
		spin_ints_ED=spin_ints_ED.to_numpy().squeeze()

		#inds_sample=np.where(np.isin(rep_spin_configs_ints, spin_ints_ED))[0]
		
		
		intersection, inds_sample, inds_ED=np.intersect1d(rep_spin_configs_ints_uq, spin_ints_ED, assume_unique=False, return_indices=True)


		if intersection.shape[0]>0:	

			n+=inds_sample.shape[0]

			#print(rep_spin_configs_ints_uq[inds_sample]-spin_ints_ED[inds_ED])
		
			log_psi_sample[inds_sample]=log_psi_ED.to_numpy().squeeze()[inds_ED]
			sign_psi_sample[inds_sample]=sign_psi_ED.to_numpy().squeeze()[inds_ED]
			mult_sample[inds_sample]=mult_ED.to_numpy().squeeze()[inds_ED]
			p_sample[inds_sample]=p_ED[inds_ED]
			sign_psi_sample_J2_0[inds_sample]=sign_psi_ED_J2_0.to_numpy().squeeze()[inds_ED]

			#print(np.sum(np.abs(sign_psi_ED.to_numpy().squeeze()[inds_ED] - sign_psi_ED_J2_0.to_numpy().squeeze()[inds_ED])))
			#print(np.sum(np.abs(sign_psi_ED.to_numpy().squeeze() - sign_psi_ED_J2_0.to_numpy().squeeze())))



		print('finished sweep {0:d}/{1:d}; {2:d}/{3:d} states identified\n'.format(j+1,N_sweeps,n,Ns))
		#print(np.abs(np.sum(sign_psi_ED.to_numpy().squeeze()*sign_psi_ED_J2_0.to_numpy().squeeze()))/N_batch)

		if n==Ns:
			break


	if n<Ns:
		print('not all states encountered!')
		exit()

	log_psi_sample=np.array(log_psi_sample)[inv_index]
	sign_psi_sample=np.array(sign_psi_sample)[inv_index]
	mult_sample=np.array(mult_sample)[inv_index]
	p_sample=np.array(p_sample)[inv_index]
	sign_psi_sample_J2_0=np.array(sign_psi_sample_J2_0)[inv_index]


	return log_psi_sample, sign_psi_sample, mult_sample, p_sample, sign_psi_sample_J2_0


def MC_sample(load_dir,NN_params,N_MC_points=10,reps=False):

	params = yaml.load(open(load_dir+'config_params.yaml'),Loader=yaml.FullLoader)
	params['N_MC_points']=N_MC_points
	params['save_data']=False
				

	DNN_psi_MC=VMC(load_dir,params_dict=params,train=False)
	DNN_psi_MC.DNN.update_params(NN_params)

	acceptance_ratio_g = DNN_psi_MC.MC_tool.sample(DNN_psi_MC.DNN)
	print('\nacc ratio: {0:0.8f}.\n'.format(acceptance_ratio_g[0]))


	return DNN_psi_MC.MC_tool


def compute_reps(spin_configs_ints, L):

	if L==4:
		basis_type=np.uint16
	elif L==6:
		basis_type=np.uint64

	N_MC_points=spin_configs_ints.shape[0]

	# compute representatives
	rep_spin_configs_ints=np.zeros((N_MC_points,), dtype=basis_type)
	representative(spin_configs_ints, rep_spin_configs_ints)

	print('finished computing representatives')

	return rep_spin_configs_ints




def evaluate_DNN(load_dir,NN_params, spin_configs_ints, log_psi_shift=0.0,):

	params = yaml.load(open(load_dir+'config_params.yaml'),Loader=yaml.FullLoader)
	params['N_MC_points']=spin_configs_ints.shape[0]
	params['save_data']=False

	DNN_psi=VMC(load_dir,params_dict=params,train=False)
	DNN_psi.DNN.update_params(NN_params)


	spinstates_ket=np.zeros((DNN_psi.N_MC_points*DNN_psi.MC_tool.N_features,), dtype=np.int8)
	integer_to_spinstate(spin_configs_ints, spinstates_ket, DNN_psi.DNN.N_features, NN_type=DNN_psi.DNN.NN_type)

	log_psi, phase_psi = DNN_psi.evaluate_NN(NN_params, spinstates_ket.reshape(DNN_psi.N_MC_points*DNN_psi.MC_tool.N_symm,DNN_psi.MC_tool.N_sites), DNN_psi.DNN.apply_fun_args)


	return log_psi._value - log_psi_shift,   phase_psi._value



def compute_Eloc(load_dir,NN_params,ints_ket,log_psi,phase_psi,log_psi_shift=0.0,):

	params = yaml.load(open(load_dir+'config_params.yaml'),Loader=yaml.FullLoader)
	params['N_MC_points']=log_psi.shape[0]
	params['save_data']=False


	DNN_psi=VMC(load_dir,params_dict=params,train=False)
	DNN_psi.DNN.update_params(NN_params)

	#DNN_psi.E_estimator.get_exact_kets()


	phase_psi=np.array(phase_psi)
	log_psi=np.array(log_psi)


	DNN_psi.E_estimator.compute_local_energy(DNN_psi.evaluate_NN,DNN_psi.DNN,DNN_psi.DNN.params,ints_ket,log_psi,phase_psi,log_psi_shift,DNN_psi.minibatch_size)

	return DNN_psi.E_estimator.Eloc_real, DNN_psi.E_estimator.Eloc_imag


