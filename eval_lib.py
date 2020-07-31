import sys,os
import numpy as np 
import pickle

#sys.path.append("..")

from cpp_code import integer_to_spinstate, representative
from cpp_code import update_offdiag_ME, update_diag_ME, c_offdiag_sum
from VMC_class import VMC
import yaml 

from pandas import read_csv


################

def phase_histpgram(phase_psi, n_bins=40):

	
	phases = (phase_psi+np.pi)%(2*np.pi) - np.pi
	phase_hist, bin_edges = np.histogram(phases ,bins=n_bins,range=(-np.pi,np.pi), density=False, )

	return phase_hist



def extract_ED_data(rep_spin_configs_ints,L,J2):

	
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
	sign_psi_ED_it=read_csv(path_to_data+ED_data_file, chunksize=N_batch, header=None, dtype=np.float64,delimiter=' ',usecols=[4,]) 
	mult_ED_it=read_csv(path_to_data+ED_data_file, chunksize=N_batch, header=None, dtype=np.float64,delimiter=' ',usecols=[6,]) 

	sign_psi_ED_it_J2_0=read_csv(path_to_data+ED_data_file_2, chunksize=N_batch, header=None, dtype=np.float64,delimiter=' ',usecols=[4,]) 
	

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


def _data_stream(data,minibatch_size,sample_size,N_minibatches):
    rng = np.random.RandomState(0)
    while True:
        perm = rng.permutation(sample_size)
        for i in range(N_minibatches):
            batch_idx = perm[i * minibatch_size : (i + 1) * minibatch_size]
            #batch_idx = np.arange(i*minibatch_size, min(sample_size, (i+1)*minibatch_size), 1)
            yield data[batch_idx], batch_idx


def bootstrap_sample(data, N_bootstrap, N_batch ):

	N_MC_points=data.shape[0]

	num_complete_batches, leftover = divmod(N_MC_points, N_batch)
	N_minibatches = num_complete_batches + bool(leftover)

	N_minibatches=1

	batches = _data_stream(data,N_batch,N_MC_points,N_minibatches)	

	Eloc_mean_s=np.zeros((N_bootstrap,N_minibatches,), dtype=np.complex128)
	Eloc_std_s=np.zeros((N_bootstrap,N_minibatches),)

	for i in range(N_bootstrap):
		for j in range(N_minibatches):
			batch, batch_idx = next(batches)

			Eloc_mean_s[i,j]=np.mean(batch, axis=0)
			Eloc_std_s[i,j]=np.std(np.abs(batch))

	return Eloc_mean_s.ravel(), Eloc_std_s.ravel() 




def MC_sample(load_dir,params_log,N_MC_points=10,reps=False):

	params = yaml.load(open(load_dir+'config_params.yaml'),Loader=yaml.FullLoader)
	params['N_MC_points']=N_MC_points
	params['save_data']=False
	params['mode']='MC'


	DNN_psi_MC=VMC(load_dir,params_dict=params,train=False)
	DNN_psi_MC.DNN_log.params=params_log


	acceptance_ratio_g = DNN_psi_MC.MC_tool.sample(DNN_psi_MC.DNN_log,DNN_psi_MC.DNN_phase)
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


def evaluate_overlap(load_dir,params_log, params_phase, L, J2, N_batch=10000):

	params = yaml.load(open(load_dir+'config_params.yaml'),Loader=yaml.FullLoader)
	params['N_MC_points']=N_batch
	params['save_data']=False

	DNN_psi=VMC(load_dir,params_dict=params,train=False)
	DNN_psi.DNN_log.params=params_log
	DNN_psi.DNN_phase.params=params_phase


	########


	
	
	# ED data
	ED_data_file  ="data-GS_J1-J2_Lx={0:d}_Ly={1:d}_J1=1.0000_J2={2:0.4f}.txt".format(L,L,J2)
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
	sign_psi_ED_it=read_csv(path_to_data+ED_data_file, chunksize=N_batch, header=None, dtype=np.float64,delimiter=' ',usecols=[4,]) 
	mult_ED_it=read_csv(path_to_data+ED_data_file, chunksize=N_batch, header=None, dtype=np.float64,delimiter=' ',usecols=[6,]) 


	log_psi_DNN=np.zeros(N_data,)
	phase_psi_DNN=np.zeros(N_data,)
	mult_DNN=np.zeros(N_data,)
	
	log_psi_exact=np.zeros(N_data,)
	sign_psi_exact=np.zeros(N_data,)
	

	spin_configs=np.zeros((N_batch*DNN_psi.MC_tool.N_features,),dtype=np.int8)

	n=0
	for j, (spin_ints_ED, log_psi_ED, sign_psi_ED, mult_ED,) in enumerate(zip(spin_int_states_it, log_psi_ED_it, sign_psi_ED_it, mult_ED_it,)):
    
		ind1=j*N_batch
		if (j+1)*N_batch>N_data:
			ind2=N_data
			inds_end=N_data%N_batch
		else:
			ind2=(j+1)*N_batch
			inds_end=N_batch

		
		#p_ED=mult_ED.to_numpy().squeeze()*np.exp(2.0*log_psi_ED).to_numpy().squeeze()
		spin_ints_ED=spin_ints_ED.to_numpy().squeeze()


		mult_DNN[ind1:ind2] = mult_ED.to_numpy().squeeze()
		
		log_psi_exact[ind1:ind2]  = log_psi_ED.to_numpy().squeeze()
		sign_psi_exact[ind1:ind2] = sign_psi_ED.to_numpy().squeeze()

		# evaluate NN

		integer_to_spinstate(spin_ints_ED, spin_configs, DNN_psi.MC_tool.N_features, NN_type=DNN_psi.NN_type)


		log_psi_DNN[ind1:ind2]   = DNN_psi.DNN_log  .evaluate(params_log,   spin_configs.reshape(DNN_psi.DNN_log  .input_shape), )[:inds_end]
		phase_psi_DNN[ind1:ind2] = DNN_psi.DNN_phase.evaluate(params_phase, spin_configs.reshape(DNN_psi.DNN_phase.input_shape), )[:inds_end]
		

		n+=N_batch

		print('finished sweep {0:d}/{1:d}; {2:d}/{3:d} states identified\n'.format(j+1,N_sweeps,n,N_data))

		# if j==2: #N_data:
		# 	break


	# if n<N_data:
	# 	print('not all states encountered!')
	# 	exit()

	
	
	norm_DNN=np.sqrt( np.sum( mult_DNN*np.exp(2.0*log_psi_DNN) ) )


	psi_DNN  =np.exp(1j*phase_psi_DNN + log_psi_DNN)/norm_DNN
	psi_exact=sign_psi_exact*np.exp(log_psi_exact)

	norm_test_DNN  =np.sum(mult_DNN*np.abs(psi_DNN)**2  )
	norm_test_exact=np.sum(mult_DNN*np.abs(psi_exact)**2)

	print('test norms', norm_test_DNN,  norm_test_exact)

	overlap=np.abs( np.sum( mult_DNN*psi_exact*psi_DNN ) )**2

	print('overlap', overlap)
	
	return overlap
	


def evaluate_DNN(load_dir,params_log, params_phase, spin_configs_ints, log_psi_shift=0.0,):

	params = yaml.load(open(load_dir+'config_params.yaml'),Loader=yaml.FullLoader)
	params['N_MC_points']=spin_configs_ints.shape[0]
	params['save_data']=False
	params['mode']='MC'

	DNN_psi=VMC(load_dir,params_dict=params,train=False)
	DNN_psi.DNN_log.params=params_log
	DNN_psi.DNN_phase.params=params_phase


	spinstates_ket=np.zeros((DNN_psi.N_MC_points*DNN_psi.MC_tool.N_features,), dtype=np.int8)
	integer_to_spinstate(spin_configs_ints, spinstates_ket, DNN_psi.DNN_log.N_features, NN_type=DNN_psi.DNN_log.NN_type)

	log_psi   = DNN_psi.DNN_log.evaluate(params_log, spinstates_ket    .reshape(DNN_psi.DNN_log.input_shape), )
	phase_psi = DNN_psi.DNN_phase.evaluate(params_phase, spinstates_ket.reshape(DNN_psi.DNN_phase.input_shape), )


	return log_psi._value - log_psi_shift,   phase_psi._value


def evaluate_sample(load_dir,params_log, params_phase,ints_ket,log_psi,phase_psi,log_psi_shift=0.0,):

	params = yaml.load(open(load_dir+'config_params.yaml'),Loader=yaml.FullLoader)
	params['N_MC_points']=log_psi.shape[0]
	params['save_data']=False
	params['mode']='MC'


	DNN_psi=VMC(load_dir,params_dict=params,train=False)
	DNN_psi.DNN_log.params=params_log
	DNN_psi.DNN_phase.params=params_phase

	#DNN_psi.E_estimator.get_exact_kets()


	phase_psi=np.array(phase_psi)
	log_psi=np.array(log_psi)


	DNN_psi.E_estimator.compute_s_primes(ints_ket,DNN_psi.NN_type)


	log_psi_bras = DNN_psi.E_estimator.evaluate_s_primes(DNN_psi.DNN_log.evaluate,params_log,DNN_psi.DNN_log.input_shape)
	log_psi_bras-=log_psi_shift
	

	psi_str="log_|psi|_bras: min={0:0.8f}, max={1:0.8f}, mean={2:0.8f}; std={3:0.8f}, diff={4:0.8f}.\n".format(np.min(log_psi_bras), np.max(log_psi_bras), np.mean(log_psi_bras), np.std(log_psi_bras), np.max(log_psi_bras)-np.min(log_psi_bras) )
	print(psi_str)
	

	phase_psi_bras = DNN_psi.E_estimator.evaluate_s_primes(DNN_psi.DNN_phase.evaluate,params_phase,DNN_psi.DNN_phase.input_shape)


	return log_psi, phase_psi,  phase_psi_bras, log_psi_bras



def compute_Eloc(load_dir,params_log, params_phase,ints_ket,log_psi,phase_psi,log_psi_shift=0.0,):

	# ints_ket=np.array([ints_ket[0]])
	# log_psi=np.array(log_psi[0])
	# phase_psi=np.array(phase_psi[0])


	params = yaml.load(open(load_dir+'config_params.yaml'),Loader=yaml.FullLoader)
	params['N_MC_points']=log_psi.shape[0]
	params['save_data']=False


	DNN_psi=VMC(load_dir,params_dict=params,train=False)
	DNN_psi.DNN_log.params=params_log
	DNN_psi.DNN_phase.params=params_phase

	#DNN_psi.E_estimator.get_exact_kets()


	phase_psi=np.array(phase_psi)
	log_psi=np.array(log_psi)


	DNN_psi.E_estimator.compute_local_energy(params_log, params_phase, ints_ket, log_psi, phase_psi, log_psi_shift, )

	return DNN_psi.E_estimator.Eloc_real, DNN_psi.E_estimator.Eloc_imag



def compute_Eloc_ED(load_dir,ints_ket,log_kets,phase_kets,L,J2,return_ED_data=False,):

	N_batch=ints_ket.shape[0]

	params = yaml.load(open(load_dir+'config_params.yaml'),Loader=yaml.FullLoader)
	params['N_MC_points']=N_batch
	params['save_data']=False
	#params['J2']=J2
				

	DNN_psi_MC=VMC(load_dir,params_dict=params,train=False)



	N_sites=L*L
	if L==6:
		basis_type=np.uint64 
	elif L==4:
		basis_type=np.uint16
	N_symm=L*L*2*2*2
	

	static_list_offdiag=DNN_psi_MC.E_estimator._static_list_offdiag
	static_list_diag=DNN_psi_MC.E_estimator._static_list_diag
	_n_offdiag_terms=DNN_psi_MC.E_estimator._n_offdiag_terms


	
	_ints_bra_rep_holder=np.zeros((N_batch,),dtype=basis_type)
	_MEs_holder=np.zeros((N_batch,),dtype=np.float64)
	
	
	_MEs=np.zeros(N_batch*_n_offdiag_terms,dtype=np.float64)
	_spinstates_bra=np.zeros((N_batch*_n_offdiag_terms,N_sites*N_symm),dtype=np.int8)
	_ints_bra_rep=np.zeros((N_batch*_n_offdiag_terms,),dtype=basis_type)
	_ints_ket_ind=np.zeros(N_batch*_n_offdiag_terms,dtype=np.uint32)
	_n_per_term=np.zeros(_n_offdiag_terms,dtype=np.int32)


	_Eloc_cos=np.zeros(N_batch, dtype=np.float64)
	_Eloc_sin=np.zeros(N_batch, dtype=np.float64)

	Eloc_real=np.zeros_like(_Eloc_cos)
	Eloc_imag=np.zeros_like(_Eloc_cos)

	# find all s'

	nn=0
	for j,(opstr,indx,J) in enumerate(static_list_offdiag):

		_spinstates_bra_holder=np.zeros((N_batch,N_sites*N_symm),dtype=np.int8)
		_ints_ket_ind_holder=-np.ones((N_batch,),dtype=np.int32)

		indx=np.asarray(indx,dtype=np.int32)
		n = update_offdiag_ME(ints_ket,_ints_bra_rep_holder,_spinstates_bra_holder,_ints_ket_ind_holder,_MEs_holder,opstr,indx,J,N_symm,'DNN')
		
		_MEs[nn:nn+n]=_MEs_holder[_ints_ket_ind_holder[:n]]
		_ints_bra_rep[nn:nn+n]=_ints_bra_rep_holder[_ints_ket_ind_holder[:n]]
		_spinstates_bra[nn:nn+n]=_spinstates_bra_holder[_ints_ket_ind_holder[:n]]
		_ints_ket_ind[nn:nn+n]=_ints_ket_ind_holder[:n]

		
		_n_per_term[j]=n
		nn+=n


	#print('finished computing s_primes\n')

	# evaluate values of s' configs
	_ints_bra_rep_complete=compute_reps(_ints_bra_rep[:nn],L) # contains Z2 symmetry

	log_psi_bras, sign_psi_bras, mult_bras, p_bras, sign_psi_bras_J2_0 = extract_ED_data(_ints_bra_rep_complete,L,J2)
	phase_psi_bras=np.pi*0.5*(sign_psi_bras+1.0)
	phase_psi_bras_J2_0=np.pi*0.5*(sign_psi_bras_J2_0+1.0)


	phase_hist_ED, _ = np.histogram(np.cos(phase_psi_bras-phase_psi_bras_J2_0) ,bins=2,range=(-1.0,1.0), density=False, )
	print('s-primes: ED vs ED(J2=0):  T:F  :  {0:d}:{1:d}'.format(phase_hist_ED[1]  ,phase_hist_ED[0])  )

	# compute local energies
	_n_per_term=_n_per_term[_n_per_term>0]
	c_offdiag_sum(_Eloc_cos, _Eloc_sin, _n_per_term,_ints_ket_ind[:nn],_MEs[:nn],log_psi_bras,phase_psi_bras,log_kets)
		

	# n_cum=0;
	# for l in range(_n_per_term.shape[0]): #=0;l<Ns;l++){

	# 	n=_n_per_term[l];

	# 	for i in range(n): #=0;i<n;i++){

	# 		j=n_cum+i
		
	# 		aux=_MEs[j] * np.exp(log_psi_bras[j]-log_kets[_ints_ket_ind[j]]);

	# 		_Eloc_cos[_ints_ket_ind[j]] += aux * np.cos(phase_psi_bras[j]);
	# 		_Eloc_sin[_ints_ket_ind[j]] += aux * np.sin(phase_psi_bras[j]);
		
	# 	n_cum+=n;



	cos_phase_kets=np.cos(phase_kets)
	sin_phase_kets=np.sin(phase_kets)


	Eloc_real = _Eloc_cos*cos_phase_kets + _Eloc_sin*sin_phase_kets
	Eloc_imag = _Eloc_sin*cos_phase_kets - _Eloc_cos*sin_phase_kets


	# diag matrix elements, only real part
	for opstr,indx,J in static_list_diag:

		indx=np.asarray(indx,dtype=np.int32)
		update_diag_ME(ints_ket, Eloc_real,opstr,indx,J)

	if return_ED_data:
		return Eloc_real, Eloc_imag, log_psi_bras, phase_psi_bras, phase_psi_bras_J2_0
	else:
		return Eloc_real, Eloc_imag



def ED_results(load_dir,):


	# ints_ket=np.array([ints_ket[0]])
	# log_kets=np.array(log_kets[0])
	# phase_kets=np.array(phase_kets[0])

	N_batch=107

	params = yaml.load(open(load_dir+'config_params.yaml'),Loader=yaml.FullLoader)
	params['N_MC_points']=N_batch
	params['save_data']=False

	DNN_psi_MC=VMC(load_dir,params_dict=params,train=False)

	ref_states, index, inv_index, count = DNN_psi_MC.E_estimator.get_exact_kets()

	return DNN_psi_MC.E_estimator, ref_states.astype(DNN_psi_MC.E_estimator.basis_type), index, inv_index, count


