import os,sys
from make_data_file import create_params_file

sys.path.append("..")


from VMC_class import VMC
import yaml 

from jax.config import config
config.update("jax_enable_x64", True)
from jax.experimental import optimizers
from jax import jit, grad, random
import jax.numpy as jnp

import numpy as np

from cpp_code import integer_to_spinstate
from energy_lib import Energy_estimator

import time
import itertools

# export KMP_DUPLICATE_LIB_OK=TRUE
# export OMP_NUM_THREADS=4
# mpiexec -n 4 python main.py --test



params = yaml.load(open('config_params_supervised.yaml'),Loader=yaml.FullLoader)
#data_dir=create_params_file(params)

data_dir=None

# create NN
DNN_psi=VMC(data_dir,params_dict=params,train=False)



##########################################################


#
def loss_log(params, batch):
	spin_configs, log_psi_ED, sign_psi_ED , p_ED = batch
	log_psi_predicted = DNN_psi.DNN.evaluate_log(params, spin_configs,)
	return jnp.mean(jnp.abs(log_psi_ED - log_psi_predicted)**2 )

def loss_phase(params, batch):
	spin_configs, log_psi_ED, sign_psi_ED , p_ED = batch
	phase_psi_predicted = DNN_psi.DNN.evaluate_phase(params, spin_configs,)
	return jnp.mean(jnp.abs( np.pi*(sign_psi_ED+1)/2 - phase_psi_predicted)**2 )


def accuracy(params, batch):
	spin_configs, log_psi_ED, sign_psi_ED , p_ED = batch
	log_psi_predicted = DNN_psi.DNN.evaluate_log(params, spin_configs,)
	phase_psi_predicted = DNN_psi.DNN.evaluate_phase(params, spin_configs,)
	return jnp.mean(p_ED * jnp.abs(log_psi_ED - log_psi_predicted)**2 ) + jnp.mean(p_ED * jnp.abs(np.pi*(sign_psi_ED+1)/2 - phase_psi_predicted)**2 )


def compute_energy(DNN_psi,spin_int_states,log_psi,phase_psi, mult ,log_psi_shift=0.0):

	N_MC_points=log_psi.shape[0]
	N_batch=N_MC_points

	DNN_psi.E_estimator=Energy_estimator(DNN_psi.comm,DNN_psi.J2,N_MC_points,N_batch,DNN_psi.L,DNN_psi.DNN.N_symm,DNN_psi.DNN.NN_type,DNN_psi.sign,) # contains all of the physics
	DNN_psi.E_estimator.init_global_params(N_MC_points,1)
		
	ti=time.time()
	DNN_psi.E_estimator.compute_local_energy(DNN_psi.evaluate_NN,DNN_psi.DNN,DNN_psi.DNN.params,spin_int_states,log_psi,phase_psi,log_psi_shift,DNN_psi.minibatch_size)
	
	Eloc_str="total local energy calculation took {0:.4f} secs.\n".format(time.time()-ti)
	print(Eloc_str)


	mod_kets=np.exp(log_psi)
	norm=np.sqrt(np.sum(mult*mod_kets**2))
	psi_NN = mod_kets*np.exp(+1j*phase_psi)/norm
	abs_psi_2=mult*np.abs(psi_NN)**2

	DNN_psi.Eloc_params_dict=dict(abs_psi_2=abs_psi_2,)

	Eloc_mean_g, Eloc_var_g, E_diff_real, E_diff_imag = DNN_psi.E_estimator.process_local_energies(mode=DNN_psi.mode,Eloc_params_dict=DNN_psi.Eloc_params_dict)
	Eloc_std_g=np.sqrt(Eloc_var_g)
	#E_MC_std_g=Eloc_std_g/np.sqrt(N_MC_points)

	return Eloc_mean_g, Eloc_std_g



def compute_overlap(DNN_psi, psi_NN, psi_ED, mult):

	DNN_psi.mode='exact'

	print('norm psi_NN', np.abs( np.sum(mult*psi_NN.conj()*psi_NN) )**2)
	print('norm psi_ED', np.abs( np.sum(mult*psi_ED.conj()*psi_ED) )**2)

	overlap=np.abs( np.sum(mult*psi_NN.conj()*psi_ED) )**2
	DNN_psi.Eloc_params_dict['overlap']=overlap
	return overlap







if __name__ == "__main__":
	rng = random.PRNGKey(0)

	num_epochs = 7000
	batch_size = 5000


	# load data
	path_to_data=os.path.expanduser('~') + '/Google_Drive/frustration_from_RBM/ED_data/eff_datasets/'
	save_name='effdata-GS_J1-J2_Lx={0:d}_Ly={1:d}_J1=1.0000_J2={2:0.4f}.txt'.format(DNN_psi.L,DNN_psi.L,DNN_psi.J2)

	datafile_id=path_to_data+save_name
	spin_int_states = np.loadtxt(datafile_id, delimiter=' ', usecols=0).astype(DNN_psi.E_estimator.basis_type)
	log_psi = np.loadtxt(datafile_id, delimiter=' ', usecols=1)
	sign_psi = np.loadtxt(datafile_id, delimiter=' ', usecols=4)
	phase_psi=np.pi*(sign_psi+1)/2
	mult = np.loadtxt(datafile_id, delimiter=' ', usecols=5)
	
	psi_ED = np.exp(log_psi + 1j*phase_psi)
	norm = np.sqrt( np.sum( mult*np.exp(2.0*log_psi) ) )
	psi_ED/=norm

	p_ED = mult*np.exp(2.0*log_psi)


	# determine # of batches
	num_train = spin_int_states.shape[0]
	num_complete_batches, leftover = divmod(num_train, batch_size)
	num_batches = num_complete_batches + bool(leftover)


	# compute spin configs
	spin_configs=np.zeros((num_train*DNN_psi.MC_tool.N_features,), dtype=np.int8)
	integer_to_spinstate(spin_int_states, spin_configs, DNN_psi.DNN.N_features, NN_type=DNN_psi.DNN.NN_type)

	all_data = (spin_configs, log_psi, sign_psi, p_ED, )


	log_psi_NN = DNN_psi.DNN.evaluate_log(DNN_psi.DNN.params, spin_configs,)
	phase_NN = DNN_psi.DNN.evaluate_phase(DNN_psi.DNN.params, spin_configs,)



	def data_stream():
		rng = np.random.RandomState(0)
		while True:
			perm = rng.permutation(num_train)
			for i in range(num_batches):
				batch_idx = perm[i * batch_size:(i + 1) * batch_size]
				yield spin_configs.reshape(num_train,-1)[batch_idx], log_psi[batch_idx], sign_psi[batch_idx], p_ED[batch_idx]
	
	batches = data_stream()


	# optimizer
	opt_init_log, opt_update_log, get_params_log = optimizers.adam(step_size=1E-4, b1=0.9, b2=0.99, eps=1e-08)
	opt_init_phase, opt_update_phase, get_params_phase = optimizers.adam(step_size=1E-3, b1=0.9, b2=0.99, eps=1e-08)

	@jit
	def update_log(i, opt_state, batch):
		params = get_params_log(opt_state)
		return opt_update_log(i, grad(loss_log)(params, batch), opt_state)

	@jit
	def update_phase(i, opt_state, batch):
		params = get_params_phase(opt_state)
		return opt_update_phase(i, grad(loss_phase)(params, batch), opt_state)


	opt_state_log = opt_init_log(DNN_psi.DNN.params)
	opt_state_phase = opt_init_phase(DNN_psi.DNN.params)

	itercount = itertools.count()

	print("\nStarting training...")
	for epoch in range(num_epochs):

		start_time = time.time()

		for _ in range(num_batches):
			batch=next(batches)
			it=next(itercount)
			opt_state_log = update_log(it, opt_state_log, batch)
			opt_state_phase = update_phase(it, opt_state_phase, batch)

			print("Batch {}".format(_), )
			
		# update parameters
		DNN_psi.DNN.params = (get_params_log(opt_state_log)[0], get_params_phase(opt_state_phase)[1], )

		batch_loss_log = loss_log(DNN_psi.DNN.params, batch)
		batch_loss_phase = loss_phase(DNN_psi.DNN.params, batch)
		batch_acc = accuracy(DNN_psi.DNN.params, batch)
		print("\n batch losses {0:0.14f}, {1:0.14f}, w-loss {2:0.14f}\n".format(batch_loss_log, batch_loss_phase, batch_acc))

		
		#exit()
		epoch_time = time.time() - start_time

		
		print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))


	# # test
	# final_loss_log = loss_log(DNN_psi.DNN.params, all_data)
	# final_loss_phase = loss_phase(DNN_psi.DNN.params, all_data)
	# final_acc = accuracy(DNN_psi.DNN.params, all_data)
	# print("\nFinal losses {0:0.14f}, {1:0.14f}, w-loss {2:0.14f}\n".format(final_loss_log, final_loss_phase, final_acc))


	# evaluate DNN
	log_psi_NN = DNN_psi.DNN.evaluate_log(DNN_psi.DNN.params, spin_configs,)._value
	phase_NN = DNN_psi.DNN.evaluate_phase(DNN_psi.DNN.params, spin_configs,)._value

	print(np.max(np.abs(log_psi_NN) - np.abs(log_psi) ))
	print(np.max(np.abs(phase_NN) - np.abs(phase_psi) ))
	#exit()
	

	# evaluate energy of DNN
	Eloc_mean, Eloc_std = compute_energy(DNN_psi, spin_int_states,log_psi_NN,phase_NN, mult,)
	print("E, E_std = ", Eloc_mean, Eloc_std )


	# compute network state
	psi_NN = np.exp(log_psi_NN+1j*phase_NN)
	norm = np.sqrt( np.sum( mult*np.exp(2.0*log_psi_NN) ) )
	psi_NN/=norm


	overlap = compute_overlap(DNN_psi, psi_NN, psi_ED, mult)


	print(overlap)






