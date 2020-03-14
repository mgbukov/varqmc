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

import time
import itertools

# export KMP_DUPLICATE_LIB_OK=TRUE
# export OMP_NUM_THREADS=4
# mpiexec -n 4 python main.py --test



params = yaml.load(open('config_params_supervised.yaml'),Loader=yaml.FullLoader)
data_dir=create_params_file(params)


# create NN
DNN_psi=VMC(data_dir,train=False)




#
def loss(params, batch):
	spin_configs, log_psi_ED, sign_psi_ED , p_ED = batch
	log_psi_predicted = DNN_psi.DNN.evaluate_log(params, spin_configs,)
	return jnp.mean(jnp.abs(log_psi_ED - log_psi_predicted)**2 )


def accuracy(params, batch):
	spin_configs, log_psi_ED, sign_psi_ED , p_ED = batch
	log_psi_predicted = DNN_psi.DNN.evaluate_log(params, spin_configs,)
	return jnp.mean(p_ED * jnp.abs(log_psi_ED - log_psi_predicted)**2 )



if __name__ == "__main__":
	rng = random.PRNGKey(0)

	num_epochs = 100
	batch_size = 5000


	# load data
	path_to_data=os.path.expanduser('~') + '/Google_Drive/frustration_from_RBM/ED_data/eff_datasets/'
	save_name='effdata-GS_J1-J2_Lx={0:d}_Ly={1:d}_J1=1.0000_J2={2:0.4f}.txt'.format(DNN_psi.L,DNN_psi.L,DNN_psi.J2)

	datafile_id=path_to_data+save_name
	spin_int_states = np.loadtxt(datafile_id, delimiter=' ', usecols=0).astype(DNN_psi.E_estimator.basis_type)
	log_psi = np.loadtxt(datafile_id, delimiter=' ', usecols=1)
	sign_psi = np.loadtxt(datafile_id, delimiter=' ', usecols=4)
	mult = np.loadtxt(datafile_id, delimiter=' ', usecols=5)

	p_ED = mult*np.exp(2.0*log_psi)


	# determine # of batches
	num_train = spin_int_states.shape[0]
	num_complete_batches, leftover = divmod(num_train, batch_size)
	num_batches = num_complete_batches + bool(leftover)


	# compute spin configs
	spin_configs=np.zeros((num_train*DNN_psi.MC_tool.N_features,), dtype=np.int8)
	integer_to_spinstate(spin_int_states, spin_configs, DNN_psi.DNN.N_features, NN_type=DNN_psi.DNN.NN_type)

	all_data = (spin_configs, log_psi, sign_psi, p_ED, )



	def data_stream():
		rng = np.random.RandomState(0)
		while True:
			perm = rng.permutation(num_train)
			for i in range(num_batches):
				batch_idx = perm[i * batch_size:(i + 1) * batch_size]
				yield spin_configs.reshape(num_train,-1)[batch_idx], log_psi[batch_idx], sign_psi[batch_idx], p_ED[batch_idx]
	
	batches = data_stream()


	# optimizer
	opt_init, opt_update, get_params = optimizers.adam(step_size=1E-2, b1=0.9, b2=0.99, eps=1e-08)

	@jit
	def update(i, opt_state, batch):
		params = get_params(opt_state)
		return opt_update(i, grad(loss)(params, batch), opt_state)

	opt_state = opt_init(DNN_psi.DNN.params)
	itercount = itertools.count()

	print("\nStarting training...")
	for epoch in range(num_epochs):

		start_time = time.time()

		for _ in range(num_batches):
			batch=next(batches)
			opt_state = update(next(itercount), opt_state, batch)

			print("Batch {}".format(_))
			
		# batch_loss = loss(DNN_psi.DNN.params, batch)
		# batch_acc = accuracy(DNN_psi.DNN.params, batch)
		# print("Batch # w-loss {0:0.14f}, loss {1:0.14f}\n".format(batch_loss, batch_acc))

		# update parameters
		DNN_psi.DNN.params = get_params(opt_state)

		
		#exit()
		epoch_time = time.time() - start_time

		
		print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))


	# test
	final_loss = loss(DNN_psi.DNN.params, all_data)
	final_acc = accuracy(DNN_psi.DNN.params, all_data)
	print("\nFinal w-loss {0:0.14f}, loss {1:0.14f}\n".format(final_loss, final_acc))


		
