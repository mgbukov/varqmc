import sys,os

os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="0"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import jax
print('local devices:', jax.local_devices() )


from jax import jit, grad, vmap, random, ops, partial
from jax.config import config
config.update("jax_enable_x64", True)
from jax.experimental import optimizers

import jax.numpy as jnp
import numpy as np


from cpp_code import Neural_Net


import time
np.set_printoptions(threshold=np.inf)

seed=0
np.random.seed(seed)
np.random.RandomState(seed)
rng = random.PRNGKey(seed)


#########################################
L=6



shapes=dict(layer_1 = [L**2, 6], 
						#layer_2 = [12,4], 
			)
NN_shape_str='{0:d}'.format(L**2) + ''.join( '--{0:d}'.format(value[1]) for value in shapes.values() )


DNN=Neural_Net(0, shapes, 1, 'DNN', 'cpx', seed=seed )
evaluate_NN=jit(DNN.evaluate)



##### data
N_symm=2*2*2*L*L
N_sites=L*L
N_samples=1000

spinstates=np.ones((N_samples,N_symm,N_sites), dtype=np.int8)


ti=time.time()
log_psi, phase_psi = evaluate_NN(DNN.params, spinstates)
tf=time.time()

print('DNN time: {0:0.4f}'.format(tf-ti))






