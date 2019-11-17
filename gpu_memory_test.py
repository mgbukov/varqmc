import sys,os

from mpi4py import MPI
comm=MPI.COMM_WORLD

os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="0"

os.environ["CUDA_VISIBLE_DEVICES"]="{0:d}".format(comm.Get_rank()) # device number


#import jax
#print('local devices:', jax.local_devices() )




from jax import device_put, jit, grad, vmap, random, ops, partial
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


print(comm.Get_rank(),os.environ["CUDA_VISIBLE_DEVICES"])

#exit()


#########################################
L=6

N_neurons=6
shapes=dict(layer_1 = [L**2, N_neurons], 
						#layer_2 = [12,4], 
			)
NN_shape_str='{0:d}'.format(L**2) + ''.join( '--{0:d}'.format(value[1]) for value in shapes.values() )


DNN=Neural_Net(0, shapes, 1, 'DNN', 'cpx', seed=seed )
evaluate_NN=jit(DNN.evaluate)





##### data
N_symm=2*2*2*L*L
N_sites=L*L
N_samples=30000 #73882

spinstates=np.ones((N_samples,N_symm,N_sites), dtype=np.int8)

from sys import getsizeof
print(getsizeof(spinstates), spinstates.nbytes)
#exit()

######
batch_size=N_samples
ti_tot=time.time()
spinstates=device_put(spinstates)
for i in range(20):
    ti=time.time()
    #for j in range(0,N_samples,batch_size):
    #    log_psi, phase_psi = evaluate_NN(DNN.params, spinstates[j:j+batch_size])
    
    #spinstates=np.ones((N_samples,N_symm,N_sites), dtype=np.int8)
    #spinstates=device_put(spinstates)
    log_psi, phase_psi = evaluate_NN(DNN.params, spinstates)
    tf=time.time()

    print(i, 'procces number:', comm.Get_rank(), 'DNN time: {0:0.4f}'.format(tf-ti))
tf_tot=time.time()

print()
print('procces number:', comm.Get_rank(), 'total time: {0:0.4f}'.format(tf_tot-ti_tot))

exit()


############


@jit
def cpx_cosh(Re_a,Im_a):
    # Cosh[a + I b] = Cos[b] Cosh[a] + I Sin[b] Sinh[a]
        
    Re = jnp.cos(Im_a)*jnp.cosh(Re_a)
    Im = jnp.sin(Im_a)*jnp.sinh(Re_a)

    return Re, Im


@jit
def cpx_log_real(Re, Im,):
    #a_fc_real = tf.log( tf.sqrt( (tf.cos(Im_Ws)*tf.cosh(Re_Ws))**2 + (tf.sin(Im_Ws)*tf.sinh(Re_Ws))**2 )  )
    return 0.5*jnp.log(Re**2+Im**2)
  
@jit
def cpx_log_imag(Re, Im,):
    #a_fc_imag = tf.atan( tf.tan(Im_Ws)*tf.tanh(Re_Ws) )
    return jnp.arctan2(Im,Re)


@jit
def logcosh_cpx(x):
    x_real, x_imag = x
    Re, Im  = cpx_cosh(x_real, x_imag)
    Re_z = cpx_log_real(Re, Im, ) 
    Im_z = cpx_log_imag(Re, Im, )
    return Re_z, Im_z


input_shape=(-1,N_sites)
reduce_shape=(-1,N_symm,N_neurons)
output_shape=(-1,N_neurons)


@jit
def evaluate(params, batch):

    # reshaping required inside evaluate func because of per-sample gradients
    batch=batch.reshape(input_shape)

    # apply dense layer
    Re_Ws, Im_Ws = DNN.apply_layer(params,batch)
    # apply logcosh nonlinearity
    Re_z, Im_z = logcosh_cpx((Re_Ws, Im_Ws))

    # symmetrize
    log_psi   = jnp.sum(Re_z.reshape(reduce_shape,order='C'), axis=[1,])
    phase_psi = jnp.sum(Im_z.reshape(reduce_shape,order='C'), axis=[1,])
    # 
    log_psi   = jnp.sum(  log_psi.reshape(output_shape), axis=[1,])
    phase_psi = jnp.sum(phase_psi.reshape(output_shape), axis=[1,])
    
    return log_psi, phase_psi




ti=time.time()
log_psi, phase_psi = evaluate(DNN.params, spinstates)
tf=time.time()

print('DNN time 2: {0:0.4f}'.format(tf-ti))






