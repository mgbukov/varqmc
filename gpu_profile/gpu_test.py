import sys,os

os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="0" # no XLA memory preallocation


#os.environ["CUDA_VISIBLE_DEVICES"]="{0:d}".format(comm.Get_rank()) # device number
#print("process {0:d} runs on GPU device {1:d}".format(comm.Get_rank(),int(os.environ["CUDA_VISIBLE_DEVICES"])))


from jax import device_put, jit, grad, vmap, random, ops, partial, lax
from jax.config import config
config.update("jax_enable_x64", True)
from jax.experimental import optimizers

import jax.numpy as jnp
import numpy as np

from jax import jit, grad, random, device_put 
from jax.experimental.stax import elementwise, BatchNorm, serial

import functools
import itertools
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np

from jax.nn import (relu, log_softmax, softmax, softplus, sigmoid, elu,
                    leaky_relu, selu, gelu, normalize)
from jax.nn.initializers import glorot_normal, normal, ones, zeros


import time
np.set_printoptions(threshold=np.inf)

seed=0
np.random.seed(seed)
np.random.RandomState(seed)
rng = random.PRNGKey(seed)

###################

L=6
N_sites=L*L
N_symm=L*L*2*2*2 # no Z symmetry
N_neurons=6
shapes=dict(layer_1 = [L**2, N_neurons])


################################################################################################


def GeneralDense_cpx(W_shape, ignore_b=False):

    def init_fun(rng,input_shape):

        init_value_W=1E-3 #1E-1

        rng_real, rng_imag = random.split(rng)
        
        output_shape=(input_shape[0],W_shape[1])

        W_real = random.uniform(rng_real,shape=W_shape, minval=-init_value_W, maxval=+init_value_W)
        W_imag = random.uniform(rng_imag,shape=W_shape, minval=-init_value_W, maxval=+init_value_W)

        if not ignore_b:

            init_value_b=1E-2

            rng_real, k1 = random.split(rng_real)
            rng_imag, k2 = random.split(rng_imag)

            b_real = random.uniform(k1,shape=(output_shape[1],), minval=-init_value_b, maxval=+init_value_b)
            b_imag = random.uniform(k2,shape=(output_shape[1],), minval=-init_value_b, maxval=+init_value_b)
            
            params=((W_real,b_real),(W_imag,b_imag))
        
        else:
            params=((W_real,),(W_imag,))
        
        return output_shape, params

    def apply_fun(params,inputs, **kwargs):
        #return jnp.einsum('ij,lj->li',params, inputs)

        # read-off params
        W_real=params[0][0]
        W_imag=params[1][0]

        if isinstance(inputs, tuple):
            inputs_real, inputs_imag = inputs
        else:
            inputs_real = inputs
            inputs_imag = None

        z_real = jnp.dot(inputs_real,W_real) 
        z_imag = jnp.dot(inputs_real,W_imag)


        if inputs_imag is not None:
            z_real -= jnp.dot(inputs_imag,W_imag)
            z_imag += jnp.dot(inputs_imag,W_real)

        if not ignore_b:
            # add bias
            z_real += params[0][1]
            z_imag += params[1][1]
       
        return z_real, z_imag


    return init_fun, apply_fun

        

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



def evaluate(params, batch):

    # reshaping required inside evaluate func because of per-sample gradients
    batch=batch.reshape(input_shape)

    # apply dense layer
    Re_Ws, Im_Ws = apply_layer(params,batch)
    # apply logcosh nonlinearity
    Re_z, Im_z = logcosh_cpx((Re_Ws, Im_Ws))

    # symmetrize
    log_psi   = jnp.sum(Re_z.reshape(reduce_shape,order='C'), axis=[1,])
    phase_psi = jnp.sum(Im_z.reshape(reduce_shape,order='C'), axis=[1,])
    # 
    log_psi   = jnp.sum(  log_psi.reshape(output_shape), axis=[1,])
    phase_psi = jnp.sum(phase_psi.reshape(output_shape), axis=[1,])
    
    return log_psi, phase_psi


evaluate_NN=jit(evaluate)



########################################################################

# define DNN
init_params, apply_layer = serial(
                                        GeneralDense_cpx(shapes['layer_1'], ignore_b=True), 
                                        #LogCosh_cpx,
                                        #GeneralDense_cpx(shapes['layer_2'], ignore_b=False), 
                                    )
_, params = init_params(rng,(1,N_sites))


# define data
N_samples=30000

batch_size=100
num_complete_batches, leftover = divmod(N_samples, batch_size)
num_batches = num_complete_batches + bool(leftover)



#spinstates=np.ones((N_samples,N_symm,N_sites), dtype=np.int8)
#print(spinstates.nbytes)


def data_stream():
    #rng = np.random.RandomState(0)
    while True:
        #perm = rng.permutation(N_samples)
        for i in range(num_batches):
            #batch_idx = perm[i * batch_size : (i + 1) * batch_size]
            batch_idx = np.arange(i*batch_size, min(N_samples, (i+1)*batch_size), 1)
            yield spinstates[batch_idx], batch_idx
batches = data_stream()





log_psi=np.zeros(N_samples,)
phase_psi=np.zeros(N_samples,)

N_epochs=10
ti_tot=time.time()
#spinstates=device_put(spinstates)
for i in range(N_epochs):
    ti=time.time()

    spinstates=np.random.uniform(size=(N_samples,N_symm,N_sites))

    for j in range(num_batches):
        batch, batch_idx = next(batches)
        log_psi[batch_idx], phase_psi[batch_idx] = evaluate_NN(params, batch)


    tf=time.time()
    print(i, 'batch time: {0:0.4f}'.format(tf-ti))
tf_tot=time.time()


print()
print('total batch time: {0:0.4f}'.format(tf_tot-ti_tot))
print()


ti_tot=time.time()
for i in range(N_epochs):
    ti=time.time()

    spinstates=np.random.uniform(size=(N_samples,N_symm,N_sites))
    log_psi, phase_psi = vmap(partial(evaluate_NN, params))(spinstates, )
    log_psi=log_psi.squeeze()
    phase_psi=phase_psi.squeeze()


    tf=time.time()
    print(i, 'per-sampe time: {0:0.4f}'.format(tf-ti))
tf_tot=time.time()


print()
print('total per-sample time: {0:0.4f}'.format(tf_tot-ti_tot))
print()



ti_tot=time.time()
for i in range(N_epochs):
    ti=time.time()

    spinstates=np.random.uniform(size=(N_samples,N_symm,N_sites))
    log_psi, phase_psi = evaluate_NN(params, spinstates)

    tf=time.time()
    print(i, 'bunch time: {0:0.4f}'.format(tf-ti))
tf_tot=time.time()


print()
print('total bunch time: {0:0.4f}'.format(tf_tot-ti_tot))
print()

