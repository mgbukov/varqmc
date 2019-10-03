import sys,os
#os.environ['XLA_FLAGS']='--xla_dump_to=/tmp/CNN_logfiles'


from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, random, device_put, vmap, ops, partial


import numpy as np
from functools import partial

seed=1
np.random.seed(seed)
np.random.RandomState(seed)
rng = random.PRNGKey(seed)


from jax.experimental.stax import GeneralConv, relu
from jax.nn.initializers import glorot_normal

import time

####################################################################



@jit
def log_cosh(a):
    return jnp.log(jnp.cosh(a))



L=4
dtype=jnp.float64 #jnp.complex128
N_sites=L*L


N_symm=2*2*2 # no Z, Tx, Ty symemtry

dimension_numbers=('NCHW', 'OIHW', 'NCHW') # default
out_chan=1
filter_shape=(2,2)
strides=(1,1)

input_shape=np.array((1,1,L,L),dtype=np.int) # NCHW input format

lhs_spec, rhs_spec, out_spec = dimension_numbers
W_init=glorot_normal(rhs_spec.index('I'), rhs_spec.index('O'))


W_init = partial(W_init, dtype=dtype)
init_params, apply_layer = GeneralConv(dimension_numbers, out_chan, filter_shape, strides=strides, padding='VALID', W_init=W_init)
       

# initialize parameters
params = init_params(rng,input_shape)[1]




@jit
def evaluate(params, batch):
    # reshaping required inside evaluate func because of per-sample gradients
    batch=jnp.reshape(batch,(-1,1,L,L))

    # apply dense layer
    Ws = apply_layer(params, batch)
    
    # apply logcosh nonlinearity
    z = log_cosh(Ws)
    #z = relu(Ws) 

    return jnp.sum(z)


@jit
def compute_grad_log_psi(params,batch,):
	return vmap(partial( jit(grad(evaluate)),   params))(batch, )
	#return vmap(partial( grad(evaluate), params))(batch, )
	



###########################

N_MC_points=100 # number of points

# define data
batch=np.ones((N_MC_points,N_symm,L,L),dtype=dtype)


#print(evaluate(params,batch))

	
# compute gradients



for _ in range (10):

    ti = time.time()
    
    d_psi = compute_grad_log_psi(params,batch)
    #print(d_psi[0][0].shape,d_psi[0][1].shape,d_psi[1][0].shape,d_psi[1][1].shape)

    print("gradients took {0:.4f} secs.".format(time.time()-ti))





