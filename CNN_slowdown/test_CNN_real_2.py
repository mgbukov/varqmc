import sys,os

from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, random, vmap

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
os.environ['XLA_FLAGS']='--xla_dump_to=/tmp/foo'




L=4
dtype=jnp.float64 
N_symm=2*2*2 # 



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
_,params = init_params(rng,input_shape)




@jit
def evaluate(params, batch):
    # reshaping required inside evaluate func because of per-sample gradients
    batch=batch.reshape(-1,1,L,L)

    # apply dense layer
    a = apply_layer(params, batch)

    # apply logcosh nonlinearity
    z=jnp.log(jnp.cosh(a))
    #z=relu(a) 
    
    return jnp.sum(z)


@jit
def compute_grad_log_psi(params,batch,):
	return vmap(partial( jit(grad(evaluate)),   params))(batch, )
	#return vmap(partial( grad(evaluate), params))(batch, )
	



###########################


# define data
N_points=300 
batch=np.ones((N_points,N_symm,L,L),dtype=dtype)

	
for _ in range (10):

    ti = time.time()
    d_psi = compute_grad_log_psi(params,batch)
    tf = time.time()
    
    print("gradients took {0:.4f} secs.".format(tf-ti))




