import sys,os
#os.environ['XLA_FLAGS']='--xla_dump_to=/tmp/CNN_logfiles'


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




L=4
N_symm=2*2*2 #s
dtype=jnp.float64 #jnp.complex128


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
rng_1, rng_2 = random.split(rng)
_, params_real = init_params(rng_1,input_shape)
_, params_imag = init_params(rng_2,input_shape)


params=[params_real,params_imag,]




#@jit
def evaluate(params, batch):
	# reshaping required inside evaluate func because of per-sample gradients
	batch=batch.reshape(-1,1,L,L)

	# apply dense layer
	Re_Ws = apply_layer(params[0], batch)
	Im_Ws = apply_layer(params[1], batch) 

	# apply logcosh nonlinearity
	Re, Im  = cpx_cosh(Re_Ws, Im_Ws)
	Re_z = cpx_log_real(Re, Im, ) 
	
	return jnp.sum(Re_z)


#@jit
def compute_grad_log_psi(params,batch,):
	#return vmap(partial( jit(grad(evaluate)),   params))(batch, )
	return vmap(partial( grad(evaluate), params))(batch, )
	



###########################

N_points=500

# define data
batch=np.ones((N_points,N_symm,L,L),dtype=dtype)

	
for _ in range (10):

	ti = time.time()
	d_psi = compute_grad_log_psi(params,batch)
	tf = time.time()
    
    print("gradients took {0:.4f} secs.".format(tf-ti))





