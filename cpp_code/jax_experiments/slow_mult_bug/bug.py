from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, random, device_put 
from jax.experimental.stax import BatchNorm


import functools
import itertools
from jax import lax, random
from jax import ops, disable_jit
import jax.numpy as jnp
import numpy as np

from jax.experimental.stax import serial, elementwise

from jax.nn import (relu, log_softmax, softmax, softplus, sigmoid, elu,
                    leaky_relu, selu, gelu, normalize)
from jax.nn.initializers import glorot_normal, normal, ones, zeros


import time

seed=1
np.random.seed(seed)
np.random.RandomState(seed)
rng = random.PRNGKey(seed)


def GeneralDense(W_shape, ignore_b=False, init_value_W=1E-2, init_value_b=1E-2):
    
    norm=jnp.sqrt(W_shape[0]+W_shape[1])

    def init_fun(rng,input_shape):        
        output_shape=(input_shape[0],W_shape[1])
        W = random.uniform(rng,shape=W_shape, minval=-init_value_W, maxval=+init_value_W)
        #W/=norm # see apply func
        if not ignore_b:
            rng, k1 = random.split(rng)
            b = random.uniform(k1,shape=(output_shape[1],), minval=-init_value_b, maxval=+init_value_b)
            params=(W,b,)
        
        else:
            params=(W,)
        
        return output_shape, params

    def apply_fun(params,inputs, **kwargs):
        z = jnp.dot(inputs,params[0])/norm
        if not ignore_b:
            # add bias
            z += params[1]       
        return z

    return init_fun, apply_fun


@jit
def logcosh(x):
    return jnp.log(jnp.cosh(x))


N=1
N_feat=16

W_shape_1=(N_feat,12)
W_shape_2=(12,8)
input_shape=(1,N_feat)


init_fun, apply_fun = serial(
							GeneralDense(W_shape_1, ignore_b=True, ),
							elementwise(logcosh),
							GeneralDense(W_shape_2, ignore_b=True, ),
							elementwise(logcosh)
							) 

output_shape, params = init_fun(rng,input_shape)


Apply_fun=jit(apply_fun)

vec = [-1, 1]
	

ti=time.time()
for _ in range(10000):
	
	inputs = 2*np.random.randint(2,size=(N,N_feat))-1


	Apply_fun(params,inputs).block_until_ready()
tf=time.time()

print(tf-ti)



