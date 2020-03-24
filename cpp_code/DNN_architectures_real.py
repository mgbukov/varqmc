from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, random, device_put 
from jax.experimental.stax import BatchNorm

from mpi4py import MPI

import functools
import itertools
from jax import lax, random
from jax import ops, disable_jit
import jax.numpy as jnp
import numpy as np

from jax.nn import (relu, log_softmax, softmax, softplus, sigmoid, elu,
                    leaky_relu, selu, gelu, normalize)
from jax.nn.initializers import glorot_normal, normal, ones, zeros

#from DNN_architectures_cpx import elementwise


# aliases for backwards compatibility
#glorot = glorot_normal
#randn = normal
#logsoftmax = log_softmax

# import numpy as np
# seed=1
# np.random.seed(seed)
# np.random.RandomState(seed)
# rng = random.PRNGKey(seed)



def GeneralDense(W_shape, ignore_b=False, init_value_W=1E-2, init_value_b=1E-2):

    def init_fun(rng,input_shape):

         
        output_shape=(input_shape[0],W_shape[1])
        W = random.uniform(rng,shape=W_shape, minval=-init_value_W, maxval=+init_value_W)
        
        if not ignore_b:
            rng, k1 = random.split(rng)
            b = random.uniform(k1,shape=(output_shape[1],), minval=-init_value_b, maxval=+init_value_b)
            params=(W,b,)
        
        else:
            params=(W,)
        
        return output_shape, params

    def apply_fun(params,inputs, **kwargs):
        z = jnp.dot(inputs,params[0])
        if not ignore_b:
            # add bias
            z += params[1]       
        return z

    return init_fun, apply_fun





# nonlinearities


@jit
def logcosh(x):
    return jnp.log(jnp.cosh(x))


##############################




'''
def Norm_real(center=True, scale=True, a_init=ones, b_init=zeros, dtype=np.float64):
    """Layer construction function for a batch normalization layer."""
    _a_init = lambda rng, shape: a_init(rng, shape, dtype) if scale else ()
    _b_init = lambda rng, shape: b_init(rng, shape, dtype) if center else ()
    
    def init_fun(rng, input_shape):
        shape = (1,)
        k1, k2 = random.split(rng)
        k2, k3 = random.split(k2)

        #a = _a_init(k1, shape)
        b = _b_init(k2, shape) - 0.01
        #c = _a_init(k3, shape)
        
        output_shape=(input_shape[0],1)
        
        return output_shape, (b,)

    def apply_fun(params, x, a=4.0, b=+0.0, c=1.0, **kwargs):
        b,   = params
        x_real, x_imag = x


        #print('MEAN', np.mean(x_real), np.std(x_real), np.min(x_real), np.max(x_real), x_real.shape)
        #exit()

        #x_real= c*(jnp.where(x_real < b, a*(x_real-b), -jnp.expm1(-a*(x_real-b)), ) + a*b)

        x_real=a*jnp.tanh((x_real-b)/a) #+ b
        
        #print('MEAN post', np.mean(x_real), np.std(x_real), np.min(x_real), np.max(x_real), x_real.shape)
        #print()

        #print(x_real)
        #exit()

        return (x_real, x_imag)
        
    return init_fun, apply_fun
'''




def Regularization(output_layer_shape,center=True, scale=True, a_init=ones, b_init=zeros, dtype=np.float64):
    """Layer construction function for a batch normalization layer."""
    _a_init = lambda rng, shape: a_init(rng, shape, dtype) if scale else ()
    _b_init = lambda rng, shape: b_init(rng, shape, dtype) if center else ()


    def init_fun(rng, input_shape):
        
        k1, k2 = random.split(rng)
        k2, k3 = random.split(k2)
        k3, k4 = random.split(k3)

        #a = _a_init(k1, shape)
        b_shape = (1,)
        b = _b_init(k2, b_shape) - 0.0
        
        # init_value_W=1E-2
        # W_shape=output_layer_shape+(1,) 
        # W1 = 1.0 + random.uniform(k3,shape=output_layer_shape, minval=-init_value_W, maxval=+init_value_W)    
        # W2 = 1.0 + random.uniform(k4,shape=output_layer_shape, minval=-init_value_W, maxval=+init_value_W)    

        output_shape=(input_shape[0],1)
        
        return output_shape, (b,)

    def apply_fun(params, x, reduce_shape, output_shape, **kwargs):
        b,   = params
        
        # symmetrize
        # 1/(N/p) = p/N : p different terms in sum left
        # uncorrelated: 1/\sqrt(p) 
        # correlated: 1/p
        log_psi   = jnp.sum(x.reshape(reduce_shape,order='C'), axis=[1,])#/jnp.sqrt(128.0)
        
        # sum over hidden neurons
        log_psi   = jnp.sum(  log_psi.reshape(output_shape), axis=[1,])   

        # regularize output
        a=8.0
        log_psi=a*jnp.tanh((log_psi-b)/a) + b
        
        
        return log_psi
        
    return init_fun, apply_fun




def Phase_arg(output_layer_shape,center=True, scale=True, a_init=ones, b_init=zeros, dtype=np.float64):
    """Layer construction function for a batch normalization layer."""
    _a_init = lambda rng, shape: a_init(rng, shape, dtype) if scale else ()
    _b_init = lambda rng, shape: b_init(rng, shape, dtype) if center else ()


    def init_fun(rng, input_shape):
        
        k1, k2 = random.split(rng)
       
        b_shape = (1,)
        b = _b_init(k2, b_shape) 
        
        # b_shape = output_layer_shape #
        # b = _a_init(k2, b_shape)
        
        output_shape=(input_shape[0],1)
        
        return output_shape, ()

    def apply_fun(params, x, reduce_shape, output_shape, **kwargs):
        #b,   = params

        #print(x[-1,...].sum(), x[-2,...].sum(), x[-16,...].sum())

        #phase_psi = jnp.exp(1j*x)
        phase_psi=x 

        # symmetrize
        phase_psi   = jnp.sum(phase_psi.reshape(reduce_shape,order='C'), axis=[1,])#/jnp.sqrt(128.0)
        
        # sum over hidden neurons
        phase_psi   = jnp.sum(  phase_psi.reshape(output_shape), axis=[1,])
        #phase_psi = jnp.dot(phase_psi,b)
        
        #print('THERE', phase_psi[-16], phase_psi[-1])
        

        # regularize output
        #phase_psi=jnp.angle(phase_psi)
        #phase_psi=jnp.pi*jnp.tanh(phase_psi-b)

        #print(phase_psi)
        #exit()
        
        return phase_psi
        
    return init_fun, apply_fun


###############





