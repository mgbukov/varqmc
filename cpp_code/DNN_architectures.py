
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, random, device_put 
#from jax.experimental.stax import relu, BatchNorm

import functools
import itertools
from jax import lax
from jax import random
import jax.numpy as jnp

from jax.nn import (relu, log_softmax, softmax, softplus, sigmoid, elu,
                    leaky_relu, selu, gelu, normalize)
from jax.nn.initializers import glorot_normal, normal, ones, zeros

# aliases for backwards compatibility
glorot = glorot_normal
randn = normal
logsoftmax = log_softmax




def periodic_padding(inputs,filter_shape,strides):
    n_x=filter_shape[0]-strides[0]
    n_y=filter_shape[1]-strides[1]
    return jnp.pad(inputs, ((0,0),(0,0),(0,n_x),(0,n_y)), mode='wrap')




def GeneralDeep(W_shape, ignore_b=False):

    def init_fun(rng,input_shape):

        init_value_W=1E-1
        init_value_b=1E-1 

        if not ignore_b:
            W = random.uniform(rng,shape=W_shape, minval=-init_value_W, maxval=+init_value_W)
            b = random.uniform(rng,shape=(W_shape[0],), minval=-init_value_b, maxval=+init_value_b)
            params=(W,b)
        else:
            W = random.uniform(rng,shape=W_shape, minval=-init_value_W, maxval=+init_value_W)
            params=(W,)
        
        return None, params

    def apply_fun(params,inputs):
        #return jnp.einsum('ij,lj->li',params, inputs)
        W=params
        return jnp.dot(inputs,W.T)


    return init_fun, apply_fun



def GeneralConv(dimension_numbers, out_chan, filter_shape,
                strides=None, padding='VALID', W_init=None,
                b_init=normal(1e-6), ignore_b=False, dtype=jnp.float64):
    """Layer construction function for a general convolution layer."""
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    one = (1,) * len(filter_shape)
    strides = strides or one
    W_init = W_init or glorot_normal(rhs_spec.index('I'), rhs_spec.index('O'))
    def init_fun(rng, input_shape):
        filter_shape_iter = iter(filter_shape)
        kernel_shape = [out_chan if c == 'O' else
                        input_shape[lhs_spec.index('C')] if c == 'I' else
                        next(filter_shape_iter) for c in rhs_spec]
        output_shape = lax.conv_general_shape_tuple(
            input_shape, kernel_shape, strides, padding, dimension_numbers)
        
        k1, k2 = random.split(rng)

        if not ignore_b:
            bias_shape = [out_chan if c == 'C' else 1 for c in out_spec]
            bias_shape = tuple(itertools.dropwhile(lambda x: x == 1, bias_shape))

            W, b = W_init(k1, kernel_shape, dtype=dtype), b_init(k2, bias_shape, dtype=dtype)
            return output_shape, (W, b)
        else:
            W = W_init(k1, kernel_shape, dtype=dtype)
            return output_shape, (W, )
    def apply_fun(params, inputs, **kwargs):

        # move into lax.conv_general_dilated after defining padding='PERIODIC'
        inputs=periodic_padding(inputs,filter_shape,strides)

        if not ignore_b:
            W, b = params
            return lax.conv_general_dilated(inputs, W, strides, padding, one, one,
                                            dimension_numbers) + b
        else:
            W = params
            return lax.conv_general_dilated(inputs, W, strides, padding, one, one,
                                            dimension_numbers)

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
def logcosh_cpx(Re_Ws, Im_Ws):
    Re, Im  = cpx_cosh(Re_Ws, Im_Ws)
    Re_z = cpx_log_real(Re, Im, ) 
    Im_z = cpx_log_imag(Re, Im, )
    return Re_z, Im_z


@jit
def logcosh_real(Re_Ws, Im_Ws):
    Re, Im  = cpx_cosh(Re_Ws, Im_Ws)
    Re_z = cpx_log_real(Re, Im, )
    return Re_z


@jit
def logcosh_imag(Re_Ws, Im_Ws):
    Re, Im  = cpx_cosh(Re_Ws, Im_Ws)
    Im_z = cpx_log_imag(Re, Im, )

    return Im_z

