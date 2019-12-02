from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
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

# aliases for backwards compatibility
#glorot = glorot_normal
#randn = normal
#logsoftmax = log_softmax

# import numpy as np
# seed=1
# np.random.seed(seed)
# np.random.RandomState(seed)
# rng = random.PRNGKey(seed)


def periodic_padding(inputs,filter_shape,strides):
    n_x=filter_shape[0]-strides[0]
    n_y=filter_shape[1]-strides[1]
    return jnp.pad(inputs, ((0,0),(0,0),(0,n_x),(0,n_y)), mode='wrap')




def GeneralDense_cpx(W_shape, ignore_b=False):

    def init_fun(rng,input_shape):

        init_value_W=1E-2 #1E-1

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


def GeneralConv_cpx(dimension_numbers, out_chan, filter_shape,
                strides=None, padding='VALID', W_init=None,
                b_init=normal(1e-6), ignore_b=False, dtype=jnp.float64):

    """Layer construction function for a general convolution layer."""
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    one = (1,) * len(filter_shape)
    strides = strides or one
    W_init = W_init or glorot_normal(rhs_spec.index('I'), rhs_spec.index('O'))
    
    
    def init_fun(rng, input_shape, padding=padding):
        if padding=='PERIODIC':
            # add padding dimensions
            input_shape+=np.array((0,0)+strides)
            padding='VALID'

        filter_shape_iter = iter(filter_shape)
        kernel_shape = [out_chan if c == 'O' else
                        input_shape[lhs_spec.index('C')] if c == 'I' else
                        next(filter_shape_iter) for c in rhs_spec]
        output_shape = lax.conv_general_shape_tuple(
            input_shape, kernel_shape, strides, padding, dimension_numbers)
        
        rng_real, rng_imag = random.split(rng)
        k1_real, k2_real = random.split(rng_real)
        k1_imag, k2_imag = random.split(rng_imag)

        W_real = W_init(k1_real, kernel_shape, dtype=dtype)
        W_imag = W_init(k1_imag, kernel_shape, dtype=dtype)

        if not ignore_b:
            bias_shape = [out_chan if c == 'C' else 1 for c in out_spec]
            bias_shape = tuple(itertools.dropwhile(lambda x: x == 1, bias_shape))

            b_real = b_init(k2_real, bias_shape, dtype=dtype)
            b_imag = b_init(k2_imag, bias_shape, dtype=dtype)

            params = ((W_real,b_real),(W_imag,b_imag))
        
        else:
            params = ((W_real,),(W_imag,))
            
        return output_shape, params


    if padding=='PERIODIC':
        padding='VALID'

    def apply_fun(params, inputs, **kwargs):

        # read-off params
        W_real=params[0][0]
        W_imag=params[1][0]

        if isinstance(inputs, tuple):
            inputs_real, inputs_imag = inputs
        else:
            inputs_real = inputs
            inputs_imag = None

        # move into lax.conv_general_dilated after defining padding='PERIODIC'
        inputs_real=periodic_padding(inputs_real.astype(W_real.dtype),filter_shape,strides)
        
        z_real = lax.conv_general_dilated(inputs_real, W_real, strides, padding, one, one, dimension_numbers)
        z_imag = lax.conv_general_dilated(inputs_real, W_imag, strides, padding, one, one, dimension_numbers)

        if inputs_imag is not None:
            # move into lax.conv_general_dilated after defining padding='PERIODIC'
            inputs_imag=periodic_padding(inputs_imag.astype(W_imag.dtype),filter_shape,strides)

            z_real -= lax.conv_general_dilated(inputs_imag, W_imag, strides, padding, one, one, dimension_numbers)
            z_imag += lax.conv_general_dilated(inputs_imag, W_real, strides, padding, one, one, dimension_numbers)

        if not ignore_b:
            # read-off params
            b_real=params[0][1]
            b_imag=params[1][1]

            z_real += b_real
            z_imag += b_imag
       
        return z_real, z_imag

    return init_fun, apply_fun






# nonlinearities

@jit
def poly_cpx(x):
    x_real, x_imag = x
    Re = 0.5*x_real**2 - 0.0833333*x_real**4 + 0.0222222*x_real**6 - 0.5*x_imag**2 + 0.5*x_real**2*x_imag**2 - 0.333333*x_real**4*x_imag**2 - 0.0833333*x_imag**4 + 0.333333*x_real**2*x_imag**4 - 0.0222222*x_imag**6
    Im = x_real*x_imag - 0.333333*x_real**3*x_imag + 0.133333*x_real**5*x_imag + 0.333333*x_real*x_imag**3 - 0.444444*x_real**3*x_imag**3 + 0.133333*x_real*x_imag**5
    return Re, Im


@jit
def poly_real(x):
    x_real, x_imag = x
    Re_z = 0.5*x_real**2 - 0.0833333*x_real**4 + 0.0222222*x_real**6 - 0.5*x_imag**2 + 0.5*x_real**2*x_imag**2 - 0.333333*x_real**4*x_imag**2 - 0.0833333*x_imag**4 + 0.333333*x_real**2*x_imag**4 - 0.0222222*x_imag**6
    return Re_z


@jit
def poly_imag(x):
    x_real, x_imag = x
    Im_z = x_real*x_imag - 0.333333*x_real**3*x_imag + 0.133333*x_real**5*x_imag + 0.333333*x_real*x_imag**3 - 0.444444*x_real**3*x_imag**3 + 0.133333*x_real*x_imag**5
    return Im_z


#############################################


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

@jit
def logcosh_real(Ws):
    Re_Ws, Im_Ws = Ws
    Re, Im  = cpx_cosh(Re_Ws, Im_Ws)
    Re_z = cpx_log_real(Re, Im, )
    return Re_z


@jit
def logcosh_imag(Ws):
    Re_Ws, Im_Ws = Ws
    Re, Im  = cpx_cosh(Re_Ws, Im_Ws)
    Im_z = cpx_log_imag(Re, Im, )
    return Im_z

###############

LogCosh_cpx=elementwise(logcosh_cpx)
Poly_cpx=elementwise(poly_cpx)
