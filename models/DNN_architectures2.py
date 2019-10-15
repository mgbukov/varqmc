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

from jax.nn import (relu, log_softmax, softmax, softplus, sigmoid, elu,
                    leaky_relu, selu, gelu, normalize)
from jax.nn.initializers import glorot_normal, normal, ones, zeros

# aliases for backwards compatibility
glorot = glorot_normal
randn = normal
logsoftmax = log_softmax

import numpy as np
seed=1
np.random.seed(seed)
np.random.RandomState(seed)
rng = random.PRNGKey(seed)


def periodic_padding(inputs,filter_shape,strides):
    n_x=filter_shape[0]-strides[0]
    n_y=filter_shape[1]-strides[1]
    return jnp.pad(inputs, ((0,0),(0,0),(0,n_x),(0,n_y)), mode='wrap')




def GeneralDeep_cpx(W_shape, ignore_b=False):

    def init_fun(rng,input_shape):

        init_value_W=1E-1

        rng_real, rng_imag = random.split(rng)
        
        output_shape=(input_shape[0],W_shape[0])

        W_real = random.uniform(rng_real,shape=W_shape, minval=-init_value_W, maxval=+init_value_W)
        W_imag = random.uniform(rng_imag,shape=W_shape, minval=-init_value_W, maxval=+init_value_W)

        if not ignore_b:

            init_value_b=1E-1 

            rng_real, k1 = random.split(rng_real)
            rng_imag, k2 = random.split(rng_imag)

            b_real = random.uniform(k1,shape=(W_shape[0],), minval=-init_value_b, maxval=+init_value_b)
            b_imag = random.uniform(k2,shape=(W_shape[0],), minval=-init_value_b, maxval=+init_value_b)
            
            params=((W_real,b_real),(W_imag,b_imag))
        
        else:
            params=((W_real,),(W_imag,))
        
        return output_shape, params

    def apply_fun(params,inputs, **kwargs):
        #return jnp.einsum('ij,lj->li',params, inputs)

        # read-off params
        W_real=params[0][0]
        W_imag=params[1][0]

        inputs_real, inputs_imag = inputs

        z_real = jnp.dot(inputs_real,W_real.T) 
        z_imag = jnp.dot(inputs_real,W_imag.T)


        if inputs_imag is not None:
            z_real -= jnp.dot(inputs_imag,W_imag.T)
            z_imag += jnp.dot(inputs_imag,W_real.T)

        if not ignore_b:
            # read-off params
            b_real=params[0][1]
            b_imag=params[1][1]

            z_real += b_real
            z_imag += b_imag
       
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
            input_shape+=jnp.array((0,0)+strides)
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

        inputs_real, inputs_imag = inputs

        # move into lax.conv_general_dilated after defining padding='PERIODIC'
        inputs_real=periodic_padding(inputs_real,filter_shape,strides)
        
        z_real = lax.conv_general_dilated(inputs_real, W_real, strides, padding, one, one, dimension_numbers)
        z_imag = lax.conv_general_dilated(inputs_real, W_imag, strides, padding, one, one, dimension_numbers)

        if inputs_imag is not None:
            # move into lax.conv_general_dilated after defining padding='PERIODIC'
            inputs_imag=periodic_padding(inputs_imag,filter_shape,strides)

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


LogCosh_cpx=elementwise(logcosh_cpx)

#####################



ignore_b=False # True #

W_shape_1=(4,16)
W_shape_2=(2,4)

input_shape_1=(3,16)
#data=np.ones(input_shape_1), None
data=np.random.uniform(size=input_shape_1), None


# define DNN
init_random_params, predict = serial(
                                        GeneralDeep_cpx(W_shape_1, ignore_b=ignore_b), LogCosh_cpx,
                                        GeneralDeep_cpx(W_shape_2, ignore_b=ignore_b), LogCosh_cpx,     
                                    )
params_indx=[0,2]


output_shape, params = init_random_params(rng,input_shape_1)
z=predict(params,data)

print(z)

exit()

################################################################

N_symm=2*2*2 # no Z, Tx, Ty symmetry

dimension_numbers=('NCHW', 'OIHW', 'NCHW') # default
out_chan=1
filter_shape=(2,2)
strides=(1,1)

input_shape_1=(3,1,4,4)
#data=np.ones(input_shape_1), None
data=np.random.uniform(size=input_shape_1), None



#init_params, apply_layer = GeneralConv_cpx(dimension_numbers, out_chan, filter_shape, strides=strides, padding='PERIODIC', ignore_b=ignore_b)
            
#output_shape, params=init_params(rng,input_shape_1)
#z=apply_layer(params,data)


# define DNN
init_random_params, predict = serial(
                                        GeneralConv_cpx(dimension_numbers, out_chan, filter_shape, strides=strides, padding='PERIODIC', ignore_b=ignore_b), LogCosh_cpx,
                                        GeneralConv_cpx(dimension_numbers, out_chan, filter_shape, strides=strides, padding='PERIODIC', ignore_b=ignore_b), LogCosh_cpx,     
                                    )
params_indx=[0,2]


output_shape, params = init_random_params(rng,input_shape_1)
z=predict(params,data)




NN_params=[]
shapes_cpx=[]
size_cpx=[]
N_layer_params=[]

for indx in params_indx:
    for param in params[indx]: # loop real / imag
        for i,W in enumerate(param): # loop W / b
            shapes_cpx.append(W.shape)
            size_cpx.append(W.size)
            NN_params.append(W.flatten())
    N_layer_params.append(i+1)



shapes_cpx=np.array(shapes_cpx)
size_cpx=np.array(size_cpx) 


# print(shapes_cpx)
# print(size_cpx)
# print(N_layer_params)


NN_params=jnp.concatenate(NN_params)

print(NN_params)
#exit()


Ndims=np.insert(np.cumsum(size_cpx), 0, 0)



_k=0
orig_params=[]
for l in range(len(params_indx)):
    s=()
    for i in range(2): # loop real / imag
        t=()
        for j in range(N_layer_params[l]): # loop W / b
            t+=(NN_params[Ndims[_k]:Ndims[_k+1]].reshape(shapes_cpx[_k]),)
            _k+=1

        s+=(t,)
    
    orig_params.append(s)
    orig_params.append(())
    

print(orig_params)
print()
print(params)




