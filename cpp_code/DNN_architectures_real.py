from jax.config import config
config.update("jax_enable_x64", True)
from jax import jit, lax, random, device_put #, disable_jit

import functools
import itertools

import jax.numpy as jnp
import numpy as np

#from mpi4py import MPI

from jax.nn.initializers import glorot_normal, normal, ones, zeros





def GeneralConvPeriodic(dimension_numbers, out_chan, filter_shape, ignore_b=False, init_value_W=1E-2, init_value_b=1E-2, dense_output=False ):
    """Layer construction function for a general convolution layer."""
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    one = (1,) * len(filter_shape)
    strides = one
    padding='VALID'

    output_shape_dense=(-1,out_chan)
    transpose_shape=(0,3,2,1)

    def init_fun(rng, input_shape,):
        
        # add padding dimensions
        input_shape=tuple(input_shape)
        input_shape_aug=input_shape+np.array((0,0)+tuple(strides))

        
        filter_shape_iter = iter(filter_shape)
        kernel_shape = [out_chan if c == 'O' else
                        input_shape_aug[lhs_spec.index('C')] if c == 'I' else
                        next(filter_shape_iter) for c in rhs_spec]
        
        k1, k2 = random.split(rng)
        #W = W_init(k1, kernel_shape)
        #W = random.uniform(rng,shape=kernel_shape, minval=-init_value_W, maxval=+init_value_W)
        W = random.uniform(rng,shape=(kernel_shape[2]*kernel_shape[3]*kernel_shape[1],kernel_shape[0]), minval=-init_value_W, maxval=+init_value_W).T.reshape(kernel_shape)
        
        # normalize W
        #norm=jnp.sqrt(filter_shape[0]*filter_shape[1]*(input_shape[1]+out_chan))
        norm=jnp.sqrt(filter_shape[0]*filter_shape[1]+out_chan)
        W/=norm
        
        # output
        if not dense_output:
            output_shape = lax.conv_general_shape_tuple(input_shape_aug, kernel_shape, strides, padding, dimension_numbers)
            output_shape=output_shape[:2]+input_shape[2:]
        else:
            output_shape = output_shape_dense 
            

        if ignore_b:
            return output_shape, (W,)
        else:  
            bias_shape = [out_chan if c == 'C' else 1 for c in out_spec]
            bias_shape = tuple(itertools.dropwhile(lambda x: x == 1, bias_shape))

            #b = b_init(k2, bias_shape)
            b = random.uniform(k1,shape=bias_shape, minval=-init_value_b, maxval=+init_value_b)
            return output_shape, (W, b)

    def periodic_padding(inputs,):
        n_x=filter_shape[0]-strides[0]
        n_y=filter_shape[1]-strides[1]
        return jnp.pad(inputs.astype(np.float64), ((0,0),(0,0),(0,n_x),(0,n_y)), mode='wrap')
        

    def apply_fun(params, inputs, **kwargs):
        W = params[0]
        # move into lax.conv_general_dilated after defining padding='PERIODIC'
        a=periodic_padding(inputs,)
        if ignore_b:
            if not dense_output:
                return lax.conv_general_dilated(a, W, strides, padding, one, one, dimension_numbers=dimension_numbers) 
            else:
                return jnp.transpose(lax.conv_general_dilated(a, W, strides, padding, one, one, dimension_numbers=dimension_numbers), transpose_shape).reshape(output_shape_dense)
                       
        else:
            if not dense_output:
                return lax.conv_general_dilated(a, W, strides, padding, one, one, dimension_numbers=dimension_numbers) + params[1]
            else:
                return jnp.transpose(lax.conv_general_dilated(a, W, strides, padding, one, one, dimension_numbers=dimension_numbers) + params[1], transpose_shape).reshape(output_shape_dense)
                
    return init_fun, apply_fun



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





# nonlinearities


@jit
def logcosh(x):
    return jnp.log(jnp.cosh(x))


@jit
def xtanh(x):
    return jnp.abs(x)*jnp.tanh(x)
    
#@jit
def symmetric_pool(x,reduce_shape, output_shape,):
    # symmetrize
    x   = jnp.sum(x.reshape(reduce_shape,order='C') / jnp.sqrt(reduce_shape[1]+reduce_shape[3]),  axis=[1,3])   
    # sum over hidden neurons
    x   = jnp.sum(x.reshape(output_shape) / jnp.sqrt(output_shape[1]), axis=[1,])
    return x

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


def Reshape_layer(input_shape, output_shape):

    def init_fun(rng,input_shape):
        return output_shape, ()

    def apply_fun(params,x,):
        return x.reshape(output_shape)

    return init_fun, apply_fun



def Regularization(reduce_shape, output_shape,center=True, b_init=zeros, dtype=np.float64,):
    """Layer construction function for a batch normalization layer."""
    _b_init = lambda rng, shape: b_init(rng, shape, dtype) if center else ()

    norm_1=jnp.sqrt(reduce_shape[1]+reduce_shape[3]) 
    norm_2=jnp.sqrt(output_shape[1])


    def init_fun(rng, input_shape):
        b_shape = (1,)
        b = _b_init(rng, b_shape)
        output_shape=(-1,)
        return output_shape, (b,)

    def apply_fun(params, x, **kwargs):
        b,   = params
         
        # 1/(N/p) = p/N : p different terms in sum left
        # uncorrelated: 1/\sqrt(p) 
        # correlated: 1/p

        #print(x.shape, reduce_shape)
        
        # symmetrize
        log_psi = jnp.sum(x.reshape(reduce_shape,order='C')/norm_1, axis=[1,3])

        #print(log_psi.shape, output_shape)

        # sum over hidden neurons
        log_psi = jnp.sum(  log_psi.reshape(output_shape)/norm_2, axis=[1,])

        #print(log_psi.shape)

        # regularize output
        a=8.0
        log_psi=a*jnp.tanh((log_psi-b)/a) + b
        
        return log_psi
        
    return init_fun, apply_fun





###############





