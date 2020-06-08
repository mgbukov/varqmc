from jax.config import config
config.update("jax_enable_x64", True)
from jax import jit, lax, random, device_put #, disable_jit

import functools
import itertools

import jax.numpy as jnp
import numpy as np

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
        #W = random.uniform(rng,shape=kernel_shape, minval=-init_value_W, maxval=+init_value_W)
        W = random.uniform(k1,shape=(kernel_shape[2]*kernel_shape[3]*kernel_shape[1],kernel_shape[0]), minval=-init_value_W, maxval=+init_value_W).T.reshape(kernel_shape)
        
        # normalize W
        norm=jnp.sqrt(filter_shape[0]*filter_shape[1]*(input_shape[1]+out_chan))
        W/=norm
                

        if ignore_b:
            params=(W,)
        else:  
            bias_shape = [out_chan if c == 'C' else 1 for c in out_spec]
            bias_shape = tuple(itertools.dropwhile(lambda x: x == 1, bias_shape))

            b = random.uniform(k2,shape=bias_shape, minval=-init_value_b, maxval=+init_value_b)
            params=(W, b)

        # output
        if not dense_output:
            output_shape = lax.conv_general_shape_tuple(input_shape_aug, kernel_shape, strides, padding, dimension_numbers)
            output_shape=output_shape[:2]+input_shape[2:]
        else:
            output_shape = output_shape_dense


        return output_shape, params


    def periodic_padding(inputs,):
        n_x=filter_shape[0]-strides[0]
        n_y=filter_shape[1]-strides[1]
        return jnp.pad(inputs.astype(np.float64), ((0,0),(0,0),(0,n_x),(0,n_y)), mode='wrap')

    def flatten(x,):
        if dense_output:
            return jnp.transpose(x, transpose_shape).reshape(output_shape_dense)
        else:
            return x

    def apply_fun(params, inputs, **kwargs):
        W = params[0]
        # move into lax.conv_general_dilated after defining padding='PERIODIC'
        inputs=periodic_padding(inputs,)
        output = lax.conv_general_dilated(inputs, W, strides, padding, one, one, dimension_numbers=dimension_numbers)

        if ignore_b:
            return flatten(output)         
        else:
            return flatten(output + params[1])
                
    return init_fun, apply_fun



def GeneralDense(in_chan, out_chan, filter_size, ignore_b=False, init_value_W=1E-2, init_value_b=1E-2):
    
    norm=jnp.sqrt(filter_size*(in_chan+out_chan))

    def init_fun(rng,input_shape):   

        if out_chan==1:
            output_shape=(input_shape[0],)
            W_shape=(input_shape[1],)
            b_shape=(1,)
        else:
            output_shape=(input_shape[0],out_chan)   
            W_shape=(input_shape[1],out_chan)
            b_shape=(output_shape[1],)        

        W = random.uniform(rng,shape=W_shape, minval=-init_value_W, maxval=+init_value_W)
            
        W/=norm # see apply func

        if not ignore_b:
            rng, k1 = random.split(rng)
            b = random.uniform(k1,shape=b_shape, minval=-init_value_b, maxval=+init_value_b)
            params=(W,b,)
        
        else:
            params=(W,)
        
        return output_shape, params

    def apply_fun(params,inputs, **kwargs):
        z = jnp.dot(inputs,params[0])#/norm
        if not ignore_b:
            # add bias
            z += params[1]  
        return z

    return init_fun, apply_fun



##########################

# nonlinearities


@jit
def logcosh(x):
    return jnp.log(jnp.cosh(x))


@jit
def xtanh(x):
    #return jnp.abs(x)*jnp.tanh(x)
    #return jnp.tanh(x)
    #return jnp.sinh(x)
    return x-0.5*jnp.tanh(x)
    #return 2.0*x-jnp.tanh(x)


##########################


#@jit
def symmetric_pool(x,reduce_shape, output_shape,):
    # symmetrize
    x = jnp.sum(x.reshape(reduce_shape,order='C') / jnp.sqrt(reduce_shape[1]+reduce_shape[3]),  axis=[1,3])
    # sum over hidden neurons
    x = jnp.sum(x.reshape(output_shape) / jnp.sqrt(output_shape[1]), axis=[1,])
    return x


def symmetrize(x, reduce_shape,):
    # symmetrize
    x = jnp.sum(x.reshape(reduce_shape,order='C') / jnp.sqrt(reduce_shape[1]+reduce_shape[3]),  axis=[1,3])
    return x

def uniform_pool(x, output_shape,):
    # sum over hidden neurons
    x = jnp.sum(x.reshape(output_shape) / jnp.sqrt(output_shape[1]), axis=[1,])
    #x = jnp.sum(x.reshape(output_shape),axis=[1,])
    return x



def Flatten(x, chan=1, transpose_shape=(0,3,2,1), ):
    # transpose_shape: corresponds 'NCHW'
    return jnp.transpose(x, transpose_shape).reshape(-1,chan)
    

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






def Regularization(a=8.0,center=True, b_init=zeros, dtype=np.float64,):
    """Layer construction function for a batch normalization layer."""
    _b_init = lambda rng, shape: b_init(rng, shape, dtype) if center else ()

    def init_fun(rng, input_shape):
        b_shape = (1,)
        b = _b_init(rng, b_shape)
        output_shape=(-1,)
        return output_shape, (b,)

    def apply_fun(params, x, **kwargs):
        b,   = params
        # regularize output
        log_psi=a*jnp.tanh((x-b)/a) + b
        return log_psi
        
    return init_fun, apply_fun


###############





