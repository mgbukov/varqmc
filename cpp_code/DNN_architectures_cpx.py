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


def GeneralDense_cpx_nonholo(W_shape, ignore_b=False):

    def init_fun(rng,input_shape):

        init_value_W=1E-2 #1E-1

        rng_real, rng_imag = random.split(rng)
        
        output_shape=(input_shape[0],W_shape[1])

        W_A = random.uniform(rng_real,shape=W_shape, minval=-init_value_W, maxval=+init_value_W)
        W_B = random.uniform(rng_imag,shape=W_shape, minval=-init_value_W, maxval=+init_value_W)

        if not ignore_b:

            init_value_b=1E-2

            rng_real, k1 = random.split(rng_real)
            rng_imag, k2 = random.split(rng_imag)

            b_real = random.uniform(k1,shape=(output_shape[1],), minval=-init_value_b, maxval=+init_value_b)
            b_imag = random.uniform(k2,shape=(output_shape[1],), minval=-init_value_b, maxval=+init_value_b)
            
            params=(W_A,W_B,W_B,W_A, b_real,b_imag,)
        
        else:
            params=(W_A,-W_B,W_B,W_A,)
        
        return output_shape, params

    def apply_fun(params,inputs, **kwargs):
        #return jnp.einsum('ij,lj->li',params, inputs)

        if isinstance(inputs, tuple):
            inputs_real, inputs_imag = inputs
        else:
            inputs_real = inputs
            inputs_imag = None

        z_real = jnp.dot(inputs_real,params[0]) 
        z_imag = jnp.dot(inputs_real,params[2])


        if inputs_imag is not None:
            z_real -= jnp.dot(inputs_imag,params[1]) 
            z_imag += jnp.dot(inputs_imag,params[3])

        if not ignore_b:
            # add bias
            z_real += params[4]
            z_imag += params[5]
       
        return z_real, z_imag


    return init_fun, apply_fun


def GeneralDense_cpx(W_shape, ignore_b=False):

    def init_fun(rng,input_shape):

        init_value_W=1E-2 

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
            
            params=(W_real,W_imag,b_real,b_imag)
        
        else:
            params=(W_real,W_imag,)
        
        return output_shape, params

    def apply_fun(params,inputs, **kwargs):
        #return jnp.einsum('ij,lj->li',params, inputs)

        # read-off params
        W_real=params[0]
        W_imag=params[1]

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
            z_real += params[2]
            z_imag += params[3]
       
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

            params=(W_real,W_imag,b_real,b_imag)
        
        else:
            params=(W_real,W_imag,)
            
        return output_shape, params


    if padding=='PERIODIC':
        padding='VALID'

    def apply_fun(params, inputs, **kwargs):

        # read-off params
        W_real=params[0]
        W_imag=params[1]

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
            b_real=params[2]
            b_imag=params[3]

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






##############################

@jit
def normalize_cpx(x, mean, std_mat_inv,):
    # mean = np.mean(x)
    # var_mat_inv = (sigma_mat)^(-1/2)

    x_real, x_imag = x

    z_real=x_real-mean.real
    z_imag=x_imag-mean.imag

    return std_mat_inv[0,0,...]*z_real + std_mat_inv[0,1,...]*z_imag,   std_mat_inv[1,0,...]*z_real + std_mat_inv[1,1,...]*z_imag



#@jit
def scale_cpx(inputs,comm,axis=(0,)):

    # compute mean
    Re_mean=jnp.mean(inputs[0],axis=axis,keepdims=True)
    Im_mean=jnp.mean(inputs[1],axis=axis,keepdims=True)
    mean_loc=Re_mean+1j*Im_mean

    mean=np.zeros_like(mean_loc)

    # MPI collect   
    comm.Allreduce([mean_loc._value, MPI.DOUBLE], [mean, MPI.DOUBLE], op=MPI.SUM)
    mean=device_put(mean/comm.Get_size())
 

    ### compute V^{-1/2}

    # correlation matrix V
    V_rr=jnp.mean((inputs[0] - mean.real)**2, axis=axis)# - Re_mean**2 
    V_ii=jnp.mean((inputs[1] - mean.imag)**2, axis=axis)# - Im_mean**2 
    V_ri=jnp.mean((inputs[0] - mean.real)*(inputs[1] - mean.imag), axis=axis)
    V_loc = jnp.array([[V_rr,V_ri],[V_ri,V_ii]])

    V=np.zeros_like(V_loc)

    # MPI collect   
    comm.Allreduce([V_loc._value, MPI.DOUBLE], [V, MPI.DOUBLE], op=MPI.SUM)
    V=device_put(V/comm.Get_size())


    # compute V^{-1/2}
    Id = jnp.array([np.identity(2) for _ in range(V.shape[-1])]).T
    mask=jnp.array([np.array([[1.,-1.],[-1.,1.]]) for _ in range(V.shape[-1])]).T

    # flip A <-> D, B -> -B, C -> -C
    VV = mask*V[::-1,::-1,:].transpose([1,0,2])

    # compute det and tr for each hidden unit
    det=jnp.linalg.det(VV.T)
    trace=jnp.trace(VV,axis1=0,axis2=1)

    # aux. variables: V^{-1/2} = 1/t(Id + VV/s)
    s=jnp.sqrt(det)
    t=jnp.sqrt(trace + 2.0*s)

    std_mat_inv=jnp.einsum('k,ijk->ijk',1.0/(s*t),VV) + jnp.einsum('k,ijk->ijk',1.0/t,Id)

    #from scipy.linalg import sqrtm, pinvh
    #std_mat_inv=np.array(list(pinvh(sqrtm(std)) for std in V.T) ).T


    # print( std_mat_inv2[...,1]  )             
    # print( pinvh(1.0/t[1]*(V[...,1] + s[1]*Id[...,1]))  )
    # print( pinvh(sqrtm(V[...,1])) )
            
    return mean, std_mat_inv





def BatchNorm_cpx(axis=(0, 1, 2), epsilon=1e-5, center=True, scale=True, beta_init=zeros, gamma_init=ones, dtype=np.float64):
    """Layer construction function for a batch normalization layer."""
    _beta_init = lambda rng, shape: beta_init(rng, shape, dtype) if center else ()
    _gamma_init = lambda rng, shape: gamma_init(rng, shape, dtype) if scale else ()
    axis = (axis,) if np.isscalar(axis) else axis
    
    def init_fun(rng, input_shape):
        shape = tuple(d for i, d in enumerate(input_shape) if i not in axis) 
        k1, k2 = random.split(rng)
        beta = _beta_init(k1, shape)#.astype(np.float64)
        gamma = np.array([[_gamma_init(k2, shape),zeros(k2,shape)],[zeros(k2,shape),_gamma_init(k2, shape)]]).T#.astype(np.float64)
        gamma/=np.sqrt(2.0)

        return input_shape, (beta, gamma)

    def apply_fun(params, x, mean, std_mat_inv, **kwargs):
        #
        beta, gamma = params
        # TODO(phawkins): np.expand_dims should accept an axis tuple.
        # (https://github.com/numpy/numpy/issues/12290)
        ed = tuple(None if i in axis else slice(None) for i in range(np.ndim(x[0])))
        beta = beta[ed]
        gamma = gamma[ed]
          
        z = normalize_cpx(x, mean=mean, std_mat_inv=std_mat_inv)

        if center and scale: 
            return gamma[...,0,0]*z[0] + gamma[...,0,1]*z[1] + beta[0],   gamma[...,1,0]*z[0] + gamma[...,1,1]*z[1] + beta[1]
        if center: 
            return z[0] + beta[0], z[1] + beta[1]
        if scale: 
            return gamma[...,0,0]*z[0] + gamma[...,0,1]*z[1],   gamma[...,1,0]*z[0] + gamma[...,1,1]*z[1]
        return z

    return init_fun, apply_fun



def BatchNorm_cpx_dyn(axis=(0, 1, 2), epsilon=1e-5, center=True, scale=True, beta_init=zeros, gamma_init=ones, dtype=np.float64):
    """Layer construction function for a batch normalization layer."""
    _beta_init = lambda rng, shape: beta_init(rng, shape, dtype) if center else ()
    _gamma_init = lambda rng, shape: gamma_init(rng, shape, dtype) if scale else ()
    axis = (axis,) if np.isscalar(axis) else axis
    
    init_fun, _ = BatchNorm_cpx(axis=axis, epsilon=epsilon, center=center, scale=scale, beta_init=beta_init, gamma_init=gamma_init, dtype=dtype)

    def apply_fun(params, x, fixpoint_iter=False, mean=None, std_mat_inv=None, comm=None, **kwargs):
        #
        beta, gamma = params
        # TODO(phawkins): np.expand_dims should accept an axis tuple.
        # (https://github.com/numpy/numpy/issues/12290)
        ed = tuple(None if i in axis else slice(None) for i in range(np.ndim(x[0])))
        beta = beta[ed]
        gamma = gamma[ed]
        
        
        if fixpoint_iter: # mean -> mean + std_mat.dot(mean_old), std_mat_inv -> std_mat_inv.dot(std_mat_iv_old)
            # old stats    
            std_mat = jnp.array(list(np.linalg.inv(mat) for mat in std_mat_inv.T) ).T
                        
            # normalize
            z=normalize_cpx(x, mean=mean, std_mat_inv=std_mat_inv)
            
            #comm=p_dict['comm'] # MPI communicator
            mean_new, std_mat_inv_new = scale_cpx(z, comm, axis=axis)

            # fix point iteration: 
            mean_vec= np.array([mean_new.squeeze().real, mean_new.squeeze().imag] )

            sigma_mean = np.einsum('ijp,jp->ip',std_mat,mean_vec)
            sigma_mean = (sigma_mean[0,...] + 1j*sigma_mean[1,...]).reshape(mean.shape)
    
            mean+=sigma_mean
            #std_mat_inv[:]=jnp.einsum('ijp,jkp->ikp',std_mat_inv_new,std_mat_inv)
            np.einsum('ijp,jkp->ikp',std_mat_inv_new,std_mat_inv,out=std_mat_inv)

        else: # mean -> mean, std_mat_inv -> std_mat_inv
            mean[:], std_mat_inv[:] = scale_cpx(x, comm ,axis=axis)  
        
        
        z = normalize_cpx(x, mean=mean, std_mat_inv=std_mat_inv)

        if center and scale: 
            return gamma[...,0,0]*z[0] + gamma[...,0,1]*z[1] + beta[0],   gamma[...,1,0]*z[0] + gamma[...,1,1]*z[1] + beta[1]
        if center: 
            return z[0] + beta[0], z[1] + beta[1]
        if scale: 
            return gamma[...,0,0]*z[0] + gamma[...,0,1]*z[1],   gamma[...,1,0]*z[0] + gamma[...,1,1]*z[1]
        return z

    return init_fun, apply_fun


def init_batchnorm_cpx_params(input_shape):

    broadcast_shape=(Ellipsis,)
    for _ in range(len(input_shape[1:])):
        broadcast_shape+=(None,)

    mean=np.zeros((1,)+input_shape[1:],dtype=np.complex128)
    std_mat_inv=np.broadcast_to(np.identity(2)[broadcast_shape], (2,2)+input_shape[1:]).copy()

    return mean, std_mat_inv



def Norm_real(center=True, scale=True, a_init=ones, b_init=zeros, dtype=np.float64):
    """Layer construction function for a batch normalization layer."""
    _a_init = lambda rng, shape: a_init(rng, shape, dtype) if scale else ()
    _b_init = lambda rng, shape: b_init(rng, shape, dtype) if center else ()
    
    def init_fun(rng, input_shape):
        shape = (1,)
        k1, k2 = random.split(rng)
        k2, k3 = random.split(k2)

        a = _a_init(k1, shape)
        b = _b_init(k2, shape)+10.0
        c = _a_init(k3, shape)
        
        return input_shape, (a,b,c)

    def apply_fun(params, x, a=1.0, b=+10.0, c=1.0, **kwargs):
        #a,b,c   = params
        x_real, x_imag = x

        a=jnp.abs(a)
        x_real= c*(jnp.where(x_real < b, a*(x_real-b), -jnp.expm1(-a*(x_real-b)), ) + a*b)

        #x_real=0.1*jnp.tanh(a*x_real)
        
        #x_real=-jnp.log(jnp.cosh(x_real-1.0))
        #x_real/=6*128.

        # print(x_real)
        # exit()

        return (x_real, x_imag)
        
    return init_fun, apply_fun



###############


def serial(*layers):
    """Combinator for composing layers in serial.

    Args:
    *layers: a sequence of layers, each an (init_fun, apply_fun) pair.

    Returns:
    A new layer, meaning an (init_fun, apply_fun) pair, representing the serial
    composition of the given sequence of layers.
    """
    nlayers = len(layers)
    init_funs, apply_funs = zip(*layers)
    apply_fun_args = list((dict(),)*nlayers)
    def init_fun(rng, input_shape):
        params = []
        for init_fun in init_funs:
            rng, layer_rng = random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        return input_shape, params
    def apply_fun(params, inputs, rng=None, kwargs=apply_fun_args):
        rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
        for fun, param, rng, kwarg in zip(apply_funs, params, rngs, kwargs):
            inputs = fun(param, inputs, rng=rng, **kwarg)
        return inputs
    return init_fun, apply_fun, apply_fun_args




def elementwise(fun, **fun_kwargs):
    """Layer that applies a scalar function elementwise on its inputs."""
    init_fun = lambda rng, input_shape: (input_shape, ())
    def apply_fun(params, inputs, **kwargs):
        kwargs.pop('rng', None)
        return fun(inputs, **kwargs, **fun_kwargs) #
    return init_fun, apply_fun





LogCosh_cpx=elementwise(logcosh_cpx)
Poly_cpx=elementwise(poly_cpx)
#Normalize_cpx=elementwise(normalize_cpx)



