from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, random, device_put 
from jax.experimental.stax import BatchNorm

import functools
import itertools
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np

from jax.nn import (relu, log_softmax, softmax, softplus, sigmoid, elu,
                    leaky_relu, selu, gelu, normalize)
from jax.nn.initializers import glorot_normal, normal, ones, zeros

from DNN_architectures_cpx import scale_cpx, normalize_cpx, test



def BatchNorm_cpx(axis=(0, 1, 2), epsilon=1e-5, center=True, scale=True,
              beta_init=zeros, gamma_init=ones):
    """Layer construction function for a batch normalization layer."""
    _beta_init = lambda rng, shape: beta_init(rng, shape) if center else ()
    _gamma_init = lambda rng, shape: gamma_init(rng, shape) if scale else ()
    axis = (axis,) if np.isscalar(axis) else axis
    
    def init_fun(rng, input_shape):
        shape = tuple(d for i, d in enumerate(input_shape) if i not in axis) 
        k1, k2 = random.split(rng)
        beta = _beta_init(k1, shape)
        gamma = np.array([[_gamma_init(k2, shape),zeros(k2,shape)],[zeros(k2,shape),_gamma_init(k2, shape)]]).T
        gamma/=np.sqrt(2.0)
        return input_shape, (beta, gamma)

    def apply_fun(params, x, mean=None, std_mat_inv=None, overwrite=False, fixpoint_iter=False):
        beta, gamma = params
        # TODO(phawkins): np.expand_dims should accept an axis tuple.
        # (https://github.com/numpy/numpy/issues/12290)
        ed = tuple(None if i in axis else slice(None) for i in range(np.ndim(x[0])))
        beta = beta[ed]
        gamma = gamma[ed]
        
        if overwrite==True:

            if fixpoint_iter: # mean -> mean + std_mat.dot(mean_old), std_mat_inv -> std_mat_inv.dot(std_mat_iv_old)
                # compute new stats
                mean_new, std_mat_inv_new = scale_cpx(x,axis=axis)

                # fix point iteration: 
                std_mat = np.array(list(np.linalg.inv(mat) for mat in std_mat_inv.T) ).T
                mean_vec=np.array([mean_new.squeeze().real, mean_new.squeeze().imag])

                sigma_mean = np.einsum('ijp,jp->ip',std_mat,mean_vec)
                sigma_mean = (sigma_mean[0,...] + 1j*sigma_mean[1,...]).reshape(mean_new.shape)
        
                mean+=sigma_mean
                np.einsum('ijp,jkp->ikp',std_mat_inv_new, std_mat_inv, out=std_mat_inv)
            
            else: # mean -> mean, std_mat_inv -> std_mat_inv
                mean[...], std_mat_inv[...] = scale_cpx(x,axis=axis)  
            
        
        z = normalize_cpx(x, mean=mean, std_mat_inv=std_mat_inv)
        
        if center and scale: 
            return gamma[...,0,0]*z[0] + gamma[...,0,1]*z[1] + beta[0],   gamma[...,1,0]*z[0] + gamma[...,1,1]*z[1] + beta[1]
        if center: 
            return z[0] + beta[0], z[1] + beta[1]
        if scale: 
            return gamma[...,0,0]*z[0] + gamma[...,0,1]*z[1],   gamma[...,1,0]*z[0] + gamma[...,1,1]*z[1]
        return z

    return init_fun, apply_fun


def init_beatchnorm_params(input_shape):

    broadcast_shape=(Ellipsis,)
    for _ in range(len(input_shape[1:])):
        broadcast_shape+=(None,)

    mean=np.zeros((1,)+input_shape[1:],dtype=np.complex128)
    std_mat_inv=np.broadcast_to(np.identity(2)[broadcast_shape], (2,2)+input_shape[1:]).copy()

    return mean, std_mat_inv


A, B = BatchNorm_cpx(axis=(0,))

input_shape=(8,16)
rng = random.PRNGKey(0)
np.random.seed(0)
input_shape, (beta,gamma) = A(rng,input_shape)

params=(beta,gamma)
inputs=(np.random.uniform(size=input_shape), np.random.uniform(size=input_shape))

#print(inputs[1])
#exit()


mean,std_mat_inv=init_beatchnorm_params(input_shape)


print(mean)

zz=B(params,inputs,mean=mean,std_mat_inv=std_mat_inv,overwrite=True,fixpoint_iter=True)

print(mean)

#print(zz[0])
#print(zz[1])

exit()

A2, B2 = BatchNorm(axis=[0,1])
input_shape, (beta2,gamma2) = A2(rng,input_shape)

params2=(beta2,gamma2)
zz2=B2(params2,inputs[0])

#print(zz2)

