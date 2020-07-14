from jax.config import config
config.update("jax_enable_x64", True)
from jax import jit, grad, random, device_put 
#from jax.experimental.stax import BatchNorm

#from mpi4py import MPI

import functools
import itertools
from jax import lax, random
from jax import ops, disable_jit
import jax.numpy as jnp
import numpy as np

from jax.nn import (relu, log_softmax, softmax, softplus, sigmoid, elu,
                    leaky_relu, selu, gelu, normalize)
from jax.nn.initializers import glorot_normal, normal, ones, zeros

from functools import partial



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
    fun=partial(fun, **fun_kwargs)
    init_fun = lambda rng, input_shape: (input_shape, ())
    def apply_fun(params, inputs, **kwargs):
        kwargs.pop('rng', None)
        return fun(inputs, **kwargs) #
    return init_fun, apply_fun


