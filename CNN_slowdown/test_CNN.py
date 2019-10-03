import sys,os
#os.environ['XLA_FLAGS']='--xla_dump_to=/tmp/CNN_logfiles'


from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, random, device_put, vmap, ops, partial


import numpy as np

seed=1
np.random.seed(seed)
np.random.RandomState(seed)
rng = random.PRNGKey(seed)


from DNN_architectures import *

import time

####################################################################

NN_type='CNN' #'DNN' # 




L=4
N_sites=L*L


if NN_type=='DNN':

    W_shape=[4,N_sites]
    N_symm=L*L*2*2*2 # no Z symemtry
    init_params, apply_layer = GeneralDeep(W_shape, ignore_b=True)
    _input_shape=None

    # tuple to reshape output before symmetrization
    input_shape = (-1,N_sites)
    

elif NN_type=='CNN':

    N_symm=2*2*2 # no Z, Tx, Ty symemtry

    dimension_numbers=('NCHW', 'OIHW', 'NCHW') # default
    out_chan=1
    filter_shape=(2,2)
    strides=(1,1)

    _input_shape=np.array((1,1,L,L),dtype=np.int) # NCHW input format
    # add padding dimensions for periodic BC
    _input_shape+=np.array((0,0)+strides)

    init_params, apply_layer = GeneralConv(dimension_numbers, out_chan, filter_shape, strides=strides, padding='VALID', ignore_b=True)
        
    # tuple to reshape input before passing to evalute() func
    input_shape = (-1,1,L,L)
   

# initialize parameters
W_fc_real, = init_params(rng,_input_shape)[1]
W_fc_imag, = init_params(rng,_input_shape)[1]
params=[W_fc_real, W_fc_imag, ]



@jit
def evaluate(params, batch):
    # reshaping required inside evaluate func because of per-sample gradients
    batch=batch.reshape(input_shape)

    # apply dense layer
    Re_Ws = apply_layer(params[0], batch)
    Im_Ws = apply_layer(params[1], batch) 

    # apply logcosh nonlinearity
    Re, Im  = cpx_cosh(Re_Ws, Im_Ws)
    Re_z_fc = cpx_log_real(Re, Im, ) 
    
    return jnp.sum(Re_z_fc)


@jit
def compute_grad_log_psi(params,batch,):
	return vmap(partial( jit(grad(evaluate)),   params))(batch, )
	#return vmap(partial( grad(evaluate), params))(batch, )
	



###########################

N_MC_points=100 # number of points

# define data
batch=np.ones((N_MC_points,N_symm,L,L),dtype=np.float64)


#print(evaluate(params,batch))



for _ in range (10):

    ti = time.time()
    
    d_psi = compute_grad_log_psi(params,batch)
    #print(d_psi[0].shape)

    print("gradients took {0:.4f} secs.".format(time.time()-ti))




