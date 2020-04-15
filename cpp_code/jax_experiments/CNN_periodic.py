import os, sys

quspin_path = os.path.join(os.getcwd(),"../")
sys.path.insert(0,quspin_path)


from jax.config import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import jit, grad, random, device_put, partial
from jax.tree_util import tree_structure, tree_flatten, tree_unflatten


from DNN_architectures_real import *
from DNN_architectures_common import *


seed=1
np.random.seed(seed)
np.random.RandomState(seed)
rng = random.PRNGKey(seed)


L=4
N_symm= 2*2*2 #
N_points=2


dim_nums=('NCHW', 'OIHW', 'NCHW') # default
out_chan=2
filter_shape=(L,L)

out_chan_2=2 
filter_shape_2=(1,1) 

input_shape=np.array((N_points*N_symm,1,L,L),dtype=np.int) # NCHW input format


#init_params, apply_layer =GeneralConvPeriodic(dim_nums, out_chan, filter_shape, W_init=W_init, b_init=b_init) # 


NN_arch_log = {
                        'layer_1': GeneralConvPeriodic(dim_nums, out_chan, filter_shape,  init_value_W=1E-2, init_value_b=1E-2, ), 
                        'nonlin_1': elementwise(logcosh),
                        'layer_2': GeneralConvPeriodic(dim_nums, out_chan_2, filter_shape_2,  ignore_b=True, init_value_W=1E-2, init_value_b=1E-2, ), 
                        'nonlin_2': elementwise(logcosh),
              
                }

init_params, apply_layer, apply_fun_args = serial(*NN_arch_log.values())


output_shape,params = init_params(rng,input_shape)

print(output_shape)

exit()

#print(params[0])



batch=np.ones((N_points*N_symm,1,L,L),dtype=np.float64)
#batch[0,0,0,0]=-1




z=apply_layer(params,batch)

print(z.shape)

#print(z[0,0,...].reshape(L,L))




