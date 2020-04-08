from jax.config import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import jit, grad, random, device_put, partial
from jax.tree_util import tree_structure, tree_flatten, tree_unflatten


from DNN_architectures_real import *


seed=1
np.random.seed(seed)
np.random.RandomState(seed)
rng = random.PRNGKey(seed)


L=4
N_symm= 2*2*2 #
N_points=1


dim_nums=('NCHW', 'OIHW', 'NCHW') # default
out_chan=1
filter_shape=(L,L)

input_shape=np.array((N_points,N_symm,L,L),dtype=np.int) # NCHW input format



init_params, apply_layer =GeneralConvPeriodic(dim_nums, out_chan, filter_shape, ) # 


output_shape,params = init_params(rng,input_shape)

print(output_shape)

#print(params[0])



batch=np.ones((N_points,N_symm,L,L),dtype=np.float32)
#batch[0,0,0,0]=-1




z=apply_layer(params,batch)

print(z.shape)

#print(z[0,0,...].reshape(L,L))




