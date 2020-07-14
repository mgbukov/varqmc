from DNN_architectures_real import *
from DNN_architectures_common import *
import numpy as np 
from jax import random

np.random.seed(0)

rng = random.PRNGKey(0)

reduce_shape=(-1, 128, 12, 1)
output_shape=(-1, 12)

N=3
x=np.random.normal(size=(N*128,12))

result=Symmetric_Pool(x,reduce_shape, output_shape,)

print(result)


#init_fun, apply_fun=elementwise(Symmetric_Pool, **{'reduce_shape': reduce_shape, 'output_shape': output_shape}  ) # 

init_fun, apply_fun=elementwise(Symmetric_Pool, **dict(reduce_shape=reduce_shape, output_shape=output_shape)  ) # 

kwargs={}
result2=apply_fun(0, x, rng=rng, **kwargs)

print(result2)



