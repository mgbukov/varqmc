import numpy as np 
import jax.numpy as jnp
from jax import jit

x=np.array([0,1,2,])

p_dict=dict(a=0.0)



def func(x,**kwargs):
	return jnp.mean(x)+a


def apply_func(x,p_dict):
	return func(x,**kwargs)


def call_func(x):
	return apply_func(x)



# print(func(x,p_dict))
# p_dict=dict(a=1.0)
# print(func(x,p_dict))


#########################################
from jax.tree_util import tree_structure, tree_flatten, tree_unflatten

data=((jnp.array([[0,2,4],[1,3,5]]),) , (jnp.array([3.14, 2.71]),), )

data_flat, tree = tree_flatten(data)

print(tree.flatten_up_to(data))


data_new=tree.unflatten(data_flat)

print(data)
print(data_flat)
print(data_new)


