from jax.config import config
config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
from jax.tree_util import tree_structure, tree_flatten, tree_unflatten
from jax.flatten_util import ravel_pytree



class NN_Tree(object):

    def __init__(self,pytree):
        
        flattened, self.tree = tree_flatten(pytree)
        raveled, self.unravel_pytree = ravel_pytree(pytree)
 
        self.N_varl_params=raveled.shape[0]

        self.shapes=[]
        self.sizes=np.zeros(len(flattened),dtype=int)

        for j,x in enumerate(flattened):

            self.shapes.append(x.shape)
            self.sizes[j]=x.size


    def ravel(self,data):
        return ravel_pytree(data)[0]

    def unravel(self,data):
        return self.unravel_pytree(data)

    def flatten(self,data):
        return self.tree.flatten_up_to(data)

    def unflatten(self,data):
        return self.tree.unflatten(data)


"""
class NN_Tree(object):

    def __init__(self,pytree):
        # assumes flattened elements are all arrays

        flattened, self.tree = tree_flatten(pytree)

        self.shapes=[]
        self.sizes=np.zeros(len(flattened),dtype=int)

        for j,x in enumerate(flattened):

            self.shapes.append(x.shape)
            self.sizes[j]=x.size

    @jit
    def flatten(self,data):
        # flatten maximally
        return jnp.concatenate([x.flatten() for x in self.tree.flatten_up_to(data)])

    @jit
    def unflatten(self,data):
        # restore shapes
        cum_size=0
        data_flattened=[]
        for j,size in enumerate(self.sizes):
            data_flattened.append( jnp.array(data[cum_size:cum_size+size].reshape(self.shapes[j])) )
            cum_size+=size

        # unflatten
        return self.tree.unflatten(data_flattened)

"""


