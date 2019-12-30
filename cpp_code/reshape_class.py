import numpy as np
import jax.numpy as jnp
from jax import jit
from jax.tree_util import tree_structure, tree_flatten, tree_unflatten



class NN_Tree(object):

    def __init__(self,pytree):

        flattened, self.tree = tree_flatten(pytree)

        self.shapes=[]
        self.sizes=np.zeros(len(flattened),dtype=int)

        for j,x in enumerate(flattened):

            self.shapes.append(x.shape)
            self.sizes[j]=x.size

        
    def flatten(self,data):
        return jnp.concatenate([x.flatten() for x in self.tree.flatten_up_to(data)])

    def unflatten(self,data):

        cum_size=0
        data_flattened=[]
        for j,size in enumerate(self.sizes):
            data_flattened.append( jnp.array(data[cum_size:cum_size+size].reshape(self.shapes[j])) )
            cum_size+=size

        return self.tree.unflatten(data_flattened)




"""
class Reshape(object):

    def __init__(self,params,NN_dtype='cpx'):

        neural_layers_indx=[]
        for i, tupl in enumerate(params):
            if len(tupl)>0:
                neural_layers_indx.append(i)

        shapes=[]
        dims=[]
        N_vars_per_layer=[]

        for indx in neural_layers_indx:
            for param in params[indx]: # loop real / imag
                for i,W in enumerate(param): # loop W / b
                    shapes.append(W.shape)
                    dims.append(W.size)
            N_vars_per_layer.append(i+1)



        self.shapes=np.array(shapes)
        self.dims=np.array(dims)
        
        self.N_vars_per_layer=np.array(N_vars_per_layer) 
        self.neural_layers_indx=np.asarray(neural_layers_indx) 
        self.N_layers = len(params) # includes nonlinearities


        self.Ndims=np.insert(np.cumsum(self.dims), 0, 0)

        #print(self.N_layers, self.N_vars_per_layer, self.neural_layers_indx)
        #exit()

    def _to_gradient_format(self, params_flat):
    
        _k=0
        _l=0
        params=[]
        for l in range(self.N_layers):
            s=()

            if l in self.neural_layers_indx:

                for i in range(2): # loop real / imag
                    t=()
                    #print(l, _l, self.N_vars_per_layer)
                    for j in range(self.N_vars_per_layer[_l]): # loop W / b
                        t+=(params_flat[self.Ndims[_k]:self.Ndims[_k+1]].reshape(self.shapes[_k]),)
                        _k+=1

                    s+=(t,)

                _l+=1
                
                params.append(s)
            
            else:
                params.append(())

        return params
        

    def _from_gradient_format(self, params):
        
        NN_params_flat=[]
        for indx in self.neural_layers_indx:
            for param in params[indx]: # loop real / imag
                for W in param: # loop W / b
                    NN_params_flat.append(W.flatten())
            
        return jnp.concatenate(NN_params_flat)


    @property
    def to_gradient_format(self):
        return jit(self._to_gradient_format)

    @property
    def from_gradient_format(self):
        return jit(self._from_gradient_format)
""" 
 
