import numpy as np
import jax.numpy as jnp
from jax import jit




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
    
 
