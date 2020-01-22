from jax.config import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
from jax import jit, grad, random, device_put, partial



class My_class():

	def __init__(self,):
		pass


	def _my_func(self,x,params):
		return x*jnp.ones(shape=params)


	@jax.partial(jit, static_argnums=(0,2))
	def my_func(self,x,params):
		return self._my_func(x,params)


obj=My_class()

x=2.0
print( obj.my_func(x,(2,)) )
print( obj.my_func(x,(4,)) )

