import numpy as np 
import jax
import jax.numpy as jnp
from jax import jit, lax

print('\n\n\n\n\n')

x=np.array([0,1,2,])

p_dict=dict(a=0.0)



# def foo(x):
# 	y = lax.tag(x ** 2, "y")
# 	z = y + 1
# 	return z


# value, intermediates = collect(foo)(2.)

# print(value, intermediates)




@jax.partial(jit, static_argnums=1)
def func(x,param):
	return 2.0*jnp.mean(x) + param


def _func2(x,param):
	return 2.0*jnp.mean(x) + param

@jax.partial(jit, static_argnums=1)
def func2(x,param):
	return _func2(x,param)


param=2.0
x=np.array([1.0,0.0])

print(func(x,param))
print(func2(x,param))



