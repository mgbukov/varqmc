from jax import jit, partial
import jax.numpy as jnp
import numpy as np

@partial(jit,static_argnums=(1,))
def test(inputs,axis):
    mean=jnp.mean(inputs,axis=axis,keepdims=True)


input_shape=(8,16)
inputs=np.random.uniform(size=input_shape)

test(inputs,axis=(0,))
