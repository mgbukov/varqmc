from jax.config import config
config.update("jax_enable_x64", True)
from jax import grad
import jax.numpy as np

def tanh(x):  # Define a function
  y = np.exp(-2.0 * x)
  return (1.0 - y) / (1.0 + y)

grad_tanh = grad(tanh)  # Obtain its gradient function
print("{:0.10f}".format(grad_tanh(1.0)))   # Evaluate it at x = 1.0
print("{:0.10f}".format(1.0/np.cosh(1.0)**2))


def predict(params, input_vec):
  assert input_vec.ndim == 1
  for W, b in params:
    output_vec = np.dot(W, input_vec) + b  # `input_vec` on the right-hand side!
    input_vec = np.tanh(output_vec)
  return output_vec


from jax import vmap
from functools import partial

predictions = vmap(partial(predict, params))(input_batch)
# or, alternatively
predictions = vmap(predict, in_axes=(None, 0))(params, input_batch)


per_example_gradients = vmap(partial(grad(loss), params))(inputs, targets)




