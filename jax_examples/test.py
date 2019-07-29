import jax.numpy as np
from jax import random
key = random.PRNGKey(0)

x = random.normal(key, (5000, 5000))

print(np.dot(x, x.T) / 2)  # fast!
print(np.dot(x, x.T) / 2)  # even faster!