from jax.config import config
config.update("jax_enable_x64", True)

from jax import grad
import jax.numpy as jnp


def cosh_real_faulty(x):
    Re_x, Im_x = x
    # Cosh[a + I b] = Cos[b] Cosh[a] + I Sin[b] Sinh[a]
    Re_z = jnp.cos(Im_x)*jnp.cosh(Re_x)
    Im_z = jnp.sin(Im_x)*jnp.sinh(Re_x)
    return Re_z+Im_z


def cosh_cpx(x):
    x_real, x_imag = x
    z=jnp.cosh(x_real+1j*x_imag)
    return z.real+z.imag


def exact_grad(x):
	x_real, x_imag = x
	z=jnp.sinh(x_real+1j*x_imag)
	return z.real+z.imag

def exact_grad2(x):
    Re_x, Im_x = x
    # Sinh[a + I b] = Sin[a] Cosh[b] + I Cos[a] Sinh[b]
    Re_z = jnp.sin(Re_x)*jnp.cosh(Im_x)
    Im_z = jnp.cos(Re_x)*jnp.sinh(Im_x)
    return Re_z+Im_z

d_cosh_cpx=grad(cosh_cpx, holomorphic=False)
d_cosh_real_faulty=grad(cosh_real_faulty, holomorphic=False)


z=(0.001, 0.002)
print('numpy       {0:0.15f}'.format(d_cosh_cpx(z)[0]) )
print('custom      {0:0.15f}'.format(d_cosh_real_faulty(z)[0]) )
print('exact cosh  {0:0.15f}'.format(exact_grad(z)) )
print('exact cosh2 {0:0.15f}'.format(exact_grad2(z)) )


print()
z=(0.001, 0.001)
print('numpy       {0:0.15f}'.format(d_cosh_cpx(z)[0]) )
print('custom      {0:0.15f}'.format(d_cosh_real_faulty(z)[0]) )
print('exact cosh  {0:0.15f}'.format(exact_grad(z)) )
print('exact cosh2 {0:0.15f}'.format(exact_grad2(z)) )


print()
z=(0.01, 0.01)
print('numpy       {0:0.15f}'.format(d_cosh_cpx(z)[0]) )
print('custom      {0:0.15f}'.format(d_cosh_real_faulty(z)[0]) )
print('exact cosh  {0:0.15f}'.format(exact_grad(z)) )
print('exact cosh2 {0:0.15f}'.format(exact_grad2(z)) )