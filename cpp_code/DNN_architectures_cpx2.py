from jax.config import config
config.update("jax_enable_x64", True)
from jax import jit, lax, random, device_put #, disable_jit

import functools
import itertools

import jax.numpy as jnp
import numpy as np

from jax.nn.initializers import glorot_normal, normal, ones, zeros



def GeneralDenseComplex(in_chan, out_chan, filter_size, ignore_b=False, init_value_W=1E-2, init_value_b=1E-2):
	
	norm=jnp.sqrt(filter_size*(in_chan+out_chan))

	def init_fun(rng,input_shape):

		if out_chan==1:
			output_shape=(input_shape[0],)
			W_shape=(input_shape[1],)
			b_shape=(1,)
		else:
			output_shape=(input_shape[0],out_chan)   
			W_shape=(input_shape[1],out_chan)
			b_shape=(output_shape[1],)	

		rng_real, rng_imag = random.split(rng)

		W_real = random.uniform(rng_real,shape=W_shape, minval=-init_value_W, maxval=+init_value_W)
		W_imag = random.uniform(rng_imag,shape=W_shape, minval=-init_value_W, maxval=+init_value_W)

		W_real/=norm # see apply func
		W_imag/=norm

		if not ignore_b:
			rng_real, k1 = random.split(rng_real)
			rng_imag, k2 = random.split(rng_imag)

			b_real = random.uniform(k1,shape=b_shape, minval=-init_value_b, maxval=+init_value_b)
			b_imag = random.uniform(k2,shape=b_shape, minval=-init_value_b, maxval=+init_value_b)

			params=(W_real,W_imag,b_real,b_imag)
		
		else:
			params=(W_real,W_imag,)
		
		return output_shape, params

	def apply_fun(params,inputs, **kwargs):

		# read-off params
		W_real=params[0]
		W_imag=params[1]

		if isinstance(inputs, tuple):
			inputs_real, inputs_imag = inputs
		else:
			inputs_real = inputs
			inputs_imag = None

		z_real = jnp.dot(inputs_real,W_real) 
		z_imag = jnp.dot(inputs_real,W_imag)

		if inputs_imag is not None:
			z_real -= jnp.dot(inputs_imag,W_imag)
			z_imag += jnp.dot(inputs_imag,W_real)

		if not ignore_b:
			# add bias
			z_real += params[2]
			z_imag += params[3]
	   
		return z_real, z_imag

	return init_fun, apply_fun



##########################
# nonlinearities

@jit
def poly_cpx(x):
	x_real, x_imag = x
	Re = 0.5*x_real**2 - 0.0833333*x_real**4 + 0.0222222*x_real**6 - 0.5*x_imag**2 + 0.5*x_real**2*x_imag**2 - 0.333333*x_real**4*x_imag**2 - 0.0833333*x_imag**4 + 0.333333*x_real**2*x_imag**4 - 0.0222222*x_imag**6
	Im = x_real*x_imag - 0.333333*x_real**3*x_imag + 0.133333*x_real**5*x_imag + 0.333333*x_real*x_imag**3 - 0.444444*x_real**3*x_imag**3 + 0.133333*x_real*x_imag**5
	return Re, Im


@jit
def poly_real(x):
	x_real, x_imag = x
	Re_z = 0.5*x_real**2 - 0.0833333*x_real**4 + 0.0222222*x_real**6 - 0.5*x_imag**2 + 0.5*x_real**2*x_imag**2 - 0.333333*x_real**4*x_imag**2 - 0.0833333*x_imag**4 + 0.333333*x_real**2*x_imag**4 - 0.0222222*x_imag**6
	return Re_z


@jit
def poly_imag(x):
	x_real, x_imag = x
	Im_z = x_real*x_imag - 0.333333*x_real**3*x_imag + 0.133333*x_real**5*x_imag + 0.333333*x_real*x_imag**3 - 0.444444*x_real**3*x_imag**3 + 0.133333*x_real*x_imag**5
	return Im_z


##########################


@jit
def cosh_cpx(Re_a,Im_a):
	# Cosh[a + I b] = Cos[b] Cosh[a] + I Sin[b] Sinh[a]
		
	Re = jnp.cos(Im_a)*jnp.cosh(Re_a)
	Im = jnp.sin(Im_a)*jnp.sinh(Re_a)

	return Re, Im

@jit
def log_real(Re, Im,):
	#a_fc_real = tf.log( tf.sqrt( (tf.cos(Im_Ws)*tf.cosh(Re_Ws))**2 + (tf.sin(Im_Ws)*tf.sinh(Re_Ws))**2 )  )
	return 0.5*jnp.log(Re**2+Im**2)
  
@jit
def log_imag(Re, Im,):
	#a_fc_imag = tf.atan( tf.tan(Im_Ws)*tf.tanh(Re_Ws) )
	return jnp.arctan2(Im,Re)


@jit
def logcosh_cpx(x):
	x_real, x_imag = x
	Re, Im  = cosh_cpx(x_real, x_imag)
	Re_z = log_real(Re, Im, ) 
	Im_z = log_imag(Re, Im, )
	return Re_z, Im_z

@jit
def logcosh_real(Ws):
	Re_Ws, Im_Ws = Ws
	Re, Im  = cosh_cpx(Re_Ws, Im_Ws)
	Re_z = log_real(Re, Im, )
	return Re_z


@jit
def logcosh_imag(Ws):
	Re_Ws, Im_Ws = Ws
	Re, Im  = cosh_cpx(Re_Ws, Im_Ws)
	Im_z = log_imag(Re, Im, )
	return Im_z


##########################


def symmetric_pool_cpx(x,reduce_shape, output_shape,):
	Re_z, Im_z = x
	# symmetrize
	Re_z = jnp.sum(Re_z.reshape(reduce_shape,order='C') / jnp.sqrt(reduce_shape[1]+reduce_shape[3]),  axis=[1,3])
	Im_z = jnp.sum(Im_z.reshape(reduce_shape,order='C') / jnp.sqrt(reduce_shape[1]+reduce_shape[3]),  axis=[1,3])
	# sum over hidden neurons
	Re_z = jnp.sum(Re_z.reshape(output_shape) / jnp.sqrt(output_shape[1]), axis=[1,])
	Im_z = jnp.sum(Im_z.reshape(output_shape) / jnp.sqrt(output_shape[1]), axis=[1,])
	return (Re_z,Im_z)


def symmetrize_cpx(x, reduce_shape,):
	# symmetrize
	Re_z, Im_z = x
	Re_z = jnp.sum(Re_z.reshape(reduce_shape,order='C') / jnp.sqrt(reduce_shape[1]+reduce_shape[3]),  axis=[1,3])
	Im_z = jnp.sum(Im_z.reshape(reduce_shape,order='C') / jnp.sqrt(reduce_shape[1]+reduce_shape[3]),  axis=[1,3])
	return (Re_z,Im_z)

def uniform_pool_cpx(x, output_shape,):
	Re_z, Im_z = x
	# sum over hidden neurons
	Re_z = jnp.sum(Re_z.reshape(output_shape) / jnp.sqrt(output_shape[1]), axis=[1,])
	Im_z = jnp.sum(Im_z.reshape(output_shape) / jnp.sqrt(output_shape[1]), axis=[1,])
	return (Re_z,Im_z)



def symmetric_pool_Re(x,reduce_shape, output_shape,):
	Re_z, Im_z = x
	# symmetrize
	Re_z = jnp.sum(Re_z.reshape(reduce_shape,order='C') / jnp.sqrt(reduce_shape[1]+reduce_shape[3]),  axis=[1,3])
	# sum over hidden neurons
	Re_z = jnp.sum(Re_z.reshape(output_shape) / jnp.sqrt(output_shape[1]), axis=[1,])
	return Re_z


def symmetrize_Re(x, reduce_shape,):
	# symmetrize
	Re_z, Im_z = x
	Re_z = jnp.sum(Re_z.reshape(reduce_shape,order='C') / jnp.sqrt(reduce_shape[1]+reduce_shape[3]),  axis=[1,3])
	return Re_z

def uniform_pool_Re(x, output_shape,):
	Re_z, Im_z = x
	# sum over hidden neurons
	Re_z = jnp.sum(Re_z.reshape(output_shape) / jnp.sqrt(output_shape[1]), axis=[1,])
	return Re_z


###############


def RegularizationComplex(a=8.0,center=True, b_init=zeros, dtype=np.float64,):
	"""Layer construction function for a batch normalization layer."""
	_b_init = lambda rng, shape: b_init(rng, shape, dtype) if center else ()

	def init_fun(rng, input_shape):
		b_shape = (1,)
		b = _b_init(rng, b_shape)
		output_shape=(-1,)
		return output_shape, (b,)

	def apply_fun(params, x, **kwargs):
		b,   = params
		Re_z, Im_z = x
		# regularize output
		log_psi=a*jnp.tanh((Re_z-b)/a) + b
		return log_psi, Im_z
		
	return init_fun, apply_fun






