
from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, random, device_put 
#from jax.experimental.stax import relu, BatchNorm



def GeneralDeepComplex(W_shape):

    def init_fun(rng,):

        init_value_W=1E-1
        init_value_b=1E-1 

        W=np.zeros(W_shape,dtype=jnp.complex128)
        b=np.zeros(W_shape[0],dtype=jnp.complex128)

        W.real = random.uniform(rng,shape=W_shape, minval=-init_value_W, maxval=+init_value_W)
        W.imag = random.uniform(rng,shape=W_shape, minval=-init_value_W, maxval=+init_value_W)

        b.real = random.uniform(rng,shape=(W_shape[0],), minval=-init_value_b, maxval=+init_value_b)
        b.imag = random.uniform(rng,shape=(W_shape[0],), minval=-init_value_b, maxval=+init_value_b)
        
        W = device_put(W)
        b = device_put(b)

        params=[W,b]
        return params

    def apply_fun(params,inputs):
        return jnp.einsum('ij,lj->il',params, inputs)


    return init_fun, apply_fun


def GeneralDeep(W_shape):

    def init_fun(rng,):

        init_value_W=1E-1
        init_value_b=1E-1 

        W = random.uniform(rng,shape=W_shape, minval=-init_value_W, maxval=+init_value_W)
        b = random.uniform(rng,shape=(W_shape[0],), minval=-init_value_b, maxval=+init_value_b)
        
        params=[W,b]
        return params

    @jit
    def apply_fun(params,inputs):
        return jnp.einsum('ij,lj->il',params, inputs)


    return init_fun, apply_fun


@jit
def cpx_cosh(Re_a,Im_a):
    # Cosh[a + I b] = Cos[b] Cosh[a] + I Sin[b] Sinh[a]
        
    Re = jnp.cos(Im_a)*jnp.cosh(Re_a)
    Im = jnp.sin(Im_a)*jnp.sinh(Re_a)

    return Re, Im

@jit
def cpx_log_real(Re, Im,):
    #a_fc_real = tf.log( tf.sqrt( (tf.cos(Im_Ws)*tf.cosh(Re_Ws))**2 + (tf.sin(Im_Ws)*tf.sinh(Re_Ws))**2 )  )
    return 0.5*jnp.log(Re**2+Im**2)
  
@jit
def cpx_log_imag(Re, Im,):
   #a_fc_imag = tf.atan( tf.tan(Im_Ws)*tf.tanh(Re_Ws) )
    return jnp.arctan2(Im,Re)

