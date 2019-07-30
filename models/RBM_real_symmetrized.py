from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, vmap, random, ops, partial
from jax.experimental.stax import relu

import numpy as np

seed=0
np.random.seed(seed)
np.random.RandomState(seed)
rng = random.PRNGKey(seed)



def create_NN(shape):

	init_value_Re=1E-3#1E-3 
	init_value_Im=1E-3#1E-1 
	
	W_fc_real = random.uniform(rng,shape=shape, minval=-init_value_Re, maxval=+init_value_Re)
	W_fc_imag = random.uniform(rng,shape=shape, minval=-init_value_Im, maxval=+init_value_Im)

	# W_fc_real2 = random.uniform(rng,shape=[2,2], minval=-init_value_Re, maxval=+init_value_Re)
	# W_fc_imag2 = random.uniform(rng,shape=[2,1], minval=-init_value_Im, maxval=+init_value_Im)

	architecture=[W_fc_real, W_fc_imag,]# W_fc_real2, W_fc_imag2]

	NN_shapes=np.array([W.shape for W in architecture])
	NN_dims=np.array([np.prod(shape) for shape in NN_shapes])

	return architecture, NN_dims, NN_shapes

#'''
@jit
def evaluate_NN(params,batch,cyclicities):

	# Cosh[a + I b] = Cos[b] Cosh[a] + I Sin[b] Sinh[a]
	Re_Ws = jnp.einsum('ij,...lj->...li',params[0], batch)
	Im_Ws = jnp.einsum('ij,...lj->...li',params[1], batch)

	Re = jnp.cos(Im_Ws)*jnp.cosh(Re_Ws)
	Im = jnp.sin(Im_Ws)*jnp.sinh(Re_Ws)

	#a_fc_real = tf.log( tf.sqrt( (tf.cos(Im_Ws)*tf.cosh(Re_Ws))**2 + (tf.sin(Im_Ws)*tf.sinh(Re_Ws))**2 )  )
	#a_fc_imag = tf.atan( tf.tan(Im_Ws)*tf.tanh(Re_Ws) )
	a_fc_real = 0.5*jnp.log(Re**2 + Im**2)
	a_fc_imag = jnp.arctan2(Im,Re)

	log_psi = jnp.sum(a_fc_real,axis=[1,2])
	phase_psi = jnp.sum(a_fc_imag,axis=[1,2])

	#return jnp.divide(log_psi,cyclicities), jnp.divide(phase_psi,cyclicities)
	#return log_psi*cyclicities, phase_psi*cyclicities
	return log_psi, phase_psi  #
'''



#@jit
def evaluate_NN(params,batch,cyclicities):

	# Cosh[a + I b] = Cos[b] Cosh[a] + I Sin[b] Sinh[a]
	Re_Ws = jnp.einsum('ij,...lj->...il',params[0], batch)
	Im_Ws = jnp.einsum('ij,...lj->...il',params[1], batch)

	a_fc_real = jnp.log(jnp.cosh(Re_Ws))
	a_fc_imag = jnp.log(jnp.cosh(Im_Ws))

	# a_fc_real = jnp.log(jnp.cosh(Re_Ws-Im_Ws))
	# a_fc_imag = jnp.log(jnp.cosh(Im_Ws)) #jnp.abs(Im_Ws) #

	# a_fc_real = relu(Re_Ws)
	# a_fc_imag = relu(Im_Ws)

	# a_fc_real = jnp.abs(Re_Ws)
	# a_fc_imag = jnp.abs(Im_Ws)

	# a_fc_real = -1.0+jnp.sqrt(1.0+Re_Ws**2)
	# a_fc_imag = -1.0+jnp.sqrt(1.0+Im_Ws**2)

	a_fc2_real = jnp.sum(a_fc_real,axis=[-1])
	a_fc2_imag = jnp.sum(a_fc_imag,axis=[-1])

	#####
	
	# log_psi   = relu( jnp.dot(a_fc2_real, params[2]).ravel() )
	phase_psi = relu( jnp.dot(a_fc2_imag, params[3]).ravel() )
	
	log_psi   = jnp.log(jnp.cosh( jnp.einsum('ij,...j->...',params[2], a_fc2_real) ) )
	#phase_psi = jnp.log(jnp.cosh( jnp.einsum('ij,...j->...',params[3], a_fc2_imag) ) )

	
	#log_psi = jnp.sum(a_fc_real,axis=[1])
	#phase_psi = jnp.sum(a_fc_imag,axis=[1])

	# log_psi = jnp.sum(a_fc_real,axis=[1,2])
	# phase_psi = jnp.sum(a_fc_imag,axis=[1,2])

	#return jnp.divide(log_psi,cyclicities), jnp.divide(phase_psi,cyclicities)
	return log_psi, phase_psi  #
'''

@jit
def loss_log_psi(params,batch,cyclicities):
	log_psi, phase_psi = evaluate_NN(params,batch,cyclicities,)	
	return jnp.sum(log_psi)

@jit
def loss_phase_psi(params,batch,cyclicities):
	log_psi, phase_psi = evaluate_NN(params,batch,cyclicities,)	
	return jnp.sum(phase_psi)





@jit
def compute_grad_log_psi(NN_params,batch,cyclicities,): 
	dlog_psi_s   = vmap(partial(grad(loss_log_psi),   NN_params))(batch,cyclicities, )
	dphase_psi_s = vmap(partial(grad(loss_phase_psi), NN_params))(batch,cyclicities, )
	
	N_MC_points=dlog_psi_s[0].shape[0]

	return jnp.concatenate( [(dlog_psi+1j*dphase_psi).reshape(N_MC_points,-1) for (dlog_psi,dphase_psi) in zip(dlog_psi_s,dphase_psi_s)], axis=1  )
	


@jit
def loss_energy_exact(NN_params,batch,params_dict,cyclicities):
	log_psi, phase_psi = evaluate_NN(NN_params,batch,cyclicities)
	return 2.0*jnp.sum(params_dict['abs_psi_2']*(log_psi*params_dict['E_diff'].real + phase_psi*params_dict['E_diff'].imag ))

@jit
def loss_energy_MC(NN_params,batch,params_dict,cyclicities):
	log_psi, phase_psi = evaluate_NN(NN_params,batch,cyclicities)
	return 2.0*jnp.sum(log_psi*params_dict['E_diff'].real + phase_psi*params_dict['E_diff'].imag)/params_dict['N_MC_points']



def reshape_to_gradient_format(gradient,NN_dims,NN_shapes):
	NN_params=[]
	Ndims=np.insert(np.cumsum(NN_dims), 0, 0)
	# loop over network architecture
	for j in range(NN_dims.shape[0]): 
		NN_params.append( gradient[Ndims[j]:Ndims[j+1]].reshape(NN_shapes[j]) )
		
	return NN_params
	


def reshape_from_gradient_format(NN_params,NN_dims,NN_shapes):
	return jnp.concatenate([params.ravel() for params in NN_params])



