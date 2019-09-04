from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, vmap, random, ops, partial
from jax.experimental.stax import relu, BatchNorm

import numpy as np

seed=1
np.random.seed(seed)
np.random.RandomState(seed)
rng = random.PRNGKey(seed)

@jit
def melu(x):
	return jnp.where(jnp.abs(x)>1.0, jnp.abs(x)-0.5, 0.5*x**2)
 

'''
def create_NN(shape):

	init_value_W=1E-1 
	init_value_b=1E-1
	
	W_fc_base = random.uniform(rng,shape=shape[0], minval=-init_value_W, maxval=+init_value_W)
	
	W_fc_log_psi = random.uniform(rng,shape=shape[1], minval=-init_value_W,   maxval=+init_value_W)
	W_fc_phase   = random.uniform(rng,shape=shape[2], minval=-init_value_W, maxval=+init_value_W)

	b_fc_log_psi = random.uniform(rng,shape=(shape[1][1],), minval=-init_value_b, maxval=+init_value_b)
	b_fc_phase   = random.uniform(rng,shape=(shape[2][1],), minval=-init_value_b, maxval=+init_value_b)

	
	architecture=[W_fc_base, W_fc_log_psi, W_fc_phase, b_fc_log_psi, b_fc_phase]

	return architecture



@jit
def evaluate_NN(params,batch):

	### common layer
	Ws = jnp.einsum('ij,...lj->...il',params[0], batch)
	# nonlinearity
	#a_fc_base = jnp.log(jnp.cosh(Ws))
	a_fc_base = melu(Ws) 
	#a_fc_base = relu(Ws)
	# symmetrize
	a_fc_base = jnp.sum(a_fc_base, axis=[-1])

	
	### log_psi head
	a_fc_log_psi = jnp.dot(a_fc_base, params[1]) + params[3]
	#log_psi = jnp.log(jnp.cosh(a_fc_log_psi))
	log_psi = melu(a_fc_log_psi) 
	#log_psi = relu(a_fc_log_psi)

	### phase head
	a_fc_phase =jnp.dot(a_fc_base, params[2]) + params[4]
	#phase_psi = jnp.log(jnp.cosh(a_fc_phase))
	phase_psi = melu(a_fc_phase) 
	#phase_psi = relu(a_fc_phase)

	
	log_psi = jnp.sum(log_psi, axis=[1])#/log_psi.shape[0]
	phase_psi = jnp.sum(phase_psi, axis=[1])#/phase_psi.shape[0]	


	return log_psi, phase_psi  #

'''

def create_NN(shape):

	init_value_Re=1E-1#1E-3 
	init_value_Im=1E-1#1E-1 
	
	W_fc_real = random.uniform(rng,shape=shape, minval=-init_value_Re, maxval=+init_value_Re)
	W_fc_imag = random.uniform(rng,shape=shape, minval=-init_value_Im, maxval=+init_value_Im)

	# W_fc_real2 = random.uniform(rng,shape=[2,2], minval=-init_value_Re, maxval=+init_value_Re)
	# W_fc_imag2 = random.uniform(rng,shape=[2,1], minval=-init_value_Im, maxval=+init_value_Im)

	architecture=[W_fc_real, W_fc_imag,]# W_fc_real2, W_fc_imag2]

	return architecture



@jit
def evaluate_NN(params,batch):

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

	#print(jnp.abs(a_fc_real).max(), jnp.abs(a_fc_real).min(), jnp.abs(log_psi).max(), jnp.abs(log_psi).min(), jnp.abs(phase_psi).max(), jnp.abs(phase_psi).min() )

	return log_psi, phase_psi




@jit
def loss_log_psi(params,batch,):
	log_psi, phase_psi = evaluate_NN(params,batch,)	
	return jnp.sum(log_psi)

@jit
def loss_phase_psi(params,batch,):
	log_psi, phase_psi = evaluate_NN(params,batch,)	
	return jnp.sum(phase_psi)





@jit
def compute_grad_log_psi(NN_params,batch,): 
	dlog_psi_s   = vmap(partial(grad(loss_log_psi),   NN_params))(batch, )
	dphase_psi_s = vmap(partial(grad(loss_phase_psi), NN_params))(batch, )
	
	N_MC_points=dlog_psi_s[0].shape[0]

	return jnp.concatenate( [(dlog_psi+1j*dphase_psi).reshape(N_MC_points,-1) for (dlog_psi,dphase_psi) in zip(dlog_psi_s,dphase_psi_s)], axis=1  )
	


@jit
def loss_energy_exact(NN_params,batch,params_dict):
	log_psi, phase_psi = evaluate_NN(NN_params,batch,)
	return 2.0*jnp.sum(params_dict['abs_psi_2']*(log_psi*params_dict['E_diff'].real + phase_psi*params_dict['E_diff'].imag ))

@jit
def loss_energy_MC(NN_params,batch,params_dict,):
	log_psi, phase_psi = evaluate_NN(NN_params,batch,)
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



