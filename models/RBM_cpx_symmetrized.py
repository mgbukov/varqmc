from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, vmap, random, ops, partial

import numpy as np

seed=0
np.random.seed(seed)
np.random.RandomState(seed)
rng = random.PRNGKey(seed)



def create_NN(shape):

	init_value_Re=1E-3
	init_value_Im=1E-3 
	
	W_fc =   random.uniform(rng,shape=shape, minval=-init_value_Re, maxval=+init_value_Re, dtype=np.float64) \
		 +1j*random.uniform(rng,shape=shape, minval=-init_value_Re, maxval=+init_value_Re, dtype=np.float64)
	
	NN_shapes=np.array([W_fc.shape,])
	NN_dims=np.array([np.prod(shape) for shape in NN_shapes])

	return (W_fc,), NN_dims, NN_shapes


@jit
def evaluate_NN(params,batch):
	Ws = jnp.einsum('ij,...lj->...li',params[0], batch)
	a_fc = jnp.log(jnp.cosh(Ws))
	log_psi_cpx = jnp.sum(a_fc,axis=[1,2])
	
	return log_psi_cpx # log_psi_cpx.real, log_psi_cpx.imag


@jit
def loss_log_psi(params,batch):
	log_psi = evaluate_NN(params,batch)	
	return jnp.sum(log_psi)



@jit
def compute_grad_log_psi(NN_params,batch):
	dlog_psi_s   = vmap(partial(grad(loss_log_psi,holomorphic=True),   NN_params))(batch, )
	N_MC_points=dlog_psi_s[0].shape[0]

	"""
	dlog_psi_s   = vmap(partial(grad(loss_log_psi),   NN_params))(batch, )
	dphase_psi_s = vmap(partial(grad(loss_phase_psi), NN_params))(batch, )

	dlog_psi_s_cpx2   = vmap(partial(grad(loss_log_psi_cpx2,holomorphic=True),   (NN_params[0]+1j*NN_params[1],) ))(batch, )
	dlog_psi_s_cpx   = vmap(partial(grad(loss_log_psi_cpx,holomorphic=True),   NN_params ))(batch, )


	print((dlog_psi + 1j*dphase_psi).reshape(self.N_MC_points,-1)[0,...])
	print(dlog_psi_s_cpx2[0].reshape(self.N_MC_points,-1)[0,...])
	print((dlog_psi_s_cpx[0] - 1j*dlog_psi_s_cpx[1]).reshape(self.N_MC_points,-1)[0,...])
	"""

	return dlog_psi_s[0].reshape(N_MC_points,-1)


@jit
def loss_energy_exact(params,batch,params_dict):
	log_psi = evaluate_NN(params,batch)
	return 2.0*jnp.sum(params_dict['abs_psi_2']*(log_psi*params_dict['E_diff']).real)


@jit
def loss_energy_MC(params,batch,params_dict):
	log_psi, phase_psi = evaluate_NN(params,batch)
	return 2.0*jnp.sum( (log_psi*params_dict['E_diff']).real )/params_dict['N_MC_points']



def reshape_to_gradient_format(gradient,NN_dims,NN_shapes):
	return (gradient.reshape(NN_shapes[0]),)





