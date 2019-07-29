from jax import jit, grad, vmap

symmetrized=True  # False  # 
if symmetrized:
	from models.RBM_real_symmetrized import *
	#from models.RBM_cpx_symmetrized import *
else:
	from models.RBM_real import *

from natural_grad import natural_gradient
from MC_lib import MC_sampler
from energy_lib import Energy_estimator

from jax.config import config
config.update("jax_enable_x64", True)
from jax.experimental import optimizers

from cpp_code import integer_to_spinstate

import numpy as np
from functools import partial

from data_analysis import data

import time
np.set_printoptions(threshold=np.inf)

# mpirun -n 4 python main.py 


L=4 # system size
E_est=Energy_estimator(L,symmetrized=symmetrized) # contains all of the physics
mode='exact' # 'MC'  #
real=1 #False

# training params
N_epochs=240 #500 


### MC sampler
N_MC_points=107 #1000 #
build_dict_args=(E_est.sign,L,E_est.J2)
MC_tool=MC_sampler(build_dict_args=build_dict_args)
# number of processors must fix MC sampling ratio
N_batch=N_MC_points//MC_tool.MC_sampler.world_size


### Neural network 
N_neurons=2
NN_params,NN_dims,NN_shapes=create_NN([N_neurons,L**2])


#NN_params=data_structure.load_weights()
#NN_params=jnp.array(NN_params).squeeze()

'''
NN_params=[ jnp.array(
			[[ 0.00580084, -0.07497671, -0.06789109, -0.01204818, -0.01274435, -0.10384455,
			  -0.10245288, -0.04374218, -0.08915268,  0.00177797, -0.04978558, -0.05587124,
			   0.07693333, -0.07984009, -0.01530174, -0.12691828],
			 [-0.21518007, -0.11389974, -0.20263269, -0.12619203, -0.08462431, -0.20707012,
			  -0.17650112, -0.19922243, -0.16967753, -0.18817129, -0.20458634, -0.08676682,
			  -0.11785064, -0.03068135, -0.04197004, -0.17183342]]

		  	),

			jnp.array(
			[[-0.00231961, -0.00895621,  0.06853679,  0.0069357,  -0.02742392,  0.00391063,
			   0.02143496,  0.07258342,  0.05560984, -0.02823264,  0.02835682, -0.00693428,
			   0.01221359,  0.05616201, -0.02058408,  0.0281348 ],
			 [-0.1803375,  -0.12494415, -0.17192932, -0.11660786, -0.12339658, -0.18764674,
			  -0.12699878, -0.17132069, -0.15470591, -0.13720351, -0.18281736, -0.10516087,
			  -0.10887956, -0.07979148, -0.12417864, -0.16585421]]
		  	)
		 ]
'''



### Optimizer params
optimizer='RK' #'adam' #  'NG' # 
# initiaize natural gradient class
if real:
	N_varl_params=NN_dims.sum()
else:
	N_varl_params=NN_dims[0]
	
NG=natural_gradient(N_MC_points,N_varl_params,real=real)


# jax optimizer
if optimizer=='NG':
	step_size=1E-2
	opt_init, opt_update, get_params = optimizers.sgd(step_size=step_size)
	opt_state = opt_init(NN_params)

elif optimizer=='adam':
	step_size=1E-3
	opt_init, opt_update, get_params = optimizers.adam(step_size=step_size, b1=0.9, b2=0.99, eps=1e-08)
	if mode=='exact':
		grad_func=jit(grad(loss_energy_exact))
	elif mode=='MC':
		grad_func=jit(grad(loss_energy_MC))
	opt_state = opt_init(NN_params)

elif optimizer=='RK':
	NG.init_RK_params(1E-2)



@jit
def update_params(i,opt_state,grads):
	# pass parameters to optimizer
	params = get_params(opt_state)
	return opt_update(i, grads, opt_state)

### Energy estimator
E_est.init_global_params(N_batch,MC_tool.MC_sampler.world_size)
N_features=E_est.N_sites*E_est.N_symms

### initialize MC sampler variables
MC_tool.init_global_vars(L,N_batch,E_est.N_symms,E_est.basis_type)


### initialize data class
model_params=dict(model='RBMcpx',
				  mode=mode, 
				  symm=int(symmetrized),
				  L=L,
				  J2=E_est.J2,
				  opt=optimizer,
				  NNstrct=tuple(tuple(shape) for shape in NN_shapes),
				  epochs=N_epochs,
				  MCpts=N_MC_points,
				  
				)
extra_label=''#'-unique_configs'
data_structure=data(model_params,N_MC_points,N_epochs,extra_label=extra_label)



#RK_steps=np.zeros(N_epochs,dtype=np.float64)
#extra_name=''#'_S=0'


### train network
if mode=='exact':
	ints_ket, index, inv_index, count=E_est.get_exact_kets()
	MC_tool.ints_ket=ints_ket
	integer_to_spinstate(ints_ket, MC_tool.spinstates_ket, MC_tool.cyclicities_ket, N_features)


for epoch in range(N_epochs): 

	
	ti=time.time()

	##### MC sample #####
	if mode=='exact':
		MC_tool.exact(ints_ket,NN_params ,N_neurons, evaluate_NN=evaluate_NN)
		#MC_tool.exact(ints_ket,tuple([NN_params[0].real._value,NN_params[0].imag._value]) ,N_neurons)
	elif mode=='MC':
		MC_tool.sample(tuple([W._value for W in NN_params]) ,N_neurons)

	##### compute local energies #####
	E_est.compute_local_energy(N_batch,evaluate_NN,NN_params,MC_tool.ints_ket,MC_tool.mod_kets,MC_tool.phase_kets,MC_tool.MC_sampler)
		
	if mode=='exact':
		psi = MC_tool.mod_kets*np.exp(+1j*MC_tool.phase_kets)/np.linalg.norm(MC_tool.mod_kets[inv_index])
		abs_psi_2=count*np.abs(psi)**2
		params_dict=dict(abs_psi_2=abs_psi_2,)
		overlap=np.abs(psi[inv_index].dot(E_est.psi_GS_exact))**2
	elif mode=='MC':
		params_dict=dict(N_MC_points=N_MC_points)
	
	Eloc_mean, Eloc_std, E_diff_real, E_diff_imag = E_est.process_local_energies(mode=mode,params_dict=params_dict)
	E_MC_var=Eloc_std.real/np.sqrt(N_MC_points)
	params_dict['E_diff']=E_diff_real+1j*E_diff_imag
	
	
	# check SU(2) conservation
	E_est.compute_local_energy(N_batch,evaluate_NN,NN_params,MC_tool.ints_ket,MC_tool.mod_kets,MC_tool.phase_kets,MC_tool.MC_sampler,SdotS=True)
	SdotSloc_mean, SdotS_std, SdotS_diff_real, SdotS_diff_imag = E_est.process_local_energies(mode=mode,params_dict=params_dict,SdotS=True)
	SdotS_MC_var=SdotS_std.real/np.sqrt(N_MC_points)
	

	##### total batch
	# combine results from all cores
	MC_tool.all_gather()
	# reshape
	if symmetrized:
		MC_tool.spinstates_ket_tot=MC_tool.spinstates_ket_tot.reshape(-1,E_est.N_symms,E_est.N_sites)
	else:
		MC_tool.spinstates_ket_tot=MC_tool.spinstates_ket_tot.reshape(-1,E_est.N_sites)
	batch=MC_tool.spinstates_ket_tot


	##### check c++ and python DNN evaluation
	if epoch==0:
		MC_tool.check_consistency(evaluate_NN,NN_params)

		if mode=='exact':
			np.testing.assert_allclose(Eloc_mean.real, E_est.H.expt_value(psi[inv_index]))

		if MC_tool.MC_sampler.world_rank==0:
			print('cpp/python consistency check passed!\n')

		# data_structure._create_name('-initial_state'+extra_name)
		# np.savetxt(data_structure.file_name+'.txt', np.c_[E_est.basis.states, psi[inv_index].real , psi[inv_index].imag], fmt=['%d','%0.16f','%0.16f'])

		

	#####
	if MC_tool.MC_sampler.world_rank==0:
		print("epoch {0:d}:".format(epoch))
		print("E={0:0.14f} and SS={1:0.14f}.".format(Eloc_mean.real, SdotSloc_mean.real ), E_MC_var,  	)
		#print("overlaps",overlap,)
	#exit()
	
	#RK_steps[epoch]=NG.RK_step_size


	
	if epoch<N_epochs-1:


		if optimizer=='RK':
			# compute updated NN parameters
			if real:
				NN_params=NG.Runge_Kutta(NN_params,batch,MC_tool.cyclicities_ket,params_dict,mode,NN_dims,NN_shapes,
											compute_grad_log_psi_real,reshape_to_gradient_format,reshape_from_gradient_format,real)
			else:
				NN_params=NG.Runge_Kutta(NN_params,batch,MC_tool.cyclicities_ket,params_dict,mode,NN_dims,NN_shapes,
											compute_grad_log_psi,reshape_to_gradient_format,reshape_from_gradient_format,real)
			# reshape weights
			NN_params=reshape_to_gradient_format(NN_params,NN_dims,NN_shapes,real=real)

			loss=NG.max_grads

		else:
			##### compute gradients
			if optimizer=='NG':
				# compute NN gradients
				if real:
					NG.dlog_psi[:]=compute_grad_log_psi_real(NN_params,batch,MC_tool.cyclicities_ket)
				else:
					NG.dlog_psi[:]=compute_grad_log_psi(NN_params,batch,MC_tool.cyclicities_ket)
				# compute enatural gradients
				grads=NG.compute(params_dict,mode=mode)
				# reshape gradient in jax opt format
				grads=reshape_to_gradient_format(grads,NN_dims,NN_shapes,real)
				loss=NG.max_grads
				
			elif optimizer=='adam':
				grads=grad_func(NN_params,batch,params_dict,MC_tool.cyclicities_ket)
				loss=[np.max(np.real(grads)),0.0]

			##### apply gradients
			opt_state = update_params(epoch, opt_state, grads)
			NG.update_params() # update NG params
			NN_params=get_params(opt_state)

		##### compute loss
		r2=NG.cost_function(mode=mode,params_dict=params_dict).real
	

		print("world {0:d} calculation took {1:0.4f}secs.\n".format(MC_tool.MC_sampler.world_rank,time.time()-ti) )
		#exit()

			
		

	##### store data
	if MC_tool.MC_sampler.world_rank==0:
		data_structure.excess_energy[epoch,0]=(Eloc_mean - E_est.E_GS)/E_est.N_sites
		data_structure.excess_energy[epoch,1]=E_MC_var/E_est.N_sites
		data_structure.SdotS[epoch,0]=SdotSloc_mean/(0.5*E_est.N_sites*(0.5*E_est.N_sites+1.0))
		data_structure.SdotS[epoch,1]=SdotS_MC_var/(0.5*E_est.N_sites*(0.5*E_est.N_sites+1.0))
		data_structure.loss[epoch,:]=loss
		data_structure.r2[epoch]=r2
		data_structure.phase_psi[epoch,:]=MC_tool.phase_kets_tot
		data_structure.mod_psi[epoch,:]=MC_tool.mod_kets_tot


#close MPI 
MC_tool.MC_sampler.mpi_close()


# print(NN_params[0])
# print(NN_params[1])


# save data
#data_structure.save(NN_params=NN_params)
#exit()

# data_structure._create_name('-final_state'+extra_name)
# np.savetxt(data_structure.file_name+'.txt', np.c_[E_est.basis.states, psi[inv_index].real , psi[inv_index].imag], fmt=['%d','%0.16f','%0.16f'])

# data_structure._create_name('-RK-steps'+extra_name)
# np.savetxt(data_structure.file_name+'.txt', np.c_[RK_steps], fmt=['%0.16f'])



data_structure.compute_phase_hist()
data_structure.plot(save=0)



