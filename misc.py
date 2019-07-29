
'''
# Stot=0 state for (16,2), (16,2) complex RBM w/o Marshall rule
NN_params=[ jnp.array(
			[[ 4.78676599e-02, -3.54945473e-02, -1.70757310e-02,  8.44422923e-03,
			   2.72073711e-02,  5.81558490e-02,  1.58043149e-03, -1.63857837e-01,
			  -1.62884294e-02,  1.15565306e-01, -9.70395325e-03, -5.15075363e-02,
			  -8.00275918e-03, -8.77732443e-03,  7.48335874e-03,  5.05925163e-02],
			 [ 1.06181094e-03, -1.07898835e-04,  1.28978782e-02, -3.21756713e-03,
			  -2.72668444e-03,  1.52177623e-02, -3.15395549e-02, -2.31671813e-02,
			   3.94137117e-02,  2.68391546e-02, -5.73053202e-02, -2.75380610e-03,
			  -2.77049728e-02,  1.26984523e-02,  1.42431805e-01,  2.38991188e-02]]
		  	),

			jnp.array(
			[[ 0.09239515,  0.07260364,  0.04379739,  0.06234778, -0.04584127, -0.07231739,
			  -0.02327265, -0.34972727,  0.04988223,  0.23388068,  0.0585562,   0.07872041,
			  -0.02303791, -0.06475063, -0.03482387, -0.07152269],
			 [-0.08013041,  0.02556722, -0.03830734,  0.00974384,  0.07350479,  0.02523769,
			   0.00823282, -0.01437445, -0.01420937,  0.00101997, -0.21374431,  0.06182008,
			  -0.00401305, -0.01908299,  0.21892955, -0.03586252]]
		  	)
		 ]
'''







W_fc_real,W_fc_imag=NN_params[0],NN_params[1]
W_fc = W_fc_real+1j*W_fc_imag 
states=MC_tool.spinstates_ket.reshape(N_batch,E_est.N_symms,E_est.N_sites,order='C')

tanh_Ws=np.tanh(np.einsum('il,snl->isn',W_fc,states))
O = np.einsum('isn,snj->sij', tanh_Ws,states)
O_flattened = O.reshape(N_batch,N_neurons*E_est.N_sites)

W_Re_grad =  + 2.0*np.einsum('s,sij->ij', np.abs(psi)**2*count*E_diff_real, O.real) \
 			 + 2.0*np.einsum('s,sij->ij', np.abs(psi)**2*count*E_diff_imag, O.imag)

W_Im_grad =  + 2.0*np.einsum('s,sij->ij', np.abs(psi)**2*count*E_diff_imag, O.real) \
 			 - 2.0*np.einsum('s,sij->ij', np.abs(psi)**2*count*E_diff_real, O.imag) 


# jax
training_gradient_fun = jit(grad(loss))
loss_args=(psi,count,E_diff_real,E_diff_imag)
dNN_params = training_gradient_fun(NN_params, MC_tool.spinstates_ket.reshape(-1,E_est.N_symms,E_est.N_sites), *loss_args)




O_expt =  np.einsum('s,sk->k',     np.abs(psi)**2*count, O_flattened)
OO_expt = np.einsum('s,sk,sl->kl', np.abs(psi)**2*count, O_flattened.conj(), O_flattened)
S =  OO_expt - np.einsum('k,l->kl',O_expt.conj(),O_expt)
#S += delta*np.diag(np.diag(S))



per_example_gradients = vmap(partial(grad(loss), NN_params))(MC_tool.spinstates_ket.reshape(-1,E_est.N_symms,E_est.N_sites), *loss_args)
per_example_gradients = np.array(per_example_gradients)


# print(W_Re_grad)
# print(dNN_params[0])
# print(np.sum(per_example_gradients[0],axis=0))
# exit()


per_example_dlog_psi = vmap(partial(grad(loss_log_psi), NN_params))(MC_tool.spinstates_ket.reshape(-1,E_est.N_symms,E_est.N_sites), )
per_example_dphase_psi = vmap(partial(grad(loss_phase_psi), NN_params))(MC_tool.spinstates_ket.reshape(-1,E_est.N_symms,E_est.N_sites), )

per_example_dlog_psi=np.array(per_example_dlog_psi)
per_example_dphase_psi=np.array(per_example_dphase_psi)




#O_jax=(per_example_dlog_psi+1j*per_example_dphase_psi).reshape(2,107,32)
#O_jax=np.concatenate(O_jax,axis=1)#

O_jax=(per_example_dlog_psi+1j*per_example_dphase_psi)[0].reshape(107,32)


O_jax_expt=np.dot(np.abs(psi)**2*count,O_jax)
OO_jax_expt = np.einsum('s,sk,sl->kl',np.abs(psi)**2*count,O_jax.conj(), O_jax)
S_jax =  OO_jax_expt - np.einsum('k,l->kl',O_jax_expt.conj(),O_jax_expt)

E_diff=E_diff_real+1j*E_diff_imag
F = np.einsum('s,sk->k', np.abs(psi)**2*count*E_diff, O_flattened.conj())
F_jax = np.einsum('s,sk->k', np.abs(psi)**2*count*E_diff, O_jax.conj())




W=W_fc.reshape(-1)
alpha=W
#alpha_jax=np.concatenate([W.real,W.imag])


Sa=S.dot(alpha)
Sa_jax=S_jax.dot(alpha)

print(Sa)
print( Sa_jax )