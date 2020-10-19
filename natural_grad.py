from jax.config import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, device_put

from jax.experimental.optimizers import make_schedule

from mpi4py import MPI
import numpy as np
from scipy.sparse.linalg import cg #,cgs,bicg,bicgstab
#from cg import cg
from scipy.linalg import eigh,eig,inv,pinv,pinvh

import pickle, time



class natural_gradient():

	def __init__(self,comm,compute_grad_log_psi, NN_dtype, NN_Tree, TDVP_opt, mode='MC', RK=False, adaptive_SR_cutoff=False, hessian=None, hessian2=None):
				 
		self.comm=comm
		self.logfile=None
		self.NN_dtype=NN_dtype

		self.NN_Tree=NN_Tree
		self.mode=mode
		self.RK=RK
		self.adaptive_SR_cutoff=adaptive_SR_cutoff

	
		self.compute_grad_log_psi=compute_grad_log_psi
		self.hessian=hessian
		self.hessian2=hessian2


		self.TDVP_opt = TDVP_opt # 'svd' # 'inv' # 'cg' #

		self.dE=0.0
		self.curvature=0.0
		

		self.debug_mode=True
		
		# CG params
		self.check_on=False # toggles S-matrix checks
		self.cg_maxiter=1E4
		self.tol=1E-7 # CG tolerance

		self.step_size=1.0

		if self.RK:
			self.delta=0.0 # NG S matrix regularization strength
			self.grad_clip=1E4
		else:
			self.delta=100.0 # S-matrix regularizer
			self.grad_clip=1E4 #50.0


		self.SNR_weight_sum_exact=1.0
		self.SNR_weight_sum_gauss=1.0


		self.iteration=0
		

	def init_global_variables(self,N_MC_points,N_batch,N_varl_params_vec,n_iter,N_minibatches,):


		self.N_batch=N_batch
		self.N_MC_points=N_MC_points
		self.N_varl_params=np.sum(N_varl_params_vec)
		self.N_varl_params_vec=N_varl_params_vec
			
		self.N_minibatches=N_minibatches
		self.minibatch_size=np.int(np.ceil(self.N_batch/self.N_minibatches))

		self.batch_size = np.int(self.minibatch_size*self.N_minibatches)
		

		######  preallocate memory
		dtype=np.float64

		self.F_vector=np.zeros(self.N_varl_params,dtype=dtype)
		
		if self.NN_dtype=='real':
			self.dlog_psi_aux=np.zeros((self.batch_size,self.N_varl_params),dtype=np.float64)
			#self.ddlog_psi_aux=np.zeros((self.batch_size,self.N_varl_params,self.N_varl_params),dtype=np.float64)


			self.dlog_psi=np.zeros([self.N_batch,self.N_varl_params],dtype=np.float64)
			#self.ddlog_psi=np.zeros([self.N_batch,self.N_varl_params,self.N_varl_params],dtype=np.float64)


			self.O_expt=np.zeros(self.N_varl_params,dtype=np.float64)

			self.F_vector_log=None
			self.F_vector_phase=None

			self.E_diff_weighted=np.zeros(self.N_batch,dtype=dtype)
		
		else:
			self.dlog_psi_aux=np.zeros((self.batch_size,self.N_varl_params),dtype=np.complex128)

			self.dlog_psi=np.zeros([self.N_batch,self.N_varl_params],dtype=np.complex128)
			self.O_expt=np.zeros(self.N_varl_params,dtype=np.complex128)

			self.F_vector_log=np.zeros_like(self.F_vector)
			self.F_vector_phase=np.zeros_like(self.F_vector)

			self.E_diff_weighted=np.zeros(self.N_batch,dtype=np.complex128)

		
		self.nat_grad=np.zeros_like(self.F_vector)
		self.current_grad_guess=np.zeros_like(self.F_vector)
		self.S_matrix=np.zeros(2*self.F_vector.shape,dtype=dtype)
		self.S_matrix_reg=1E-15*np.eye(self.F_vector.shape[0])
		self.nat_grad_guess=np.zeros_like(self.F_vector)

		self.OO_expt=np.zeros([self.N_varl_params,self.N_varl_params],dtype=dtype)
		self.O_expt2=np.zeros_like(self.OO_expt)


		self.S_eigvals=np.zeros_like(self.F_vector)
		self.VF_overlap=np.zeros_like(self.F_vector)

		self.SNR_exact=np.zeros_like(self.F_vector)
		self.SNR_gauss=np.zeros_like(self.F_vector)

		if self.adaptive_SR_cutoff:
			self.Q_expt=np.zeros(self.N_varl_params,dtype=dtype)
			self.QQ_expt=np.zeros([self.N_batch,self.N_varl_params],dtype=np.float64)
		
		

		self.S_norm=0.0 # S-matrix norm
		self.F_norm=0.0 # F norm
		self.S_logcond=0.0
		self.Flog_norm=0.0
		self.Fphase_norm=0.0

			
		
		if self.mode=='MC':
			if self.comm.Get_rank()==0:
				self.S_lastiters=np.zeros([n_iter,self.N_varl_params,self.N_varl_params],dtype=dtype) # array to store last S-matrices
				self.F_lastiters=np.zeros([n_iter,self.N_varl_params,],dtype=dtype) # array to store last F-vectors
			else:
				self.S_lastiters=np.array([[None],[None]])
				self.F_lastiters=np.array([[None],[None]])
		
		
	def debug_helper(self):

		if self.comm.Get_rank()==0:

			# store last n_iter data points
			
			self.S_lastiters[:-1,...]=self.S_lastiters[1:,...]
			self.F_lastiters[:-1,...]=self.F_lastiters[1:,...]
			
			# set last step data
			self.S_lastiters[-1,...]=self.S_matrix
			self.F_lastiters[-1,...]=self.F_vector

		self.comm.Barrier() # Gatherv is blocking, so this is probably superfluous
		

	
	def compute_S_matrix(self,Eloc_params_dict=None):
		
		if self.mode=='exact':
			abs_psi_2=Eloc_params_dict['abs_psi_2']#.copy()

			self.O_expt[:]=jnp.einsum('s,sj->j',abs_psi_2,self.dlog_psi).block_until_ready() 
				
			if self.NN_dtype=='real':
			
				self.OO_expt[:] = jnp.dot(self.dlog_psi.T, jnp.dot(jnp.diag(abs_psi_2), self.dlog_psi)	).block_until_ready()  		
			else:
				self.OO_expt[:] = jnp.dot(self.dlog_psi.real.T, jnp.dot(jnp.diag(abs_psi_2), self.dlog_psi.real)  ).block_until_ready() \
								+ jnp.dot(self.dlog_psi.imag.T, jnp.dot(jnp.diag(abs_psi_2), self.dlog_psi.imag)  ).block_until_ready()  		

			
		elif self.mode=='ED':
			abs_psi_2=Eloc_params_dict['abs_psi_2']

			self.comm.Allreduce(np.asarray(jnp.dot(abs_psi_2,self.dlog_psi).block_until_ready() ), self.O_expt[:], op=MPI.SUM) 
			
			# expand dimension
			abs_psi_2=np.tile(abs_psi_2,[self.N_varl_params,1],)

			
			if self.NN_dtype=='real':
				self.comm.Allreduce(( np.asarray(jnp.dot(self.dlog_psi.T*abs_psi_2, self.dlog_psi).block_until_ready()  ) ), self.OO_expt[:], op=MPI.SUM )
			else:
				self.comm.Allreduce(  np.asarray(   jnp.dot(self.dlog_psi.real.T*abs_psi_2, self.dlog_psi.real).block_until_ready() \
				                	  	  			+jnp.dot(self.dlog_psi.imag.T*abs_psi_2, self.dlog_psi.imag).block_until_ready()    ), \
									self.OO_expt[:], op=MPI.SUM
									)


		elif self.mode=='MC':

			self.comm.Allreduce(np.asarray(jnp.sum(self.dlog_psi,axis=0).block_until_ready() ), self.O_expt[:], op=MPI.SUM)
			#self.comm.Allreduce(np.sum(self.dlog_psi,axis=0), self.O_expt[:], op=MPI.SUM)
			
			self.O_expt/=self.N_MC_points

			if self.NN_dtype=='real':
				self.comm.Allreduce( np.asarray( jnp.dot(self.dlog_psi.T, self.dlog_psi).block_until_ready()  ), self.OO_expt[:], op=MPI.SUM )
				#self.comm.Allreduce( np.dot(self.dlog_psi.T, self.dlog_psi), self.OO_expt[:], op=MPI.SUM )
			else:
				self.comm.Allreduce(	np.asarray(  jnp.dot(self.dlog_psi.real.T, self.dlog_psi.real).block_until_ready() \
				                	  	  			 +jnp.dot(self.dlog_psi.imag.T, self.dlog_psi.imag).block_until_ready()    ), \
									self.OO_expt[:], op=MPI.SUM
									)

			self.OO_expt/=self.N_MC_points


		if self.NN_dtype=='real':
			self.O_expt2[:] = jnp.outer(self.O_expt,self.O_expt)
		else:
			self.O_expt2[:] = jnp.outer(self.O_expt.real,self.O_expt.real) + jnp.outer(self.O_expt.imag,self.O_expt.imag)


		self.S_matrix[:] = self.OO_expt - self.O_expt2 + self.S_matrix_reg

		
	

	def compute_F_vector(self,Eloc_params_dict=None):

		self.E_diff_weighted[:]=Eloc_params_dict['E_diff'].copy()

		if self.mode in 'exact':
			abs_psi_2=Eloc_params_dict['abs_psi_2'].copy()
			self.E_diff_weighted*=abs_psi_2

			if self.NN_dtype=='real':
				self.F_vector[:]   = jnp.dot(self.E_diff_weighted,self.dlog_psi).block_until_ready()
			else:
				
				self.F_vector_log[:]   = jnp.dot(self.E_diff_weighted.real,self.dlog_psi.real).block_until_ready()
				self.F_vector_phase[:] = jnp.dot(self.E_diff_weighted.imag,self.dlog_psi.imag).block_until_ready()

				self.F_vector[:]=self.F_vector_log+self.F_vector_phase

		if self.mode in 'ED':
			self.E_diff_weighted*=Eloc_params_dict['abs_psi_2']

			if self.NN_dtype=='real':
				self.comm.Allreduce(np.asarray(  jnp.dot(self.E_diff_weighted, self.dlog_psi).block_until_ready() ), self.F_vector[:], op=MPI.SUM)
				
				
			else:
				
				self.comm.Allreduce(	np.asarray( jnp.dot(self.E_diff_weighted.real,self.dlog_psi.real).block_until_ready() ), \
									self.F_vector_log[:], op=MPI.SUM
									)
		
				self.comm.Allreduce(	np.asarray( jnp.dot(self.E_diff_weighted.imag,self.dlog_psi.imag).block_until_ready() ), \
									self.F_vector_phase[:], op=MPI.SUM
									)
		
				self.F_vector[:]=self.F_vector_log+self.F_vector_phase


		elif self.mode=='MC':
		
			if self.NN_dtype=='real':
				
				self.comm.Allreduce(np.asarray(  jnp.dot(self.E_diff_weighted, self.dlog_psi).block_until_ready() ), self.F_vector[:], op=MPI.SUM)
				#self.comm.Allreduce(np.dot(self.E_diff_weighted, self.dlog_psi), self.F_vector[:], op=MPI.SUM)
			
				self.F_vector/=self.N_MC_points

			else:
				self.comm.Allreduce(	np.asarray( jnp.dot(self.E_diff_weighted.real,self.dlog_psi.real).block_until_ready() ), \
									self.F_vector_log[:], op=MPI.SUM
									)
				self.F_vector_log/=self.N_MC_points



				self.comm.Allreduce(	np.asarray( jnp.dot(self.E_diff_weighted.imag,self.dlog_psi.imag).block_until_ready() ), \
									self.F_vector_phase[:], op=MPI.SUM
									)
				self.F_vector_phase/=self.N_MC_points



				self.F_vector[:]=self.F_vector_log+self.F_vector_phase



	def signal_to_noise_ratio(self,lmbda,V,Eloc_params_dict):

		# re-set variables
		self.SNR_exact*=0.0
		self.SNR_gauss*=0.0
					
		
		Eloc_var=Eloc_params_dict['Eloc_var']
		E_diff=Eloc_params_dict['E_diff']
		if self.mode in ['exact','ED']:
			E_diff*=np.sqrt( Eloc_params_dict['abs_psi_2'] )


		self.QQ_expt[:]=np.asarray(jnp.dot(self.dlog_psi-np.tile(self.O_expt,[self.N_batch,1],), V).block_until_ready() ) 
		self.comm.Allreduce( (np.asarray(  jnp.dot(E_diff**2, self.QQ_expt**2).block_until_ready() ) ), self.Q_expt[:], op=MPI.SUM)
					
		self.Q_expt/=self.N_MC_points

		# add connected piece
		self.Q_expt-=self.VF_overlap**2

		# take only values above machine precision
		#finite_k,=np.where( (np.abs(self.VF_overlap)/np.max(np.abs(self.VF_overlap))>1E-14) & (np.abs(lmbda)/np.max(lmbda)>1E-14) )
		finite_k,=np.where( (np.abs(self.VF_overlap)/np.max(np.abs(self.VF_overlap))>1E-14) & (np.abs(lmbda)/np.max(np.abs(lmbda))>1E-14) )
					
		self.SNR_exact[finite_k]=np.abs(self.VF_overlap[finite_k])/(np.sqrt(np.abs(self.Q_expt[finite_k])) + 1E-14)
		self.SNR_gauss[finite_k]=1.0/np.sqrt(1.0 + (np.abs(lmbda[finite_k])/(self.VF_overlap[finite_k]**2 ) )*Eloc_var)


		if self.mode=='MC':
			self.SNR_exact*=np.sqrt(self.N_MC_points)
			self.SNR_gauss*=np.sqrt(self.N_MC_points)


		# adjust tolerance according to SNR
		threshold=1.0
		weight=np.abs(self.VF_overlap)/(np.abs(self.VF_overlap).sum())
		inds,= np.where((self.SNR_exact>threshold) )

		if len(inds)>0:
			self.SNR_weight_sum_exact=weight[inds].sum()
		else:
			self.SNR_weight_sum_exact=0.0
		

		inds_gauss, = np.where((self.SNR_gauss>threshold) )
		if len(inds_gauss)>0:
			self.SNR_weight_sum_gauss=weight[inds_gauss].sum()
		else:
			self.SNR_weight_sum_gauss=0.0


		print('k-components kept:', inds.shape, self.SNR_weight_sum_exact, self.SNR_weight_sum_gauss)

		#print(1.0/lmbda[inds])

		
		# if self.comm.Get_rank()==0 and self.iteration>1:

		# 	import matplotlib.pyplot as plt
		# 	plt.plot(self.SNR_gauss,'r', label='SNR gauss')
		# 	plt.plot(self.SNR_exact,'b', label='SNR exact')
			
		# 	plt.plot(threshold*np.ones_like(self.SNR_exact),'-k', label='{0:0.2f}'.format(threshold) )
		# 	plt.plot(np.abs(self.VF_overlap), 'm', label='V^t F')
		# 	plt.plot(lmbda, 'c', label='spec', linewidth=4)
		# 	plt.yscale('log')
		# 	plt.title('iter= {0:d}, N_MC={0:d}'.format(self.iteration, self.N_MC_points))
		# 	plt.grid()
		# 	plt.legend()
		# 	plt.show()

		# 	exit()


		return inds


	def compute_r2_cost(self,Eloc_params_dict):
		# S a = F
		Eloc_var=Eloc_params_dict['Eloc_var']
		return ( (np.dot(self.nat_grad, np.dot(self.S_matrix,self.nat_grad)) - 2.0*np.dot(self.F_vector,self.nat_grad) + Eloc_var )/Eloc_var )
		

	def _S_matrix_checks(self):

		# import pickle
		# file_name='./bug'
		# with open(file_name+'.pkl', 'wb') as handle:
		# 	pickle.dump([self.dlog_psi, self.OO_expt, self.O_expt2, self.O_expt, self.S_matrix, self.F_vector], handle, protocol=pickle.HIGHEST_PROTOCOL)


		
		norm=jnp.linalg.norm(self.S_matrix).block_until_ready()

		
		if np.linalg.norm((self.S_matrix-self.S_matrix.T.conj())/norm ) > 1E-13: # and np.linalg.norm(self.S_matrix-self.S_matrix.T.conj()) > 1E-14:
		
			print('F : {:.20f}'.format(np.linalg.norm((self.S_matrix-self.S_matrix.T.conj())/norm )) )
			print('OO: {:.20f}'.format(np.linalg.norm( (self.OO_expt-self.OO_expt.T.conj())/np.linalg.norm(self.OO_expt) )) )
			print('O2: {:.20f}'.format(np.linalg.norm( (self.O_expt2-self.O_expt2.T.conj())/np.linalg.norm(self.O_expt2) )) )
			print('non-hermitian Fisher matrix with norm:', norm)
			
			np.testing.assert_allclose(self.S_matrix/norm,self.S_matrix.T.conj()/norm, rtol=1E-14, atol=1E-14)

			exit()
		
		E = eigh(self.S_matrix/norm,eigvals_only=True)
		

		if np.any(E <- 1E-14):
			print('E ', E)
			print()
			print('OO',eigh(self.OO_expt/np.linalg.norm(self.OO_expt),eigvals_only=True))
			print()
			print('O2',eigh(self.O_expt2/np.linalg.norm(self.O_expt2),eigvals_only=True))

			print('negative eigenvalues')


			np.linalg.cholesky(self.S_matrix/norm)


			exit()


	def _TDVP_solver(self, S, F, nat_grad_guess, Eloc_params_dict):

		info=0

		# S=np.random.normal(size=S.shape)
		# S=S+S.T
		# F=np.random.normal(size=F.shape)

		if self.TDVP_opt == 'cg':
			self.nat_grad[:], info = cg(S,F,x0=nat_grad_guess,maxiter=self.cg_maxiter,atol=self.tol,tol=self.tol) # 
		
		elif self.TDVP_opt == 'inv':
			self.nat_grad[:]=np.asarray(jnp.dot(jnp.linalg.inv(S), F).block_until_ready())
		
		elif self.TDVP_opt == 'svd':
			lmbda, V = jnp.linalg.eigh(S/self.S_norm,)
			#lmbda, V = np.linalg.eigh(S/self.S_norm,)
			#lmbda, V = eigh(S/self.S_norm,)
			lmbda*=self.S_norm

		

			# a1= jnp.dot(V ,  jnp.dot( np.diag(1.0/(lmbda+1E-14)), jnp.dot(V.T, F) ) ) #[-4:]
			# a2 = inv(S/self.S_norm).dot(F)/self.S_norm #[-4:]
			# print(np.linalg.norm(a1-a2))
			# print(np.sqrt(np.dot( (a1-a2).conj() , np.dot(S,a1-a2) )) )
			# exit()

			self.S_eigvals[:]=lmbda
			self.VF_overlap[:]= jnp.dot(V.T, F)


			# print(self.S_norm)
			# print(np.linalg.norm(self.VF_overlap))
			# print(np.linalg.norm(V))
			# print(np.linalg.norm(lmbda))
			# #exit()


			if self.NN_dtype=='real' and self.adaptive_SR_cutoff:
				SNR_inds=self.signal_to_noise_ratio(lmbda,V,Eloc_params_dict)

		
			if self.adaptive_SR_cutoff and self.NN_dtype=='real':
				self.nat_grad[:] = jnp.dot(V[:,SNR_inds] ,  jnp.dot( np.diag(1.0/lmbda[SNR_inds] ), self.VF_overlap[SNR_inds] ) )
			else:
				self.nat_grad[:] = jnp.dot(V ,  jnp.dot( jnp.diag(lmbda/(lmbda**2 + (self.tol)**2) ), self.VF_overlap ) )
				#self.nat_grad[:] = jnp.dot(V ,  jnp.dot( np.diag( 2.0 / ( lmbda * (1.0 + np.exp(8.0*self.tol*lmbda[-1]/np.abs(lmbda)) )  )   ), self.VF_overlap ) )
			

			#print(np.linalg.norm(self.nat_grad))
			#exit()

		return info


	def _compute_grads(self,NN_params,batch,):

		if self.mode=='MC':
		
			self.dlog_psi[:]=self.compute_grad_log_psi(NN_params,batch,)
		
		else: # ED

			for j in range(self.N_minibatches):

				
				batch_idx=np.arange(j*self.minibatch_size, (j+1)*self.minibatch_size)	
				#print(batch.shape, batch_idx.shape, self.dlog_psi_aux.shape)

				self.dlog_psi_aux[batch_idx]=self.compute_grad_log_psi(NN_params,batch[batch_idx],)
				#self.ddlog_psi_aux[batch_idx]=self.hessian(NN_params,batch[batch_idx],)
				
			
			self.dlog_psi[:]=self.dlog_psi_aux[:self.N_batch]
			#self.ddlog_psi[:]=self.ddlog_psi_aux[:self.N_batch]


	def _compute_hessian(self,NN_params,batch,weights, logfile):

		if self.mode=='MC':
		
			self.dlog_psi[:]=self.compute_grad_log_psi(NN_params,batch,)
		
		else: # ED


			ddlog_psi=np.zeros((self.N_varl_params,self.N_varl_params),dtype=np.float64)

			batch_shape=batch.shape
			new_batch_shape=(-1,)+batch.shape[2:]

			# pad weights
			weights=np.pad(weights,(0,self.minibatch_size*self.N_minibatches - weights.shape[0]),'constant')


			for j in range(self.N_minibatches):

				batch_idx=np.arange(j*self.minibatch_size, (j+1)*self.minibatch_size)
			
				ti=time.time()	
				ddlog_psi[...]+=self.hessian2(NN_params,batch[batch_idx].reshape(new_batch_shape),weights[batch_idx]).squeeze()
				#self.hessian2(NN_params,batch[batch_idx].reshape(new_batch_shape),weights[batch_idx]).squeeze().block_until_ready()
				tf=time.time()

				# ti2=time.time()	
				# #ddlog_psi[...]+=self.hessian(NN_params,batch[batch_idx],weights[batch_idx])
				# self.hessian(NN_params,batch[batch_idx],weights[batch_idx]).block_until_ready()
				# #jnp.einsum('s,smn->mn',weights[batch_idx], self.hessian(NN_params,batch[batch_idx],weights[batch_idx]) ).block_until_ready()
				# tf2=time.time()


				# print(tf-ti, tf2-ti2)

				print('finished iteration {0:d}/{1:d} in {2:0.6f} secs on {3:d} states.'.format(j,self.N_minibatches, tf-ti,len(batch_idx)))
				if logfile is not None:
					logfile.flush()

			H = np.zeros(ddlog_psi.shape,dtype=np.float64)
			self.comm.Allreduce( ddlog_psi , H[...], op=MPI.SUM )

			print('hermiticity check:', np.linalg.norm(H-H.conj().T))


		return 2.0*H



	def compute(self,NN_params,batch,Eloc_params_dict,):

		t0=time.time()
		#self.dlog_psi[:]=self.compute_grad_log_psi(NN_params,batch[:self.N_batch],)
		self._compute_grads(NN_params,batch,) # 
		
		#A=self.compute_grad_log_psi(NN_params,batch[:self.N_batch],)
		#print('HEREEEEE', np.max(np.abs(self.dlog_psi - A) ))
		#exit()	

		t1=time.time()
		self.compute_F_vector(Eloc_params_dict=Eloc_params_dict,)
		t2=time.time()
		self.compute_S_matrix(Eloc_params_dict=Eloc_params_dict,)
		t3=time.time()
		


		print("evaluation took -- gradients: {0:0.6} secs; F_vector: {1:0.6} secs; S-matrix: {2:0.6} secs.".format(t1-t0, t2-t1, t3-t2) )
		
		### compute natural_gradients using cg
		# regularize Fisher metric
		#print("max[diag(S)]", np.abs(np.diag(self.S_matrix)).max())

		if not self.adaptive_SR_cutoff:
			self.S_matrix += self.delta*np.diag(np.diag(self.S_matrix))
			#self.S_matrix += self.delta*np.linalg.norm(self.S_matrix)*np.eye(self.S_matrix.shape[0]) 

		if self.debug_mode and self.mode=='MC':
			self.debug_helper()


		####################################################### 
		
		if self.check_on:
			# check for symmetry and positivity
			self._S_matrix_checks()


		# compute norm
		self.S_norm=np.linalg.norm(self.S_matrix)
		self.F_norm=np.linalg.norm(self.F_vector)
		self.S_logcond=np.log(np.linalg.cond(self.S_matrix))
		if self.NN_dtype=='cpx':
			self.Flog_norm=np.linalg.norm(self.F_vector_log)
			self.Fphase_norm=np.linalg.norm(self.F_vector_phase)


		#######################################################

		
		# solve TDVP EOM

		info=1
		while info>0 and self.cg_maxiter<1E5:
			
			info = self._TDVP_solver(self.S_matrix,self.F_vector, self.nat_grad_guess, Eloc_params_dict)
				
			# affects CG solver only
			if info>0:
				self.cg_maxiter*=2
				print('cg failed to converge in {0:d} iterations to tolerance {1:0.14f}; increasing maxiter to {2:d}'.format(info,self.tol,int(self.cg_maxiter)))
		
		
		###############

		# clip gradients
		# self.nat_grad[:]=np.where(np.abs(self.nat_grad) < self.grad_clip, self.nat_grad, self.grad_clip) 


		# normalize gradients
		self.dE=2.0*np.dot(self.F_vector,self.nat_grad)
		self.curvature=np.sqrt( np.dot(self.nat_grad, np.dot(self.S_matrix, self.nat_grad) ) )


		self.iteration+=1

		if not self.RK:	
			return self.nat_grad#/self.curvature
		else:
			return self.nat_grad

		#return self.nat_grad #/self.curvature
		
		

	def update_NG_params(self,grad_guess,self_time=1.0):

		#if self.delta>self.tol:
		self.delta *= np.exp(-0.075*self_time)
		

		self.nat_grad_guess[:]=grad_guess
		

class Runge_Kutta_solver():

	def __init__(self,step_size, NN_Tree, return_grads, reestimate_local_energy, compute_r2, NG, adaptive_step):

		self.NN_Tree=NN_Tree
		self.return_grads=return_grads

		self.NG=NG

		self.reestimate_local_energy=reestimate_local_energy
		self.compute_r2=compute_r2
		self.r2=0.0
		self.dE=0.0

		self.S_norm=0.0
		self.F_norm=0.0
		self.Flog_norm=0.0
		self.Fphase_norm=0.0

		self.S_eigvals=np.zeros(NN_Tree.N_varl_params,dtype=np.float64)
		self.VF_overlap=np.zeros_like(self.S_eigvals)

		# RK params
		self.adaptive_step=adaptive_step
		self.step_size=step_size
		self.time=0.0
		self.RK_tol=1E-5 # 5E-5 
		self.RK_inv_p=1.0/3.0
		
		self.counter=0
		self.iteration=0
			
		self.S_matrix=np.zeros((NN_Tree.N_varl_params, NN_Tree.N_varl_params),dtype=np.float64)

		self.dy=np.zeros(NN_Tree.N_varl_params,dtype=np.float64)
		self.dy_star=np.zeros_like(self.dy)

		self.init_grad=np.zeros_like(self.dy)
		self.params=np.zeros_like(self.dy)

		self.k1=np.zeros_like(self.dy)
		self.k2=np.zeros_like(self.dy)
		self.k3=np.zeros_like(self.dy)
		self.k4=np.zeros_like(self.dy)
		self.k5=np.zeros_like(self.dy)
		self.k6=np.zeros_like(self.dy)
		
		self.SNR_exact=np.zeros_like(self.dy)
		self.SNR_gauss=np.zeros_like(self.dy)
		self.SNR_weight_sum_exact=0.0
		self.SNR_weight_sum_gauss=0.0



	def run(self,NN_params,batch,Eloc_params_dict,):

		# flatten weights
		self.params[:]=self.NN_Tree.ravel(NN_params,)

		#params_norm=jnp.max(jnp.abs(params)).block_until_ready()
		params_norm=jnp.linalg.norm(self.params).block_until_ready()
		

		#initial_curvature=self.NG.curvature
		if self.NG is not None:
			self.NG.debug_mode=True
			self.init_grad[:]=self.return_grads(NN_params,batch,Eloc_params_dict,)
			self.NG.debug_mode=False

			self.S_matrix[:,:]=self.NG.S_matrix
			#S_norm=self.NG.S_norm

			self.r2=self.compute_r2(Eloc_params_dict)
			self.dE=self.NG.dE*self.step_size

			self.F_norm=self.NG.F_norm
			self.S_norm=self.NG.S_norm
			self.Flog_norm=self.NG.Flog_norm
			self.Fphase_norm=self.NG.Fphase_norm

			self.SNR_exact[:]=self.NG.SNR_exact
			self.SNR_gauss[:]=self.NG.SNR_gauss
			self.SNR_weight_sum_exact=self.NG.SNR_weight_sum_exact
			self.SNR_weight_sum_gauss=self.NG.SNR_weight_sum_gauss

			self.S_eigvals[:]=self.NG.S_eigvals
			self.VF_overlap[:]=self.NG.VF_overlap

		else:
			self.init_grad[:]=self.return_grads(NN_params,batch,Eloc_params_dict,)



		if self.adaptive_step==True:

			self.counter+=1

			error_ratio=0.0
			local_iteration=0
			while error_ratio<1.0:

				print('    RK iteration {0:d}:'.format(local_iteration) )

				### RK step 1
				self.k1[:]=-self.step_size*self.init_grad
			
				### RK step 2
				NN_params_shifted=self.NN_Tree.unravel(self.params+self.k1)
				Eloc_params_dict, batch = self.reestimate_local_energy(self.iteration, NN_params_shifted, batch, Eloc_params_dict)
				self.k2[:]=-self.step_size*self.return_grads(NN_params_shifted,batch,Eloc_params_dict,)

				# full-step solution difference
				self.dy[:]=0.5*self.k1 + 0.5*self.k2


				#######
				### RK step 1
				self.k3[:]=-0.5*self.step_size*self.init_grad # 0.5*k1
				
				### RK step 2
				NN_params_shifted=self.NN_Tree.unravel(self.params+self.k3)
				Eloc_params_dict, batch = self.reestimate_local_energy(self.iteration, NN_params_shifted, batch, Eloc_params_dict)
				self.k4[:]=-0.5*self.step_size*self.return_grads(NN_params_shifted,batch,Eloc_params_dict,)

				# first half-step solution difference
				self.dy_star[:]=0.5*self.k3 + 0.5*self.k4
			

				### RK step 1
				NN_params_shifted=self.NN_Tree.unravel(self.params+self.dy_star)
				Eloc_params_dict, batch = self.reestimate_local_energy(self.iteration, NN_params_shifted, batch, Eloc_params_dict)
				self.k5[:]=-0.5*self.step_size*self.return_grads(NN_params_shifted,batch,Eloc_params_dict,)

				### RK step 2
				NN_params_shifted=self.NN_Tree.unravel(self.params+self.dy_star+self.k5)
				Eloc_params_dict, batch = self.reestimate_local_energy(self.iteration, NN_params_shifted, batch, Eloc_params_dict)
				self.k6[:]=-0.5*self.step_size*self.return_grads(NN_params_shifted,batch,Eloc_params_dict,)

				# second half-step solution difference
				self.dy_star[:]+=0.5*self.k5 + 0.5*self.k6



				#######
				if self.NG is not None:
					norm=np.sqrt( np.dot(self.dy-self.dy_star, np.dot(self.S_matrix, self.dy-self.dy_star) ) )/params_norm
					#norm=np.linalg.norm(self.dy-self.dy_star)/params_norm
				else:
					norm=np.linalg.norm(self.dy-self.dy_star)/params_norm
					#norm=np.max(np.dot(self.NG.S_matrix, self.dy-self.dy_star) )/params_norm
					#norm=np.max(np.abs(self.dy-self.dy_star))/params_norm

			
				error_ratio=6.0*self.RK_tol/norm

				self.step_size*=min(2.0,max(0.2,0.9*error_ratio**self.RK_inv_p))
				
				self.counter+=4 # five gradient calculations


				
				local_iteration+=1


		else:

			self.counter+=2

			### RK step 1
			self.k1[:]=-self.step_size*self.init_grad
		
			### RK step 2
			NN_params_shifted=self.NN_Tree.unravel(self.params+self.k1)
			Eloc_params_dict, batch = self.reestimate_local_energy(self.iteration, NN_params_shifted, batch, Eloc_params_dict)
			self.k2[:]=-self.step_size*self.return_grads(NN_params_shifted,batch,Eloc_params_dict,)

			# full-step solution difference
			self.dy_star[:]=0.5*self.k1 + 0.5*self.k2

			norm=0.0



		# update params
		self.iteration+=1
		self.time+=self.step_size
		
		print('    RK_steps={0:d}-step_size={1:0.15f}-time={2:0.4f}-norm={3:0.14f}.\n'.format(self.counter, self.step_size, self.time, norm, ) )


		return self.dy_star # - 1.0/6.0*(self.dy-self.dy_star)


