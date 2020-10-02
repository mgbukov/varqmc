import sys, os

from cpp_code import update_offdiag_ME_exact, update_offdiag_ME, update_diag_ME, c_offdiag_sum, c_offdiag_sum_H

from mpi4py import MPI
import numpy as np
import time

from jax.config import config
config.update("jax_enable_x64", True)
from jax import jit, disable_jit
import jax.numpy as jnp

from cpp_code import integer_to_spinstate

from pandas import read_csv


def _consolidate_static(static_list):
	eps = 10 * np.finfo(np.float64).eps

	static_dict={}
	for opstr,bonds in static_list:
		if opstr not in static_dict:
			static_dict[opstr] = {}

		for bond in bonds:
			J = bond[0]
			indx = tuple(bond[1:])
			if indx in static_dict[opstr]:
				static_dict[opstr][indx] += J
			else:
				static_dict[opstr][indx] = J

	static_list = []
	for opstr,opstr_dict in static_dict.items():
		for indx,J in opstr_dict.items():
			if np.abs(J) > eps:
				static_list.append((opstr,indx,J))


	return static_list





def data_stream(data,minibatch_size,sample_size,N_minibatches):
    #rng = np.random.RandomState(0)
    while True:
        #perm = rng.permutation(sample_size)
        for i in range(N_minibatches):
            #batch_idx = perm[i * minibatch_size : (i + 1) * minibatch_size]
            batch_idx = np.arange(i*minibatch_size, min(sample_size, (i+1)*minibatch_size), 1)
            #batch_idx = np.arange(i*minibatch_size, (i+1)*minibatch_size, 1)
            yield data[batch_idx], batch_idx,



class Energy_estimator():

	def __init__(self,comm,DNN_log,DNN_phase,mode,J2,N_MC_points,N_batch,L,N_symm,sign,minibatch_size,semi_exact):

		# MPI commuicator
		self.comm=comm
		self.DNN_log=DNN_log
		self.DNN_phase=DNN_phase

		if DNN_phase is None:
			self.NN_dtype='cpx'
		else:
			self.NN_dtype='real'


		self.N_MC_points=N_MC_points
		self.N_batch=N_batch
		self.logfile=None
		self.mode=mode

		self.minibatch_size=minibatch_size

		self.semi_exact_log=semi_exact[0]
		self.semi_exact_phase=semi_exact[1]




		###### define model parameters ######
		Lx=L
		Ly=Lx # linear dimension of spin 1 2d lattice
		N_sites = Lx*Ly # number of sites for spin 1
		#
		###### setting up user-defined symmetry transformations for 2d lattice ######
		sites = np.arange(N_sites,dtype=np.int32) # sites [0,1,2,....]

		x = sites%Lx # x positions for sites
		y = sites//Lx # y positions for sites
		sublattice=(x%2)^(y%2)

		T_x = (x+1)%Lx + Lx*y # translation along x-direction
		T_y = x +Lx*((y+1)%Ly) # translation along y-direction

		T_a = (x+1)%Lx + Lx*((y+1)%Ly) # translation along anti-diagonal
		T_d = (x-1)%Lx + Lx*((y+1)%Ly) # translation along diagonal


		###### setting up hamiltonian ######
		J1=1.0 # spin=spin interaction
		J2=J2 # magnetic field strength
		#sign=-1.0 # -1: Marshal rule is on; +1 Marshal rule is off
		#sign=+1.0 # -1: Marshal rule is on; +1 Marshal rule is off

		self.N_sites=N_sites
		self.N_symm=N_symm
		self.sign=int(sign)
		self.J2=J2
		self.L=L



		###### setting up Hamiltonian site-coupling lists
		J1_list=[[J1,i,T_x[i]] for i in range(N_sites)] + [[J1,i,T_y[i]] for i in range(N_sites)]
		J1_pm_list=[[sign*0.5*J1,i,T_x[i]] for i in range(N_sites)] + [[sign*0.5*J1,i,T_y[i]] for i in range(N_sites)]
		J2_list=[[J2,i,T_d[i]] for i in range(N_sites)] + [[J2,i,T_a[i]] for i in range(N_sites)]
		J2_pm_list=[[0.5*J2,i,T_d[i]] for i in range(N_sites)] + [[0.5*J2,i,T_a[i]] for i in range(N_sites)]
	

		self.static_off_diag=[ ["+-",J1_pm_list],["-+",J1_pm_list],]
		self.static_diag=[ ["zz",J1_list], ]
		
		if self.J2>0:
			self.static_off_diag+=[ ["+-",J2_pm_list],["-+",J2_pm_list] ]
			self.static_diag+=[["zz",J2_list],]
		
		static_list_offdiag = _consolidate_static(self.static_off_diag)
		static_list_diag = _consolidate_static(self.static_diag)
		

		self._n_offdiag_terms=len(static_list_offdiag)
		self._static_list_offdiag=static_list_offdiag
		self._static_list_diag=static_list_diag


		# self.static_off_diag=self.static_off_diag_SdotS
		# self.static_diag=self.static_diag_SdotS
		
		# self._n_offdiag_terms=len(static_list_offdiag_SdotS)
		# self._static_list_offdiag=static_list_offdiag_SdotS
		# self._static_list_diag=static_list_diag_SdotS


		self.E_GS_density_approx=-0.5
		if Lx==4:
			self.Hdim=107
			self.basis_type=np.uint16
			self.MPI_basis_dtype=MPI.SHORT	
			if J2==0:
				self.E_GS= -11.228483 #-0.7017801875*self.N_sites
			else:
				self.E_GS= -8.45792 #-0.528620*self.N_sites
		elif Lx==6:
			self.Hdim=15804956
			self.basis_type=np.uint64
			self.MPI_basis_dtype=MPI.LONG
			if J2==0:
				self.E_GS= -24.4393969968 #-0.6788721388*self.N_sites
			else:
				self.E_GS= -18.13716 #-0.503810*self.N_sites	
		elif Lx==8:
			self.Hdim=None
			self.basis_type=np.uint64
			self.MPI_basis_dtype=MPI.LONG


	def get_exact_kets(self):


		assert(self.L==4)

		from quspin.basis import spin_basis_general
		from quspin.operators import hamiltonian

		Lx=self.L
		Ly=self.L

		sites = np.arange(self.N_sites,dtype=np.int32) # sites [0,1,2,....]
			
		x = sites%Lx # x positions for sites
		y = sites//Lx # y positions for sites

		T_x = (x+1)%Lx + Lx*y # translation along x-direction
		T_y = x +Lx*((y+1)%Ly) # translation along y-direction

		P_x = x + Lx*(Ly-y-1) # reflection about x-axis
		P_y = (Lx-x-1) + Lx*y # reflection about y-axis
		P_d = y + Lx*x

		Z   = -(sites+1) # spin inversion

		###### setting up bases ######
		self.basis_symm = spin_basis_general(self.N_sites, pauli=False, Ns_block_est=200,
											Nup=self.N_sites//2,
											kxblock=(T_x,0),kyblock=(T_y,0),
											pdblock=(P_d,0),
											pxblock=(P_x,0),pyblock=(P_y,0),
											zblock=(Z,0),
											block_order=['zblock','pdblock','pyblock','pxblock','kyblock','kxblock']
										)
		self.basis = spin_basis_general(self.N_sites, pauli=False, Nup=self.N_sites//2)
		
		self.H=hamiltonian(self.static_off_diag+self.static_diag, [], basis=self.basis,dtype=np.float64) #
		self.H_symm=hamiltonian(self.static_off_diag+self.static_diag, [], basis=self.basis_symm,dtype=np.float64)
		
		
		ref_states, index, inv_index, count=np.unique(self.basis_symm.representative(self.basis.states), return_index=True, return_inverse=True, return_counts=True)
		self.ref_states=ref_states
		self.count=count

		# j=1#5
		# y=np.zeros(self.H.Ns)
		# index=basis.index(ref_states[j])
		# y[index]=1.0
		# print(self.H.expt_value(y))

		# H_symm=hamiltonian(self.static_off_diag+self.static_diag, [], basis=basis_symm,dtype=np.float64)
		E,V = self.H.eigsh(k=2,which='BE')
		self.psi_GS_exact=V[:,0]

		# print(E[:10])
		# exit()

		# print(E[0],self.SdotS.expt_value(self.psi_GS_exact))

		# S,_ = self.SdotS.eigsh(k=2,which='BE')
		# #S,_ = self.SdotS.eigh()
		# print(S)

		# print( np.linalg.norm( (self.H.dot(self.SdotS) - self.SdotS.dot(self.H)).toarray() ) )

		# exit()

		
		return ref_states.astype(self.basis_type), index, inv_index, count



	def load_exact_basis(self,NN_type,MC_tool,N_features,load_file,skiprows):


		self.MC_tool=MC_tool
		self.N_features=N_features


		
		# load data
		ref_states=read_csv(load_file, skiprows=skiprows, nrows=self.N_batch, header=None, dtype=self.basis_type, delimiter=' ',usecols=[0,]) 
		self.MC_tool.count=read_csv(load_file, skiprows=skiprows, nrows=self.N_batch, header=None, dtype=np.uint16, delimiter=' ',usecols=[6,]).to_numpy().squeeze() 
		
		#print(self.comm.Get_rank(),ref_states.to_numpy().squeeze() )
		#exit()

		self.MC_tool.ints_ket=ref_states.to_numpy().squeeze()
		#self.MC_tool.count=self.count


		# print(self.comm.Get_rank(), self.count.shape[0]!=self.N_batch, self.count.shape[0],self.N_batch,self.MC_tool.log_mod_kets.shape[0], skiprows)
		# exit()

		### compute exact GS
		
		# load data
		log_psi_GS =read_csv(load_file, skiprows=skiprows, nrows=self.N_batch, header=None, dtype=np.float64, delimiter=' ',usecols=[1,]) 
		if self.sign>0:
			sign_psi_GS=read_csv(load_file, skiprows=skiprows, nrows=self.N_batch, header=None, dtype=np.float64, delimiter=' ',usecols=[4,]) 
		else:
			sign_psi_GS=read_csv(load_file, skiprows=skiprows, nrows=self.N_batch, header=None, dtype=np.float64, delimiter=' ',usecols=[3,]) 
		
		# build psi
		self.psi_GS_exact = sign_psi_GS.to_numpy().squeeze() * np.exp(log_psi_GS.to_numpy().squeeze())

		

		# bcast psi
		# if self.comm.Get_rank()==0:
		# 	self.psi_GS_exact=np.zeros(self.Hdim,dtype=np.float64)
		# else:
		# 	self.psi_GS_exact=np.array([None])
		# self.comm.Gatherv([psi_GS,   MPI.DOUBLE], [self.psi_GS_exact[:],   MPI.DOUBLE], root=0)

		print('starting int_to-state', self.MC_tool.spinstates_ket.shape,)
		#exit()

		#(N_configs, N_symm, L, L) --> (N_configs*N_symm*L*L, )
		# compute spin s-configs
		integer_to_spinstate(self.MC_tool.ints_ket, self.MC_tool.spinstates_ket[:self.MC_tool.N_batch*self.MC_tool.N_features], self.N_features, NN_type=NN_type)

		
		print('\nfisnihed loading data.\n')
		#exit()
		
		# gather s data
		self.comm.Barrier()
		self.comm.Allgatherv([self.MC_tool.ints_ket,    self.MC_tool.MPI_basis_dtype], [self.MC_tool.ints_ket_g[:],   self.MC_tool.MPI_basis_dtype], )
		#self.comm.Gatherv([self.MC_tool.count,   MPI.SHORT], [self.MC_tool.count_g[:],   MPI.SHORT], root=0)
		
		#exit()

		### compute s'
		self.precompute_s_primes(self.MC_tool.ints_ket,NN_type)

		
		# find indices of s primes
		self.s_prime_inds=np.searchsorted(self.MC_tool.ints_ket_g,self.ints_bra_uq,)
		

		# free up memory
		# self.MC_tool.ints_ket_g=None
		# self.MC_tool.ints_ket=None
		# self.ints_bra_uq=None
		# del log_psi_GS, sign_psi_GS
		




	
	def lookup_s_primes(self,data):
		return data[self.s_prime_inds][self.inv_index]



	def precompute_s_primes(self,ints_ket,NN_type):

		# preallocate variables
		self._reset_locenergy_params()


		# diag matrix elements, only real part
		for opstr,indx,J in self._static_list_diag:

			indx=np.asarray(indx,dtype=np.int32)
			update_diag_ME(ints_ket,self.Eloc_diag,opstr,indx,J) 


		# off-diag matrix elements
		nn=0
		for j,(opstr,indx,J) in enumerate(self._static_list_offdiag):
			
			self._ints_ket_ind_holder[:]=-np.ones((self.N_batch,),dtype=np.int32)

			indx=np.asarray(indx,dtype=np.int32)
			n = update_offdiag_ME_exact(ints_ket,self._ints_bra_rep_holder,self._ints_ket_ind_holder,self._MEs_holder,opstr,indx,J)
			
			self._MEs[nn:nn+n]=self._MEs_holder[self._ints_ket_ind_holder[:n]]
			self._ints_bra_rep[nn:nn+n]=self._ints_bra_rep_holder[self._ints_ket_ind_holder[:n]]
			self._ints_ket_ind[nn:nn+n]=self._ints_ket_ind_holder[:n]

			
			self._n_per_term[j]=n
			nn+=n

		# evaluate network on unique representatives only

		# if self.L==4:
		self.ints_bra_uq, index, inv_index, =np.unique(self._ints_bra_rep[:nn], return_index=True, return_inverse=True, )
		nn_uq=self.ints_bra_uq.shape[0]
		

		self.nn=nn
		self.nn_uq=nn_uq
		self.index=index
		self.inv_index=inv_index


	def init_global_params(self,N_MC_points,n_iter,SdotS=False):

		self._ints_bra_rep_holder=np.zeros((self.N_batch,),dtype=self.basis_type)
		self._MEs_holder=np.zeros((self.N_batch,),dtype=np.float64)
		self._ints_ket_ind_holder=-np.ones((self.N_batch,),dtype=np.int32)

		if self.mode=="MC":

			if self.comm.Get_rank()==0:
				self.Eloc_real_g=np.zeros((n_iter,N_MC_points),dtype=np.float64)
				self.Eloc_imag_g=np.zeros_like(self.Eloc_real_g)
			else:
				self.Eloc_real_g=np.array([[None],[None]])
				self.Eloc_imag_g=np.array([[None],[None]])

			self._spinstates_bra_holder=np.zeros((self.N_batch,self.N_sites*self.N_symm),dtype=np.int8)
		
		
		elif self.mode=='exact' or (self.mode=='ED' and (self.semi_exact_log or self.semi_exact_phase)):
			self._spinstates_bra_holder=np.zeros((self.N_batch,self.N_sites*self.N_symm),dtype=np.int8)
		

		#self.SdotS_real_tot=np.zeros(N_MC_points,dtype=np.float64)
		#self.SdotS_imag_tot=np.zeros_like(self.SdotS_real_tot)


	def debug_helper(self):

		if self.comm.Get_rank()==0:
			self.Eloc_real_g[:-1,...]=self.Eloc_real_g[1:,...]
			self.Eloc_imag_g[:-1,...]=self.Eloc_imag_g[1:,...]

		self.comm.Barrier() # Gatherv is blocking, so this is probably superfluous
		
		self.comm.Gatherv([self.Eloc_real,  MPI.DOUBLE], [self.Eloc_real_g[-1,:], MPI.DOUBLE], root=0)
		self.comm.Gatherv([self.Eloc_imag,  MPI.DOUBLE], [self.Eloc_imag_g[-1,:], MPI.DOUBLE], root=0)


	def _reset_locenergy_params(self,):

		#######
		self._MEs=np.zeros(self.N_batch*self._n_offdiag_terms,dtype=np.float64)
		self._ints_bra_rep=np.zeros((self.N_batch*self._n_offdiag_terms,),dtype=self.basis_type)
		self._ints_ket_ind=np.zeros(self.N_batch*self._n_offdiag_terms,dtype=np.uint32)
		self._n_per_term=np.zeros(self._n_offdiag_terms,dtype=np.int32)

		self.Eloc_diag=np.zeros(self.N_batch, dtype=np.float64)
		self.Eloc_diag_H=np.zeros(self.N_batch, dtype=np.float64)		


		if (not self.mode=='ED') or (self.semi_exact_log or self.semi_exact_phase):
			self._spinstates_bra=np.zeros((self.N_batch*self._n_offdiag_terms,self.N_sites*self.N_symm),dtype=np.int8)
		


	def _reset_Eloc_vars(self,):

		self._Eloc_cos=np.zeros(self.N_batch, dtype=np.float64)
		self._Eloc_sin=np.zeros(self.N_batch, dtype=np.float64)

		self.Eloc_real=np.zeros_like(self._Eloc_cos)
		self.Eloc_imag=np.zeros_like(self._Eloc_cos)
		

	def reestimate_local_energy_phase(self, iteration, NN_params_phase, batch, params_dict):


		# if self.DNN_phase.semi_exact==False:
		# 	phase_kets=self.DNN_phase.evaluate(NN_params_phase, batch.reshape(self.DNN_phase.input_shape), )
		# else: # exact phases
		# 	phase_kets=self.DNN_phase.evaluate(NN_params_phase, self.MC_tool.ints_ket, )


		if self.mode=='ED':
			for j in range(self.MC_tool.N_minibatches):

				batch_idx=np.arange(j*self.MC_tool.minibatch_size//self.N_symm, (j+1)*self.MC_tool.minibatch_size//self.N_symm)	
				self.MC_tool.phase_kets_aux[batch_idx] = self.DNN_phase.evaluate(NN_params_phase,batch[batch_idx].reshape(self.DNN_phase.input_shape),  )
		
			phase_kets=self.MC_tool.phase_kets_aux[:self.N_batch]

			self.MC_tool.phase_kets=phase_kets
			self.comm.Allgatherv([self.MC_tool.phase_kets,  MPI.DOUBLE], [self.MC_tool.phase_kets_g[:], MPI.DOUBLE], )
			
			if self.semi_exact_log==True and self.semi_exact_phase==False: # exact logs, optimize phases
				phase_psi_bras = self.evaluate_s_primes(self.DNN_phase.evaluate,NN_params_phase,self.DNN_phase.input_shape,semi_exact=self.semi_exact_phase,)
			else:
				phase_psi_bras = self.lookup_s_primes(self.MC_tool.phase_kets_g)
			
		
		else:
			phase_kets=np.asarray(self.DNN_phase.evaluate(NN_params_phase, batch.reshape(self.DNN_phase.input_shape), ))
			phase_psi_bras = self.evaluate_s_primes(self.DNN_phase.evaluate,NN_params_phase,self.DNN_phase.input_shape,semi_exact=self.semi_exact_phase,)

		
		# print(phase_kets)
		# print(phase_psi_bras)
		# #print(self.MC_tool.phase_kets)
		# exit()

		self.compute_Eloc(self.log_kets, phase_kets, self.log_psi_bras, phase_psi_bras, debug_mode=False,)

		Eloc_mean_g, Eloc_var_g, E_diff_real, E_diff_imag = self.process_local_energies(params_dict)

		params_dict['E_diff']=E_diff_imag
		params_dict['Eloc_mean']=Eloc_mean_g
		params_dict['Eloc_var']=Eloc_var_g
		#params_dict['Eloc_mean_part']=Eloc_mean_g.imag
		

		return params_dict, batch


	def ED_compute_local_energy_hessian(self,params_log,params_phase,ints_ket, log_kets,phase_kets,log_psi_shift, dlog_kets, dphase_kets, verbose=True, ):

		log_psi_bras   = self.lookup_s_primes(self.MC_tool.log_mod_kets_g)
		phase_psi_bras = self.lookup_s_primes(self.MC_tool.phase_kets_g)


		log_psi_bras-=log_psi_shift
		
		HO_log   =    self.compute_Eloc_H(log_kets, phase_kets, log_psi_bras, phase_psi_bras, dlog_kets)
		HO_phase = 1j*self.compute_Eloc_H(log_kets, phase_kets, log_psi_bras, phase_psi_bras, dphase_kets)
		
		# compute_Eloc must come after compute_Eloc_H
		self.compute_Eloc(log_kets, phase_kets, log_psi_bras, phase_psi_bras,)
		Eloc=self.Eloc_real+1j*self.Eloc_imag


		self.log_kets=log_kets
		self.log_psi_bras=log_psi_bras

		return Eloc, HO_log, HO_phase




	def compute_local_energy(self,params_log,params_phase,ints_ket,log_kets,phase_kets,log_psi_shift, verbose=True, ):
		
		ti=time.time()
		if (not self.mode=='ED') or (self.semi_exact_log or self.semi_exact_phase) :
			self.compute_s_primes(ints_ket,self.DNN_log.NN_type)


		str_1="computing s_primes took {0:.4f} secs.".format(time.time()-ti)
		#if self.logfile!=None:
		#	self.logfile.write(str_1)
		if self.comm.Get_rank()==0 and verbose:
			print(str_1)



		unique_str="{0:d}/{1:d} unique configs; using minibatch size {2:d}.".format(self.nn_uq, self.nn, self.minibatch_size)
		print(unique_str)
		
		ti=time.time()
		if self.NN_dtype=='real':

			if self.mode=='ED':


				if self.semi_exact_log==False and self.semi_exact_phase==True: # exact phases, optimize the logs
					log_psi_bras = self.evaluate_s_primes(self.DNN_log.evaluate, params_log, self.DNN_log.input_shape,semi_exact=self.semi_exact_log,)	
				else:
					log_psi_bras = self.lookup_s_primes(self.MC_tool.log_mod_kets_g)

				
				if self.semi_exact_log==True and self.semi_exact_phase==False: # exact logs, optimize phases
					phase_psi_bras = self.evaluate_s_primes(self.DNN_phase.evaluate,params_phase,self.DNN_phase.input_shape,semi_exact=self.semi_exact_phase,)
				else:
					phase_psi_bras = self.lookup_s_primes(self.MC_tool.phase_kets_g)	

					


			else:
				log_psi_bras = self.evaluate_s_primes(self.DNN_log.evaluate, params_log, self.DNN_log.input_shape,semi_exact=self.semi_exact_log,)	
				phase_psi_bras = self.evaluate_s_primes(self.DNN_phase.evaluate,params_phase,self.DNN_phase.input_shape,semi_exact=self.semi_exact_phase,)

		else:
			if self.mode=='ED':
				log_psi_bras = self.lookup_s_primes(self.MC_tool.log_mod_kets_g)
				phase_psi_bras = self.lookup_s_primes(self.MC_tool.phase_kets_g)

			else:
				log_psi_bras, phase_psi_bras = self.evaluate_s_primes(self.DNN_log.evaluate, params_log, self.DNN_log.input_shape,semi_exact=self.semi_exact_log,)
		
		log_psi_bras-=log_psi_shift
		


		psi_str="log_|psi|_bras: min={0:0.8f}, max={1:0.8f}, mean={2:0.8f}; std={3:0.8f}, diff={4:0.8f}.".format(np.min(log_psi_bras), np.max(log_psi_bras), np.mean(log_psi_bras), np.std(log_psi_bras), np.max(log_psi_bras)-np.min(log_psi_bras) )
		if self.comm.Get_rank()==0 and verbose:
			print(psi_str)


		str_2="evaluating s_primes took {0:.4f} secs.".format(time.time()-ti)
		if self.comm.Get_rank()==0 and verbose:
			print(str_2)


		
		self.compute_Eloc(log_kets, phase_kets, log_psi_bras, phase_psi_bras,)

		self.log_kets=log_kets
		self.log_psi_bras=log_psi_bras


	def compute_s_primes(self,ints_ket,NN_type):

		# preallocate variables
		self._reset_locenergy_params()


		# diag matrix elements, only real part
		for opstr,indx,J in self._static_list_diag:

			indx=np.asarray(indx,dtype=np.int32)
			update_diag_ME(ints_ket,self.Eloc_diag,opstr,indx,J) 


		# off-diag matrix elements
		nn=0
		for j,(opstr,indx,J) in enumerate(self._static_list_offdiag):
			
			self._spinstates_bra_holder[:]=np.zeros((self.N_batch,self.N_sites*self.N_symm),dtype=np.int8)
			self._ints_ket_ind_holder[:]=-np.ones((self.N_batch,),dtype=np.int32)

			indx=np.asarray(indx,dtype=np.int32)
			n = update_offdiag_ME(ints_ket,self._ints_bra_rep_holder,self._spinstates_bra_holder,self._ints_ket_ind_holder,self._MEs_holder,opstr,indx,J,self.N_symm,NN_type)
			
			self._MEs[nn:nn+n]=self._MEs_holder[self._ints_ket_ind_holder[:n]]
			self._ints_bra_rep[nn:nn+n]=self._ints_bra_rep_holder[self._ints_ket_ind_holder[:n]]
			self._spinstates_bra[nn:nn+n]=self._spinstates_bra_holder[self._ints_ket_ind_holder[:n]]
			self._ints_ket_ind[nn:nn+n]=self._ints_ket_ind_holder[:n]

			#sample: [s1, s2, s3, ]
			#s_primes: [ [s11, s12, s12], [s21, s22,], [...], ]
			
			self._n_per_term[j]=n
			nn+=n

		# print(ints_ket)
		# print(self._MEs[:nn])
		# print(self._ints_bra_rep[:nn])
		# exit()


		# evaluate network on unique representatives only

		# if self.L==4:
		self.ints_bra_uq, index, inv_index, =np.unique(self._ints_bra_rep[:nn], return_index=True, return_inverse=True, )
		nn_uq=self.ints_bra_uq.shape[0]
		# else:
		# 	nn_uq=nn
		# 	index=np.arange(nn)
		# 	inv_index=index


		self.nn=nn
		self.nn_uq=nn_uq
		self.index=index
		self.inv_index=inv_index

	

	def evaluate_s_primes(self,evaluate_NN,NN_params,input_shape,semi_exact=False):

		### evaluate network using minibatches

		if self.minibatch_size > 0:
		
			num_complete_batches, leftover = divmod(self.nn_uq, self.minibatch_size)
			N_minibatches = num_complete_batches + bool(leftover)

			if semi_exact:
				data=np.zeros((N_minibatches*self.minibatch_size,),dtype=self.ints_bra_uq.dtype)
				data[:self.nn_uq,...]=self.ints_bra_uq
			else:
				data=np.zeros((N_minibatches*self.minibatch_size,)+self._spinstates_bra.shape[1:],dtype=self._spinstates_bra.dtype)
				data[:self.nn_uq,...]=self._spinstates_bra[:self.nn][self.index]
				


			
			# preallocate data
			prediction_bras=np.zeros(N_minibatches*self.minibatch_size,dtype=np.float64)
			if self.NN_dtype=='cpx':
				prediction_bras_2=np.zeros_like(prediction_bras)


			ti=time.time()
			for j in range(N_minibatches):
				#ti=time.time()

				batch_idx=np.arange(j*self.minibatch_size, (j+1)*self.minibatch_size)
				batch=data[batch_idx]

				if self.NN_dtype=='real':
					if semi_exact:
						prediction_bras[batch_idx] = evaluate_NN(NN_params, batch,)
					else:
						prediction_bras[batch_idx] = evaluate_NN(NN_params, batch.reshape(input_shape),)
				else:
					if semi_exact:
						prediction_bras[batch_idx], prediction_bras_2[batch_idx] = evaluate_NN(NN_params, batch,)
					else:
						prediction_bras[batch_idx], prediction_bras_2[batch_idx] = evaluate_NN(NN_params, batch.reshape(input_shape),)
				

				# with disable_jit():
				# 	log, phase = evaluate_NN(DNN.params, batch.reshape(batch.shape[0],self.N_symm,self.N_sites), DNN.apply_fun_args )
			
				#print(log_psi_bras[batch_idx]-log)
				#print(log[:2])

			prediction_bras=prediction_bras[:self.nn_uq]
			if self.NN_dtype=='cpx':
				prediction_bras_2=prediction_bras_2[:self.nn_uq]

			print("network evaluation on {0:d} configs took {1:0.6} secs.".format(data.shape[0], time.time()-ti) )
	

		else:

			### evaluate network on entire sample
			if self.NN_dtype=='real':
				prediction_bras = np.asarray(evaluate_NN(NN_params,self._spinstates_bra[:self.nn][self.index].reshape(input_shape),  ) )
			else:	
				prediction_bras, prediction_bras_2 = np.asarray( evaluate_NN(NN_params,self._spinstates_bra[:self.nn][self.index].reshape(input_shape),  ) )
			

		if self.NN_dtype=='real':
			return prediction_bras[self.inv_index]
		else:
			return prediction_bras[self.inv_index], prediction_bras_2[self.inv_index]

		#######


	def compute_Eloc(self, log_kets, phase_kets, log_psi_bras,phase_psi_bras, debug_mode=True):


		self._reset_Eloc_vars()
		
		# compute real and imaginary part of local energy
		self._n_per_term=self._n_per_term[self._n_per_term>0]
		c_offdiag_sum(self._Eloc_cos, self._Eloc_sin, self._n_per_term,self._ints_ket_ind[:self.nn],self._MEs[:self.nn],log_psi_bras,phase_psi_bras,log_kets)
		
	
		cos_phase_kets=np.cos(phase_kets)
		sin_phase_kets=np.sin(phase_kets)


		self.Eloc_real = self._Eloc_cos*cos_phase_kets + self._Eloc_sin*sin_phase_kets
		self.Eloc_imag = self._Eloc_sin*cos_phase_kets - self._Eloc_cos*sin_phase_kets

		# add diagonal contribution
		self.Eloc_real+=self.Eloc_diag


		#################################
		#
		# check variance of E_loc 
		if debug_mode and self.mode=='MC':
			self.debug_helper()


		return self.Eloc_real+1j*self.Eloc_imag

		

	def compute_Eloc_H(self, log_kets, phase_kets, log_psi_bras,phase_psi_bras, dpsi_kets,):

		
		HO = np.zeros(dpsi_kets.shape, dtype=np.complex128)
		N_varl_params=dpsi_kets.shape[1]

		_n_per_term=self._n_per_term[self._n_per_term>0]

		

		# loop over NN parameters
		for i in range(N_varl_params):


			# Allgather data
			self.comm.Allgatherv([np.ascontiguousarray(dpsi_kets[:,i]),  MPI.DOUBLE], [self.MC_tool.dpsi_kets_g[:], MPI.DOUBLE], )

			# find s' contributions
			dpsi_bras   = self.lookup_s_primes(self.MC_tool.dpsi_kets_g)

		
			
			self._reset_Eloc_vars()
			
			# compute real and imaginary part of local energy
			c_offdiag_sum_H(self._Eloc_cos, self._Eloc_sin, _n_per_term,self._ints_ket_ind[:self.nn],self._MEs[:self.nn],log_psi_bras, phase_psi_bras, log_kets, dpsi_bras)
			
		
			cos_phase_kets=np.cos(phase_kets)
			sin_phase_kets=np.sin(phase_kets)


			Eloc_real = self._Eloc_cos*cos_phase_kets + self._Eloc_sin*sin_phase_kets
			Eloc_imag = self._Eloc_sin*cos_phase_kets - self._Eloc_cos*sin_phase_kets

			# add diagonal contribution
			Eloc_real+=self.Eloc_diag*dpsi_kets[:,i]

			
			HO[:,i]=Eloc_real+1j*Eloc_imag


		return HO




		return self.Eloc_real+1j*self.Eloc_imag


	def process_local_energies(self,Eloc_params_dict,):


		Eloc=self.Eloc_real+1j*self.Eloc_imag

		# np.set_printoptions(threshold=np.inf,precision=16)
		# #print(self.comm.Get_rank(), Eloc.real)
		# print(self.comm.Get_rank(), Eloc_params_dict['abs_psi_2'])
		# exit()
		
		Eloc_mean_g=np.zeros(1, dtype=np.complex128)
		Eloc_var_g=np.zeros(1, dtype=np.float64)

		if self.mode=='MC':

			# Eloc_mean=np.mean(Eloc).real
			# Eloc_var=np.sum( np.abs(Eloc)**2)/self.Eloc_real_tot.shape[0] - Eloc_mean**2

			self.comm.Allreduce(np.sum(       Eloc    ), Eloc_mean_g, op=MPI.SUM)
			Eloc_mean_g/=self.N_MC_points

			
			self.comm.Allreduce(np.sum(np.abs(Eloc)**2), Eloc_var_g,  op=MPI.SUM)
			Eloc_var_g/=self.N_MC_points
			Eloc_var_g-=np.abs(Eloc_mean_g)**2

			Eloc_mean_g=Eloc_mean_g[0]
			Eloc_var_g=np.abs(Eloc_var_g[0])


		elif self.mode=='exact':
			abs_psi_2=Eloc_params_dict['abs_psi_2']
			Eloc_mean_g=np.sum(Eloc*abs_psi_2).real
			Eloc_var_g=np.sum(abs_psi_2*np.abs(Eloc)**2) - Eloc_mean_g**2
			
		elif self.mode=='ED':

			abs_psi_2=Eloc_params_dict['abs_psi_2']
			Eloc_mean=np.sum(abs_psi_2*Eloc)
			self.comm.Allreduce(Eloc_mean, Eloc_mean_g, op=MPI.SUM)
			
			self.comm.Allreduce(np.sum(abs_psi_2*np.abs(Eloc)**2), Eloc_var_g,  op=MPI.SUM)
			Eloc_var_g-=np.abs(Eloc_mean_g)**2

			#print('{0:0.15f}'.format(Eloc_var_g[0]))
			# print('{0:0.15f}'.format(np.abs(Eloc_mean_g[0])**2))
			# exit()

			Eloc_mean_g=Eloc_mean_g[0]
			Eloc_var_g=Eloc_var_g[0]



		E_diff_real=self.Eloc_real-Eloc_mean_g.real
		E_diff_imag=self.Eloc_imag-Eloc_mean_g.imag


		self.Eloc_mean_g=Eloc_mean_g
		self.Eloc_var_g=Eloc_var_g

		return Eloc_mean_g, Eloc_var_g, E_diff_real, E_diff_imag




def compute_outliers(data):

	q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
	iqr = q75 - q25
	# calculate the outlier cutoff
	cut_off = iqr * 1.5
	lower, upper = q25 - cut_off, q75 + cut_off
	
	return (lower, upper), (q25, q75, iqr)

