from quspin.operators._make_hamiltonian import _consolidate_static

from cpp_code import update_offdiag_ME, update_diag_ME, c_offdiag_sum

from mpi4py import MPI
import numpy as np
import time

from jax.config import config
config.update("jax_enable_x64", True)
from jax import jit, disable_jit
import jax.numpy as jnp


def compute_outliers(data):

	q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
	iqr = q75 - q25
	# calculate the outlier cutoff
	cut_off = iqr * 1.5
	lower, upper = q25 - cut_off, q75 + cut_off
	
	return (lower, upper), (q25, q75, iqr)



def data_stream(data,minibatch_size,sample_size,N_minibatches):
    #rng = np.random.RandomState(0)
    while True:
        #perm = rng.permutation(sample_size)
        for i in range(N_minibatches):
            #batch_idx = perm[i * minibatch_size : (i + 1) * minibatch_size]
            batch_idx = np.arange(i*minibatch_size, min(sample_size, (i+1)*minibatch_size), 1)
            yield data[batch_idx], batch_idx



class Energy_estimator():

	def __init__(self,comm,J2,N_MC_points,N_batch,L,N_symm,NN_type,sign,):

		# MPI commuicator
		self.comm=comm
		self.N_MC_points=N_MC_points
		self.N_batch=N_batch
		self.NN_type=NN_type
		self.logfile=None


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

	

		####### SdotS term
		zz_list=[[1.0,i,j] for i in range(N_sites) for j in range(N_sites) if j<i]
		pm_list=[[sign**(sublattice[i])*sign**(sublattice[j])*0.5,i,j] for i in range(N_sites) for j in range(N_sites) if j<i]

		# zz_list=[[1.0,i,j] for i in range(N_sites) for j in range(N_sites)]
		# pm_list=[[sign**(sublattice[i])*sign**(sublattice[j])*0.5,i,j] for i in range(N_sites) for j in range(N_sites)]
		

		self.static_off_diag_SdotS=[ ["+-",pm_list],["-+",pm_list] ]
		self.static_diag_SdotS=[ ["zz",zz_list], ]


		static_list_offdiag_SdotS = _consolidate_static(self.static_off_diag_SdotS)
		static_list_diag_SdotS = _consolidate_static(self.static_diag_SdotS)


		self._n_offdiag_terms_SdotS=len(static_list_offdiag_SdotS)
		self._static_list_offdiag_SdotS=static_list_offdiag_SdotS
		self._static_list_diag_SdotS=static_list_diag_SdotS


		###### setting up Hamiltonian site-coupling lists
		J1_list=[[J1,i,T_x[i]] for i in range(N_sites)] + [[J1,i,T_y[i]] for i in range(N_sites)]
		J1_pm_list=[[sign*0.5*J1,i,T_x[i]] for i in range(N_sites)] + [[sign*0.5*J1,i,T_y[i]] for i in range(N_sites)]
		J2_list=[[J2,i,T_d[i]] for i in range(N_sites)] + [[J2,i,T_a[i]] for i in range(N_sites)]
		J2_pm_list=[[0.5*J2,i,T_d[i]] for i in range(N_sites)] + [[0.5*J2,i,T_a[i]] for i in range(N_sites)]
	

		self.static_off_diag=[ ["+-",J1_pm_list],["-+",J1_pm_list], ["+-",J2_pm_list],["-+",J2_pm_list] ]
		self.static_diag=[ ["zz",J1_list],["zz",J2_list] ]
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
			self.basis_type=np.uint16
			self.MPI_basis_dtype=MPI.SHORT	
			if J2==0:
				self.E_GS= -11.228483 #-0.7017801875*self.N_sites
			else:
				self.E_GS= -8.45792 #-0.528620*self.N_sites
		elif Lx==6:
			self.basis_type=np.uint64
			self.MPI_basis_dtype=MPI.LONG
			if J2==0:
				self.E_GS= -24.4393969968 #-0.6788721388*self.N_sites
			else:
				self.E_GS= -18.13716 #-0.503810*self.N_sites	
		elif Lx==8:
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
		
		self.SdotS=hamiltonian(self.static_off_diag_SdotS+self.static_diag_SdotS, [], basis=self.basis,dtype=np.float64)

		ref_states, index, inv_index, count=np.unique(self.basis_symm.representative(self.basis.states), return_index=True, return_inverse=True, return_counts=True)
		self.ref_states=ref_states
		self.count=count

		# j=1#5
		# y=np.zeros(self.H.Ns)
		# index=basis.index(ref_states[j])
		# y[index]=1.0
		# print(self.H.expt_value(y))

		# print(ref_states[j])
		# print(basis.int_to_state(ref_states[j]))


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



	def init_global_params(self,N_MC_points,n_iter,SdotS=False):

		self._spinstates_bra_holder=np.zeros((self.N_batch,self.N_sites*self.N_symm),dtype=np.int8)
		self._ints_bra_rep_holder=np.zeros((self.N_batch,),dtype=self.basis_type)
		self._MEs_holder=np.zeros((self.N_batch,),dtype=np.float64)
		self._ints_ket_ind_holder=-np.ones((self.N_batch,),dtype=np.int32)

		if self.comm.Get_rank()==0:
			self.Eloc_real_g=np.zeros((n_iter,N_MC_points),dtype=np.float64)
			self.Eloc_imag_g=np.zeros_like(self.Eloc_real_g)
		else:
			self.Eloc_real_g=np.array([[None],[None]])
			self.Eloc_imag_g=np.array([[None],[None]])

		#self.SdotS_real_tot=np.zeros(N_MC_points,dtype=np.float64)
		#self.SdotS_imag_tot=np.zeros_like(self.SdotS_real_tot)


	def debug_helper(self):

		if self.comm.Get_rank()==0:
			self.Eloc_real_g[:-1,...]=self.Eloc_real_g[1:,...]
			self.Eloc_imag_g[:-1,...]=self.Eloc_imag_g[1:,...]

		self.comm.Barrier() # Gatherv is blocking, so this is probably superfluous
		
		self.comm.Gatherv([self.Eloc_real,  MPI.DOUBLE], [self.Eloc_real_g[-1,:], MPI.DOUBLE], root=0)
		self.comm.Gatherv([self.Eloc_imag,  MPI.DOUBLE], [self.Eloc_imag_g[-1,:], MPI.DOUBLE], root=0)


	def _reset_locenergy_params(self,SdotS=False):

		#######
		if SdotS:
			self._MEs=np.zeros(self.N_batch*self._n_offdiag_terms_SdotS,dtype=np.float64)
			self._spinstates_bra=np.zeros((self.N_batch*self._n_offdiag_terms_SdotS,self.N_sites*self.N_symm),dtype=np.int8)
			self._ints_bra_rep=np.zeros((self.N_batch*self._n_offdiag_terms_SdotS,),dtype=self.basis_type)
			self._ints_ket_ind=np.zeros(self.N_batch*self._n_offdiag_terms_SdotS,dtype=np.uint32)
			self._n_per_term=np.zeros(self._n_offdiag_terms_SdotS,dtype=np.int32)
		else:
			self._MEs=np.zeros(self.N_batch*self._n_offdiag_terms,dtype=np.float64)
			self._spinstates_bra=np.zeros((self.N_batch*self._n_offdiag_terms,self.N_sites*self.N_symm),dtype=np.int8)
			self._ints_bra_rep=np.zeros((self.N_batch*self._n_offdiag_terms,),dtype=self.basis_type)
			self._ints_ket_ind=np.zeros(self.N_batch*self._n_offdiag_terms,dtype=np.uint32)
			self._n_per_term=np.zeros(self._n_offdiag_terms,dtype=np.int32)


		self._Eloc_cos=np.zeros(self.N_batch, dtype=np.float64)
		self._Eloc_sin=np.zeros(self.N_batch, dtype=np.float64)

		self.Eloc_real=np.zeros_like(self._Eloc_cos)
		self.Eloc_imag=np.zeros_like(self._Eloc_cos)	


	def compute_local_energy(self,evaluate_NN,DNN,NN_params,ints_ket,log_kets,phase_kets,log_psi_shift,minibatch_size,SdotS=False):
		

		if SdotS:
			static_list_offdiag=self._static_list_offdiag_SdotS	
			static_list_diag=self._static_list_diag_SdotS
		else:
			static_list_offdiag=self._static_list_offdiag
			static_list_diag=self._static_list_diag
		
		# preallocate variables
		self._reset_locenergy_params(SdotS=SdotS)

		nn=0
		for j,(opstr,indx,J) in enumerate(static_list_offdiag):
			
			self._spinstates_bra_holder[:]=np.zeros((self.N_batch,self.N_sites*self.N_symm),dtype=np.int8)
			self._ints_ket_ind_holder[:]=-np.ones((self.N_batch,),dtype=np.int32)

			indx=np.asarray(indx,dtype=np.int32)
			n = update_offdiag_ME(ints_ket,self._ints_bra_rep_holder,self._spinstates_bra_holder,self._ints_ket_ind_holder,self._MEs_holder,opstr,indx,J,self.N_symm,self.NN_type)
			
			self._MEs[nn:nn+n]=self._MEs_holder[self._ints_ket_ind_holder[:n]]
			self._ints_bra_rep[nn:nn+n]=self._ints_bra_rep_holder[self._ints_ket_ind_holder[:n]]
			self._spinstates_bra[nn:nn+n]=self._spinstates_bra_holder[self._ints_ket_ind_holder[:n]]
			self._ints_ket_ind[nn:nn+n]=self._ints_ket_ind_holder[:n]

			
			self._n_per_term[j]=n
			nn+=n

		# print(ints_ket)
		# print(self._MEs[:nn])
		# print(self._ints_bra_rep[:nn])
		# exit()


		# evaluate network on unique representatives only

		_ints_bra_uq, index, inv_index, count=np.unique(self._ints_bra_rep[:nn], return_index=True, return_inverse=True, return_counts=True)
		nn_uq=_ints_bra_uq.shape[0]


		unique_str="{0:d}/{1:d} unique configs; using minibatch size {2:d}.\n".format(nn_uq, nn, minibatch_size)
		if self.logfile!=None:
			self.logfile.write(unique_str)


		### evaluate network using minibatches

		if minibatch_size > 0:
		
			num_complete_batches, leftover = divmod(nn_uq, minibatch_size)
			N_minibatches = num_complete_batches + bool(leftover)

			data=self._spinstates_bra[:nn][index]
			batches = data_stream(data,minibatch_size,nn_uq,N_minibatches)

			# preallocate data
			log_psi_bras=np.zeros(nn_uq,dtype=np.float64)
			phase_psi_bras=np.zeros(nn_uq,dtype=np.float64)

			for j in range(N_minibatches):

				batch, batch_idx = next(batches)
				log_psi_bras[batch_idx], phase_psi_bras[batch_idx] = evaluate_NN(NN_params, batch.reshape(batch.shape[0],self.N_symm,self.N_sites), DNN.apply_fun_args )
				
				# with disable_jit():
				# 	log, phase = evaluate_NN(DNN.params, batch.reshape(batch.shape[0],self.N_symm,self.N_sites), DNN.apply_fun_args )
			
				#print(log_psi_bras[batch_idx]-log)
				#print(log[:2])

				
			log_psi_bras=log_psi_bras[inv_index] - log_psi_shift
			phase_psi_bras=phase_psi_bras[inv_index]


		else:

			### evaluate network on entire sample
			log_psi_bras, phase_psi_bras = evaluate_NN(NN_params,self._spinstates_bra[:nn][index].reshape(nn_uq,self.N_symm,self.N_sites),)
			log_psi_bras=log_psi_bras[inv_index]._value - log_psi_shift
			phase_psi_bras=phase_psi_bras[inv_index]._value



		# Eloc_cos=self._MEs[:nn]*np.exp(log_psi_bras-log_kets)*np.cos(phase_psi_bras)
		# Eloc_sin=self._MEs[:nn]*np.exp(log_psi_bras-log_kets)*np.sin(phase_psi_bras)
		
		# cos_phase_kets=np.cos(phase_kets)
		# sin_phase_kets=np.sin(phase_kets)

		# indxs=(np.exp(log_psi_bras-log_kets)>1.0)

		# print(np.exp(log_psi_bras-log_kets))
		# print()
		# print(log_psi_bras)
		# print()
		# print( (Eloc_cos*cos_phase_kets) )
		# print()
		# print( (Eloc_cos*cos_phase_kets)[indxs] )
		# print()
		# print(Eloc_cos.sum()*cos_phase_kets, Eloc_sin.sum()*sin_phase_kets)
		# print()
		# print(log_kets)
		# exit()

		#######

		psi_str="log_|psi|_bras: min={0:0.8f}, max={1:0.8f}, mean={2:0.8f}; std={3:0.8f}, diff={4:0.8f}.\n".format(np.min(log_psi_bras), np.max(log_psi_bras), np.mean(log_psi_bras), np.std(log_psi_bras), np.max(log_psi_bras)-np.min(log_psi_bras) )
		if self.logfile!=None:
			self.logfile.write(psi_str)
		if self.comm.Get_rank()==0:
			print(psi_str)
		

		# compute real and imaginary part of local energy
		self._n_per_term=self._n_per_term[self._n_per_term>0]
		c_offdiag_sum(self._Eloc_cos, self._Eloc_sin, self._n_per_term,self._ints_ket_ind[:nn],self._MEs[:nn],log_psi_bras,phase_psi_bras,log_kets)
		

		cos_phase_kets=np.cos(phase_kets)
		sin_phase_kets=np.sin(phase_kets)


		self.Eloc_real = self._Eloc_cos*cos_phase_kets + self._Eloc_sin*sin_phase_kets
		self.Eloc_imag = self._Eloc_sin*cos_phase_kets - self._Eloc_cos*sin_phase_kets

		#print(self.Eloc_real)

		# diag matrix elements, only real part
		for opstr,indx,J in static_list_diag:

			indx=np.asarray(indx,dtype=np.int32)
			update_diag_ME(ints_ket,self.Eloc_real,opstr,indx,J) 

		
		#print(self.Eloc_real)
		#exit()

		#################################
		#
		# check variance of E_loc 
		self.debug_helper()


		"""
		lower, upper = 0.0,0.0
		q25, q75, iqr = 0.0, 0.0, 0.0
		if self.comm.Get_rank()==0:
			(lower, upper), (q25, q75, iqr) = compute_outliers(self.Eloc_real_g[-1,...])
		lower = self.comm.bcast(lower, root=0)
		upper = self.comm.bcast(upper, root=0)
		q25 = self.comm.bcast(q25, root=0)
		q75 = self.comm.bcast(q75, root=0)
		iqr = self.comm.bcast(iqr, root=0)

		# identify outliers
		#outliers = [Eloc_s for Eloc_s in self.Eloc_real if Eloc_s < lower or Eloc_s > upper]
		self.inds_outliers=np.where( np.logical_or(self.Eloc_real < lower , self.Eloc_real > upper) )[0]

		#outliers_str='Eloc outliers:\n   {0}\nwith inds:\n   {1}\n'.format(self.Eloc_real[self.inds_outliers],self.inds_outliers)
		outliers_stats_str='Percentiles: 25th={0:.3f}, 75th={1:.3f}, IQR={2:.3f}.\n'.format(q25, q75, iqr, )
		outliers_str='Eloc outliers:\n   {}\n'.format(self.Eloc_real[self.inds_outliers],)

		if self.logfile!=None:
			self.logfile.write(outliers_stats_str)
			self.logfile.write(outliers_str)
	
		print(outliers_stats_str)
		print(outliers_str)
		"""

		if SdotS:
			self.SdotS_real=2.0*self.Eloc_real # double off-diagonal contribution
			self.SdotS_real=self.Eloc_real+0.75*self.N_sites # diagonal contribution

			self.SdotS_imag=2.0*self.Eloc_imag # double off-diagonal contribution



	def process_local_energies(self,mode='MC',Eloc_params_dict=None,SdotS=False):

		if SdotS:
			loc=self.SdotS_real+1j*self.SdotS_imag
		else:
			loc=self.Eloc_real+1j*self.Eloc_imag


	
		if mode=='MC':

			Eloc_mean_g=np.zeros(1, dtype=np.complex128)
			Eloc_var_g=np.zeros(1, dtype=np.float64)

			# Eloc_mean=np.mean(loc).real
			# Eloc_var=np.sum( np.abs(loc)**2)/self.Eloc_real_tot.shape[0] - Eloc_mean**2

			self.comm.Allreduce(np.sum(       loc    ), Eloc_mean_g, op=MPI.SUM)
			Eloc_mean_g/=self.N_MC_points

			
			self.comm.Allreduce(np.sum(np.abs(loc)**2), Eloc_var_g,  op=MPI.SUM)
			Eloc_var_g/=self.N_MC_points
			Eloc_var_g-=np.abs(Eloc_mean_g)**2

			Eloc_mean_g=Eloc_mean_g[0]
			Eloc_var_g=Eloc_var_g[0]


		elif mode=='exact':
			abs_psi_2=Eloc_params_dict['abs_psi_2']
			Eloc_mean_g=np.sum(loc*abs_psi_2).real
			Eloc_var_g=np.sum(abs_psi_2*np.abs(loc)**2) - Eloc_mean_g**2
			

		if SdotS:
			E_diff_real=self.SdotS_real-Eloc_mean_g.real
			E_diff_imag=self.SdotS_imag-Eloc_mean_g.imag
		else:
			E_diff_real=self.Eloc_real-Eloc_mean_g.real
			E_diff_imag=self.Eloc_imag-Eloc_mean_g.imag


		self.Eloc_mean_g=Eloc_mean_g
		self.Eloc_var_g=Eloc_var_g

		return Eloc_mean_g, Eloc_var_g, E_diff_real, E_diff_imag



