import sys, os

from cpp_code import update_offdiag_ME, update_diag_ME, c_offdiag_sum

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
            #batch_idx = np.arange(i*minibatch_size, (i+1)*minibatch_size, 1)
            yield data[batch_idx], batch_idx,



class Energy_estimator_exact():

	def __init__(self,comm,DNN_log,DNN_phase,mode,J2,N_MC_points,N_batch,L,N_symm,sign,minibatch_size):

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

		


	def get_exact_kets(self,NN_type,MC_tool,N_features):


		self.MC_tool=MC_tool
		self.N_features=N_features

		#assert(self.L==4)

		ED_data_file  ="data-GS_J1-J2_Lx={0:d}_Ly={1:d}_J1=1.0000_J2={2:0.4f}.txt".format(self.L,self.L,self.J2)
		path_to_data=os.path.expanduser('~') + '/Google_Drive/frustration_from_RBM/ED_data/'
	
		skiprows=self.N_MC_points//self.comm.Get_size()*self.comm.Get_rank()
		if self.comm.Get_rank() == self.comm.Get_size()-1 and self.comm.Get_size()>1:
		 	skiprows+=1

		#print(self.comm.Get_rank(), skiprows, self.N_batch)

		ref_states=read_csv(path_to_data+ED_data_file, skiprows=skiprows, nrows=self.N_batch, header=None, dtype=self.basis_type, delimiter=' ',usecols=[0,]) 
		self.count=read_csv(path_to_data+ED_data_file, skiprows=skiprows, nrows=self.N_batch, header=None, dtype=np.uint16, delimiter=' ',usecols=[6,]) 
	
		
		self.MC_tool.ints_ket=ref_states.to_numpy().squeeze()
		self.MC_tool.count=self.count.to_numpy().squeeze()

		# compute spin s-configs
		integer_to_spinstate(self.MC_tool.ints_ket, self.MC_tool.spinstates_ket, self.N_features, NN_type=NN_type)

		
		# gather s data
		self.comm.Barrier()
		self.comm.Allgatherv([self.MC_tool.ints_ket,    self.MC_tool.MPI_basis_dtype], [self.MC_tool.ints_ket_g[:],   self.MC_tool.MPI_basis_dtype], )
		#self.comm.Gatherv([self.MC_tool.count,   MPI.SHORT], [self.MC_tool.count_g[:],   MPI.SHORT], root=0)
		

		### compute s'
		self.compute_s_primes(self.MC_tool.ints_ket,NN_type)

		
		# find indices of s primes
		self.s_prime_inds=np.searchsorted(self.MC_tool.ints_ket_g,self.ints_bra_uq,)
		
		# if self.comm.Get_rank()==1:
		# 	print(self.ints_bra_uq)
		# 	print(self.MC_tool.ints_ket_g)
		# 	print(self.s_prime_inds)
		# exit()


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

	def lookup_s_primes(self,data):
		return data[self.s_prime_inds][self.inv_index]






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


	def _reset_locenergy_params(self,):

		#######
		self._MEs=np.zeros(self.N_batch*self._n_offdiag_terms,dtype=np.float64)
		self._spinstates_bra=np.zeros((self.N_batch*self._n_offdiag_terms,self.N_sites*self.N_symm),dtype=np.int8)
		self._ints_bra_rep=np.zeros((self.N_batch*self._n_offdiag_terms,),dtype=self.basis_type)
		self._ints_ket_ind=np.zeros(self.N_batch*self._n_offdiag_terms,dtype=np.uint32)
		self._n_per_term=np.zeros(self._n_offdiag_terms,dtype=np.int32)

		self.Eloc_diag=np.zeros(self.N_batch, dtype=np.float64)	


	def _reset_Eloc_vars(self,):

		self._Eloc_cos=np.zeros(self.N_batch, dtype=np.float64)
		self._Eloc_sin=np.zeros(self.N_batch, dtype=np.float64)

		self.Eloc_real=np.zeros_like(self._Eloc_cos)
		self.Eloc_imag=np.zeros_like(self._Eloc_cos)
		

	def reestimate_local_energy_phase(self, iteration, NN_params_phase, batch, params_dict):

		phase_kets = self.DNN_phase.evaluate(NN_params_phase, batch.reshape(self.DNN_phase.input_shape))

		if self.mode=='ED':
			self.MC_tool.phase_kets[:]=phase_kets
			self.comm.Allgatherv([self.MC_tool.phase_kets,  MPI.DOUBLE], [self.MC_tool.phase_kets_g[:], MPI.DOUBLE], )
			phase_psi_bras = self.lookup_s_primes(self.MC_tool.phase_kets_g)
		else:
			sphase_psi_bras = self.evaluate_s_primes(self.DNN_phase.evaluate,NN_params_phase,self.DNN_phase.input_shape,)

		self.compute_Eloc(self.log_kets, phase_kets, self.log_psi_bras, phase_psi_bras, debug_mode=False,)

		Eloc_mean_g, Eloc_var_g, E_diff_real, E_diff_imag = self.process_local_energies(params_dict)

		params_dict['E_diff']=E_diff_imag
		params_dict['Eloc_mean']=Eloc_mean_g
		params_dict['Eloc_var']=Eloc_var_g
		#params_dict['Eloc_mean_part']=Eloc_mean_g.imag
		

		return params_dict, batch



	def compute_local_energy(self,params_log,params_phase,ints_ket,log_kets,phase_kets,log_psi_shift, verbose=True, ):
		
		ti=time.time()
		if not self.mode=='ED':
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
				log_psi_bras = self.lookup_s_primes(self.MC_tool.log_mod_kets_g)
				phase_psi_bras = self.lookup_s_primes(self.MC_tool.phase_kets_g)

			else:
				log_psi_bras = self.evaluate_s_primes(self.DNN_log.evaluate, params_log, self.DNN_log.input_shape,)	
				phase_psi_bras = self.evaluate_s_primes(self.DNN_phase.evaluate,params_phase,self.DNN_phase.input_shape,)

		else:
			if self.mode=='ED':
				log_psi_bras = self.lookup_s_primes(self.MC_tool.log_mod_kets_g)
				phase_psi_bras = self.lookup_s_primes(self.MC_tool.phase_kets_g)

			else:
				log_psi_bras, phase_psi_bras = self.evaluate_s_primes(self.DNN_log.evaluate, params_log, self.DNN_log.input_shape,)
		
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



	

	def evaluate_s_primes(self,evaluate_NN,NN_params,input_shape,):

		### evaluate network using minibatches

		if self.minibatch_size > 0:
		
			num_complete_batches, leftover = divmod(self.nn_uq, self.minibatch_size)
			N_minibatches = num_complete_batches + bool(leftover)

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
				batch=data[batch_idx].reshape(input_shape)

				if self.NN_dtype=='real':
					prediction_bras[batch_idx] = evaluate_NN(NN_params, batch,)
				else:
					prediction_bras[batch_idx], prediction_bras_2[batch_idx] = evaluate_NN(NN_params, batch,)
				
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
				prediction_bras = evaluate_NN(NN_params,self._spinstates_bra[:self.nn][self.index].reshape(input_shape),  )._value
			else:	
				prediction_bras, prediction_bras_2 = evaluate_NN(NN_params,self._spinstates_bra[:self.nn][self.index].reshape(input_shape),  )._value
			

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
		if debug_mode:
			self.debug_helper()


		return self.Eloc_real+1j*self.Eloc_imag



	def process_local_energies(self,Eloc_params_dict,):


		Eloc=self.Eloc_real+1j*self.Eloc_imag
		
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
			Eloc_var_g=Eloc_var_g[0]


		elif self.mode=='exact':
			abs_psi_2=Eloc_params_dict['abs_psi_2']
			Eloc_mean_g=np.sum(Eloc*abs_psi_2).real
			Eloc_var_g=np.sum(abs_psi_2*np.abs(Eloc)**2) - Eloc_mean_g**2
			
		elif self.mode=='ED':

			abs_psi_2=Eloc_params_dict['abs_psi_2']
			Eloc_mean=np.sum(abs_psi_2*Eloc)
			self.comm.Allreduce(np.sum(Eloc_mean), Eloc_mean_g, op=MPI.SUM)
			
			self.comm.Allreduce(np.sum(abs_psi_2*np.abs(Eloc)**2), Eloc_var_g,  op=MPI.SUM)
			Eloc_var_g-=np.abs(Eloc_mean_g)**2

			Eloc_mean_g=Eloc_mean_g[0]
			Eloc_var_g=Eloc_var_g[0]



		E_diff_real=self.Eloc_real-Eloc_mean_g.real
		E_diff_imag=self.Eloc_imag-Eloc_mean_g.imag


		self.Eloc_mean_g=Eloc_mean_g
		self.Eloc_var_g=Eloc_var_g

		return Eloc_mean_g, Eloc_var_g, E_diff_real, E_diff_imag



