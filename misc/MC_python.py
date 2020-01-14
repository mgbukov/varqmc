import numpy as np
from cpp_code import integer_to_spinstate, swap_spins
N_sites=16
N_symm=128


def sample(	 N_MC_points,
				thermalization_time,
				acceptance_ratio,
				#
				spin_states,
				ket_states,
				mod_kets,
				#
				s0_vec,
				DNN
				):

	N_accepted=0
	N_MC_proposals=np.zeros(1,dtype=np.int)
	# reduce MC points per chain
	auto_correlation_time = np.int(0.4/np.max([0.05, acceptance_ratio])*N_sites)

   
	N_accepted+=_MC_core(
							   N_MC_points,
							   N_MC_proposals,
							   thermalization_time,
							   auto_correlation_time,
							   #
							   spin_states,
							   ket_states,
							   mod_kets,
							   s0_vec,
							   #
							   0,
							   DNN
							)

   

	return N_accepted, np.sum(N_MC_proposals);


def _MC_core(	N_MC_points,
				N_MC_proposals,
				thermalization_time,
				auto_correlation_time,
				#
				spin_states,
				ket_states,
				mod_kets,
				s0_vec,
				#
				chain_n,
				DNN
	):		   
	
	i=0; 
	k=0; # counters
	N_accepted=0;
	spinstate_s = np.zeros(N_symm*N_sites,dtype=np.float64)
	spinstate_t = np.zeros(N_symm*N_sites,dtype=np.float64)


	s=(1<<(N_sites//2))-1;
	for l in range(N_sites):
		t=s;
		while(t==s):
			_i = np.random.randint(N_sites)
			_j = np.random.randint(N_sites)
			t = swap_spins(s,_i,_j);
		s=t;

   
	# store initial state
	s0_vec[chain_n] = s;

	
	# compute initial spin config and its amplitude value
	#self.spin_config(self.N_sites,s,spinstate_s);
	integer_to_spinstate(np.array([s],dtype=np.uint16), spinstate_s, N_symm*N_sites, NN_type='DNN')

   
	mod_psi_s=np.exp(DNN.evaluate_log(DNN.params, spinstate_s));
			
 

 
	while(k < N_MC_points):
		
		# propose a new state until a nontrivial configuration is drawn
		t=s;
		while(t==s):
			_i = np.random.randint(N_sites)
			_j = np.random.randint(N_sites)
			
			t = swap_spins(s,_i,_j);
		

		#self.spin_config(self.N_sites,t,&spinstate_t[0]);
		integer_to_spinstate(np.array([t],dtype=np.uint16), spinstate_t, 128*N_sites, NN_type='DNN')


		mod_psi_t=np.exp(DNN.evaluate_log(DNN.params, spinstate_t));


		# MC accept/reject step
		eps = np.random.uniform(0,1);
		if(eps*mod_psi_s[chain_n]*mod_psi_s[chain_n] <= mod_psi_t[chain_n]*mod_psi_t[chain_n]): # accept
			s = t;
			mod_psi_s[chain_n] = mod_psi_t[chain_n];
			# set spin configs
			for i in range(N_symm*N_sites):
				spinstate_s[i] = spinstate_t[i];
			N_accepted+=1;


		if( (N_MC_proposals[0] > thermalization_time) and (N_MC_proposals[0] % auto_correlation_time) == 0):
			
			for i in range(N_symm*N_sites):
				spin_states[k*N_sites*N_symm + i] = spinstate_s[i];

			ket_states[k] = s;
			mod_kets[k]=mod_psi_s[chain_n];


			k+=1;
			
		N_MC_proposals[0]+=1;


	return N_accepted;


