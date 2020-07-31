import sys,os
import numpy as np 
import pickle

#sys.path.append("..")

# from cpp_code import integer_to_spinstate, representative
# from cpp_code import update_offdiag_ME, update_diag_ME, c_offdiag_sum
# from VMC_class import VMC
import yaml 

from pandas import read_csv


###

L=6
J2=0.5
N_batch=10000

ED_data_file  ="data-GS_J1-J2_Lx={0:d}_Ly={1:d}_J1=1.0000_J2={2:0.4f}.txt".format(L,L,J2)
ED_data_file_2="data-GS_J1-J2_Lx={0:d}_Ly={1:d}_J1=1.0000_J2={2:0.4f}.txt".format(L,L,0.0)
path_to_data=os.path.expanduser('~') + '/Google_Drive/frustration_from_RBM/ED_data/'


if L==6:
	N_data=15804956
	basis_dtype=np.uint64 
	
elif L==4:
	N_data=107
	basis_dtype=np.uint16
else:
	print('exiting')
	exit()

mult_ED_it=read_csv(path_to_data+ED_data_file, chunksize=N_batch, header=None, dtype=np.float64,delimiter=' ',usecols=[6,]) 
spin_int_states_it=read_csv(path_to_data+ED_data_file, chunksize=N_batch, header=None, dtype=basis_dtype,delimiter=' ',usecols=[0,]) 
	

sign_psi_ED_it=read_csv(path_to_data+ED_data_file, chunksize=N_batch, header=None, dtype=np.float64,delimiter=' ',usecols=[4,]) 
sign_psi_ED_it_J2_0=read_csv(path_to_data+ED_data_file_2, chunksize=N_batch, header=None, dtype=np.float64,delimiter=' ',usecols=[4,]) 

log_psi_ED_it=read_csv(path_to_data+ED_data_file, chunksize=N_batch, header=None, dtype=np.float64,delimiter=' ',usecols=[1,]) 

mult_ED_gtr=[]
log_psi_ED_gtr=[]
log_psi_ED_max=-100.0

norm_mismatch=0
norm=0

n_match=0
n=0
for j, (spin_configs_ED, log_psi_ED, sign_psi_ED, mult_ED,  sign_psi_ED_J2_0) in enumerate(zip(spin_int_states_it,log_psi_ED_it, sign_psi_ED_it, mult_ED_it, sign_psi_ED_it_J2_0)):
	
	
	sign      =  sign_psi_ED.to_numpy().squeeze()      
	sign_J2_0 =  sign_psi_ED_J2_0.to_numpy().squeeze() 

	spin_configs_ED = spin_configs_ED.to_numpy().squeeze()

	log_psi_ED=log_psi_ED.to_numpy().squeeze()
	mult_ED=mult_ED.to_numpy().squeeze()

	
	log_psi_ED_max=max(log_psi_ED_max, np.max(log_psi_ED) )
	

	P_ED=np.exp(2.0*log_psi_ED)

	P=sign*sign_J2_0
	inds,=np.where(P>0)

	# mismatch configs
	inds2,=np.where(P<0)

	log_psi_ED_gtr+=list(log_psi_ED[inds2])
	mult_ED_gtr+=list(mult_ED[inds2])


	norm_mismatch+=np.sum(mult_ED[inds2]*P_ED[inds2])
	norm+=np.sum(mult_ED*P_ED)

	n_match+=np.sum( mult_ED[inds] )	
	n+=np.sum(mult_ED) #N_batch


log_psi_ED_gtr=np.array(log_psi_ED_gtr).squeeze()
mult_ED_gtr=np.array(mult_ED_gtr).squeeze()



print('max log_psi_ED: {0:0.10f}'.format(log_psi_ED_max))

a=2*8.0
inds_gtr,=np.where(a>=np.abs(log_psi_ED_gtr-log_psi_ED_max))

norm_gtr=np.sum(mult_ED_gtr[inds_gtr]*np.exp(2.0*log_psi_ED_gtr[inds_gtr]))

print(np.min(log_psi_ED_gtr), np.max(log_psi_ED_gtr), inds_gtr.shape, log_psi_ED_gtr.shape, norm_gtr )



print("\nmatching signs between J2=0 and J2=0.5: {0:0.4f} %".format(n_match/n),)
print("GS-weighted norm of mismatch configurations: {0:4f}\n".format(norm_mismatch), )



