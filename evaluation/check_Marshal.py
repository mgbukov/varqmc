import sys,os
import numpy as np 
import pickle
import jax

path = "../."
sys.path.insert(0,path)
from cpp_code import NN_Tree

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.ticker import LogLocator,LinearLocator

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


from scipy.linalg import eigh, inv


matplotlib.use('Qt5Agg')

os.environ["PATH"] += ':/usr/local/texlive/2015/bin/x86_64-darwin'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.tick_params(labelsize=20)


################

from eval_lib import *

sys.path.append("..")

from cpp_code import integer_to_spinstate
from VMC_class import VMC
import yaml 


#########################

save= True # False # 

# iteration=1995 # 100 # 200 # 500 # 1000 # 1500 #
# J2=0.5
# L=6
# opt='RK_RK' # 'sgd_sgd' # 'sgd_sgd' #  
# cost='SR_SR' # 'SR_SR' #
# mode='MC' # 'exact' #
# sys_time='2020_06_13-09_51_06' # '2020_06_05-21_55_27' # 

iteration=600 #100,150,200,250,300,350,500,600,
J2=0.5
L=6
opt='CNNreal-RK_RK' # 'sgd_sgd' # 'sgd_sgd' #  
cost='SR_SR' # 'SR_SR' #
mode='ED' # 'MC' # 'exact' #
NN_type='CNNreal'
sys_time='2020_07_24-18_57_48' 



#### load debug data


data_name = sys_time + '--{0:s}-{1:s}-L_{2:d}-{3:s}/'.format(opt,cost,L,mode)
#load_dir='data/' + data_name 
#load_dir='data/paper_data/seeds/' + data_name 
#load_dir='data/paper_data/MC_samples/' + data_name
load_dir='data/paper_data/exact/' + data_name  

#data_params=(NN_dtype,mode,L,J2,opt,NN_shape_str,N_MC_points,N_prss,NMCchains,)
#params_str='--model_DNN{0:s}-mode_{1:s}-L_{2:d}-J2_{3:0.1f}-opt_{4:s}-NNstrct_{5:s}-MCpts_{6:d}-Nprss_{7:d}-NMCchains_{8:d}'.format(*data_params)
params_str=''

# with open(load_dir + 'debug_files/' + 'debug-' + 'intkets_data--' + params_str + '.pkl', 'rb') as handle:
# 	int_kets, = pickle.load(handle)
# 	if L==4:
# 		int_kets.astype(np.uint16)
# 	else:
# 		int_kets.astype(np.uint64) 


# with open(load_dir + 'debug_files/' + 'debug-' + 'Eloc_data--' + params_str + '.pkl', 'rb') as handle:
# 	Eloc_real, Eloc_imag = pickle.load(handle)

######################

#file_name='NNparams'+'--iter_{0:05d}--'.format(iteration) + params_str
file_name='NNparams'+'--iter_{0:05d}'.format(iteration)

with open(load_dir + 'NN_params/' +file_name+'.pkl', 'rb') as handle:
	params_log, params_phase, apply_fun_args_log, apply_fun_args_phase, log_psi_shift = pickle.load(handle)




######################

print('\niteration number: {0:d}.\n'.format(iteration,))


######################

if L==4:
	E_estimator, basis_states, index, inv_index, count = ED_results(load_dir,)
	psi_GS=E_estimator.psi_GS_exact[index]


############


#overlap=evaluate_overlap(load_dir,params_log, params_phase, L, J2,)
overlap=-1.0
#exit()

N_MC_points=1000 # MC points

#with jax.disable_jit():
MC_tool = MC_sample(load_dir, params_log, N_MC_points=N_MC_points, reps=True)

#rep_spin_configs_ints=compute_reps(int_kets[n,:],L)
rep_spin_configs_ints=compute_reps(MC_tool.ints_ket,L)
#rep_spin_configs_ints=basis_states

#print(rep_spin_configs_ints)
#print(MC_tool.ints_ket)


################



log_psi, phase_psi = evaluate_DNN(load_dir, params_log, params_phase, rep_spin_configs_ints, )
sign_psi = np.exp(-1j*phase_psi)

log_psi, phase_psi, log_psi_bra, phase_psi_bra, rep_spin_configs_bra_ints = evaluate_sample(load_dir,params_log, params_phase,rep_spin_configs_ints,log_psi,phase_psi,)

Eloc_Re, Eloc_Im=compute_Eloc(load_dir,params_log, params_phase,rep_spin_configs_ints,log_psi,phase_psi,)

# local E for exact states
print('DNN:', Eloc_Re.mean(), Eloc_Im.mean())

#exit()

################


# get phases
log_psi_ED, sign_psi_ED, mult_ED, p_ED, sign_psi_ED_J2_0 = extract_ED_data(rep_spin_configs_ints,L,J2)
phase_psi_ED = np.pi*0.5*(sign_psi_ED+1.0)
phase_psi_ED_J2_0 = np.pi*0.5*(sign_psi_ED_J2_0+1.0)

#print('log-loss', np.mean(np.abs(log_psi-log_psi[0]-(log_psi_ED-log_psi_ED[0]) )))


# evalute exact Eloc
Eloc_real_ED ,Eloc_imag_ED, log_psi_bra_ED, phase_psi_bra_ED, phase_psi_bra_ED_J2_0 = compute_Eloc_ED(load_dir,rep_spin_configs_ints,log_psi_ED,phase_psi_ED,L,J2, return_ED_data=True)

print('exact:',Eloc_real_ED.mean(), Eloc_imag_ED.mean())


Eloc_real_ED_J2_0 ,Eloc_imag_ED_J2_0 = compute_Eloc_ED(load_dir,rep_spin_configs_ints,log_psi_ED,phase_psi_ED_J2_0,L,0.0)


########################################
if save:

	save_file_name='phase_data_'+data_name[:-1]+'_iter={0:d}'.format(iteration)

	with open(save_file_name+'.pkl', 'wb') as handle:
						pickle.dump([log_psi, phase_psi,  rep_spin_configs_ints, 
									 log_psi_bra, phase_psi_bra, rep_spin_configs_bra_ints ], 
										handle, protocol=pickle.HIGHEST_PROTOCOL
									)


########################################


#check logs
ind=np.argmax(log_psi)
ind_bra=np.argmax(log_psi_bra)

log_psi-=log_psi_bra[ind]
log_psi_bra-=log_psi_bra[ind_bra]


# take only large amps
print(log_psi_bra.shape)
inds,=np.where(2.0*log_psi_bra>-10)
log_psi_bra=log_psi_bra[inds]
phase_psi_bra=phase_psi_bra[inds]
log_psi_bra_ED= log_psi_bra_ED[inds]
phase_psi_bra_ED= phase_psi_bra_ED[inds]
phase_psi_bra_ED_J2_0=phase_psi_bra_ED_J2_0[inds]

print(inds.shape, log_psi_bra.shape)
print('\n\n\n\n\n')


C_log     = np.mean( np.abs( (log_psi_ED - log_psi_ED[ind]) - (log_psi - log_psi[ind]) ) )
C_log_max = np.max(  np.abs( (log_psi_ED - log_psi_ED[ind]) - (log_psi - log_psi[ind]) ) )

C_sign_psi      = 1.0 - np.abs( np.sum(     sign_psi_ED*sign_psi ) )/N_MC_points
C_sign_psi_J2_0 = 1.0 - np.abs( np.sum(sign_psi_ED_J2_0*sign_psi ) )/N_MC_points
C_sign_psi_ED   = 1.0 - np.abs( np.sum(sign_psi_ED_J2_0*sign_psi_ED ) )/N_MC_points




#C_sign=np.abs(np.sum(sign_psi_ED*sign_psi_ED_J2_0))/N_MC_points
#C_sign=sign_psi_ED.dot(sign_psi_ED_J2_0) - (np.sum(sign_psi_ED)*np.sum(sign_psi_ED_J2_0))

print('costs:\n')
print(C_log,C_log_max)
print(C_sign_psi, C_sign_psi_J2_0, C_sign_psi_ED)
print()


print('DNN:               E_real={0:0.8f}, E_imag={1:0.8f}, E_std={2:0.8f}.'.format(np.mean(Eloc_Re)        , Eloc_Im.mean()          , np.std(Eloc_Re)          /np.sqrt(N_MC_points)) )
print('exact:             E_real={0:0.8f}, E_imag={1:0.8f}, E_std={2:0.8f}.'.format(Eloc_real_ED.mean()     , Eloc_imag_ED.mean()     , np.std(Eloc_real_ED)     /np.sqrt(N_MC_points)) )
print('exact (J2=0-sign): E_real={0:0.8f}, E_imag={1:0.8f}, E_std={2:0.8f}.\n'.format(Eloc_real_ED_J2_0.mean(), Eloc_imag_ED_J2_0.mean(), np.std(Eloc_real_ED_J2_0)/np.sqrt(N_MC_points)) )




#print(np.sort(Eloc_real_ED_J2_0))

# print()
# print(phase_histpgram(phase_psi))
# print()
# print(phase_histpgram(phase_psi_ED))
# print()
# print(phase_histpgram(phase_psi_ED_J2_0))

a=phase_psi[ind]
b=phase_psi_ED[ind]
c=phase_psi_ED_J2_0[ind]

phase_psi=phase_psi-a
phase_psi_ED=phase_psi_ED-b
phase_psi_ED_J2_0=phase_psi_ED_J2_0-c

inds=np.where(np.cos(phase_psi_ED_J2_0-phase_psi_ED)<0.0)[0]

phase_hist_ED, _ = np.histogram(np.cos(phase_psi_ED_J2_0-phase_psi_ED) ,bins=2,range=(-1.0,1.0), density=False, )
phase_hist, _ = np.histogram(np.cos(phase_psi_ED[inds]-phase_psi[inds]) ,bins=2,range=(-1.0,1.0), density=False, )
phase_hist_J2_0, _ = np.histogram(np.cos(phase_psi_ED_J2_0[inds]-phase_psi[inds]) ,bins=2,range=(-1.0,1.0), density=False, )


phase_hist_all, _ = np.histogram(np.cos(phase_psi_ED-phase_psi) ,bins=2,range=(-1.0,1.0), density=False, )
phase_hist_J2_0_all, _ = np.histogram(np.cos(phase_psi_ED_J2_0-phase_psi) ,bins=2,range=(-1.0,1.0), density=False, )



print_str='ED(J2=0.5)  vs  ED(J2=0)     T:F  :  {0:d}:{1:d}\n'.format(phase_hist_ED[1]  ,phase_hist_ED[0])
print_str+='mismatch sample\n'
print_str+='DNN         vs  ED(J2=0.5)   T:F  :  {0:d}:{1:d}\n'.format(phase_hist[1]     ,phase_hist[0])
print_str+='DNN         vs  ED(J2=0)     T:F  :  {0:d}:{1:d}\n'.format(phase_hist_J2_0[1],phase_hist_J2_0[0])
print_str+='full sample\n'
print_str+='DNN         vs  ED(J2=0.5)   T:F  :  {0:d}:{1:d}\n'.format(phase_hist_all[1]     ,phase_hist_all[0])
print_str+='DNN         vs  ED(J2=0)     T:F  :  {0:d}:{1:d}\n\n\n'.format(phase_hist_J2_0_all[1],phase_hist_J2_0_all[0])




#######
# bras


phase_psi_bra=phase_psi_bra-a
phase_psi_bra_ED=phase_psi_bra_ED-b
phase_psi_bra_ED_J2_0=phase_psi_bra_ED_J2_0-c

inds=np.where(np.cos(phase_psi_bra_ED_J2_0-phase_psi_bra_ED)<0.0)[0]

phase_hist_bra_ED, _ = np.histogram(np.cos(phase_psi_bra_ED_J2_0-phase_psi_bra_ED) ,bins=2,range=(-1.0,1.0), density=False, )
phase_hist_bra, _ = np.histogram(np.cos(phase_psi_bra_ED[inds]-phase_psi_bra[inds]) ,bins=2,range=(-1.0,1.0), density=False, )
phase_hist_bra_J2_0, _ = np.histogram(np.cos(phase_psi_bra_ED_J2_0[inds]-phase_psi_bra[inds]) ,bins=2,range=(-1.0,1.0), density=False, )


phase_hist_bra_all, _ = np.histogram(np.cos(phase_psi_bra_ED-phase_psi_bra) ,bins=2,range=(-1.0,1.0), density=False, )
phase_hist_bra_J2_0_all, _ = np.histogram(np.cos(phase_psi_bra_ED_J2_0-phase_psi_bra) ,bins=2,range=(-1.0,1.0), density=False, )



print_str+='ED(J2=0.5)  vs  ED(J2=0)     T:F  :  {0:d}:{1:d}\n'.format(phase_hist_bra_ED[1]  ,phase_hist_bra_ED[0])
print_str+='mismatch sample\n'
print_str+='DNN         vs  ED(J2=0.5)   T:F  :  {0:d}:{1:d}\n'.format(phase_hist_bra[1]     ,phase_hist_bra[0])
print_str+='DNN         vs  ED(J2=0)     T:F  :  {0:d}:{1:d}\n'.format(phase_hist_bra_J2_0[1],phase_hist_bra_J2_0[0])
print_str+='full sample\n'
print_str+='DNN         vs  ED(J2=0.5)   T:F  :  {0:d}:{1:d}\n'.format(phase_hist_bra_all[1]     ,phase_hist_bra_all[0])
print_str+='DNN         vs  ED(J2=0)     T:F  :  {0:d}:{1:d}\n'.format(phase_hist_bra_J2_0_all[1],phase_hist_bra_J2_0_all[0])


print_str+='\n\n overlap |<psi_NN|psi_GS>|^2 :  {0:0.6f}'.format(overlap)


print(print_str)
#exit()


if save:

	save_file_name='phase_data_'+data_name[:-1]+'_iter={0:d}'.format(iteration)

	text_file = open(save_file_name+'.txt', "w")
	text_file.write(print_str)
	text_file.close()






