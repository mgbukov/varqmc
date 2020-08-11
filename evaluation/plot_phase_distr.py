import os, sys
import numpy as np
import pickle
import csv

path = "../."
sys.path.insert(0,path)
from eval_lib import *
#from cpp_code import integer_to_spinstate

import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.ticker import LogLocator,LinearLocator

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


matplotlib.use('Qt5Agg')

os.environ["PATH"] += ':/usr/local/texlive/2015/bin/x86_64-darwin'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.tick_params(labelsize=20)


##############################################
def format_func(value, tick_number):
	# find number of multiples of pi/2
	N = int(np.round(2 * value / np.pi))
	if N == 0:
		return "0"
	elif N == 1:
		return r"$\pi/2$"
	elif N == -1:
		return r"$-\pi/2$"
	elif N == 2:
		return r"$\pi$"
	elif N == -2:
		return r"$-\pi$"
	elif N % 2 > 0:
		return r"${0}\pi/2$".format(N)
	else:
		return r"${0}\pi$".format(N // 2)




##############################################

save= False # True # 



iteration=600
J2=0.5
L=6
opt='CNNreal-RK_RK' # 'sgd_sgd' # 'sgd_sgd' #  
cost='SR_SR' # 'SR_SR' #
mode='ED' # 'MC' # 'exact' #
NN_type='CNNreal'
sys_time='2020_07_24-18_57_48' 





data_name = sys_time + '--{0:s}-{1:s}-L_{2:d}-{3:s}/'.format(opt,cost,L,mode)
load_file_name='phase_data_'+data_name[:-1]+'_iter={0:d}'.format(iteration)


with open(load_file_name+'.pkl', 'rb') as handle:
	log_psi, phase_psi,  rep_spin_configs_ints, log_psi_bras, phase_psi_bras, rep_spin_configs_bras_ints = pickle.load(handle)




# wrap phases
phase_psi      = (phase_psi+np.pi)%(2*np.pi)      - np.pi
phase_psi_bras = (phase_psi_bras+np.pi)%(2*np.pi) - np.pi

# shift phases
ind=np.argmax(log_psi)
a=phase_psi[ind]
b=log_psi[ind]

phase_psi=phase_psi-a
phase_psi_bras=phase_psi_bras-a

log_psi=log_psi-b
log_psi_bras=log_psi_bras-b


#####


eps=5E-1
ind_, =np.where((np.abs(phase_psi_bras)>eps) * (np.abs(phase_psi_bras+np.pi)>eps))

phase_psi_bras=phase_psi_bras[ind_]
log_psi_bras=log_psi_bras[ind_]
rep_spin_configs_bras_ints=rep_spin_configs_bras_ints[ind_]


#rep_spin_configs_bras_ints=np.unique(rep_spin_configs_bras_ints)

rep_spin_configs_bras_ints_uq, index, inv_index, = np.unique(rep_spin_configs_bras_ints, return_index=True, return_inverse=True, )

log_psi_sample, sign_psi_sample, mult_sample, p_sample, sign_psi_sample_J2_0 = extract_ED_data(rep_spin_configs_bras_ints_uq,L,J2)



N_disagree=np.sum(0.5*np.abs(sign_psi_sample-sign_psi_sample_J2_0))
N_all=rep_spin_configs_bras_ints_uq.shape[0]

print(N_disagree, N_all)

###### DATA ######:: (eps, %) = (1E-1, 0.831), (2E-1, 0.842), (5E-1, 0.868), (1E-0, 0.901)


print("\n\nPercentage of configs where exact and AFM Marshall signs do NOT agree: {0:.3f} at eps={1:0.6f}.\n\n ".format(N_disagree/N_all, eps) )





phase_psi_sample= -(sign_psi_sample+1)*np.pi/2


#log_psi_sample=log_psi_sample[inv_index]
#phase_psi_sample=phase_psi_sample[inv_index]

# shift exact values

ind_log_bras=np.argmax(log_psi_bras)
log_bras_max=log_psi_bras[ind_log_bras]

print(2*log_bras_max, 2*np.max(log_psi_sample), 2*log_psi_sample[ind_log_bras])

log_psi_sample=log_psi_sample-log_psi_sample[ind_log_bras]+log_bras_max




############

fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True)

xlim=[-2*np.pi,2*np.pi]
ylim=[-20.0,1.0]

ax[0].plot(phase_psi,2.0*log_psi,'.b',markersize=0.5)
ax[0].set_xlim(xlim)
ax[0].set_ylim(ylim)
ax[0].set_xlabel('$\\varphi_s$')
ax[0].set_ylabel('$2\\log|\\psi_s|$')
ax[0].set_title('$s$-configs')
ax[0].xaxis.set_major_locator(plt.MultipleLocator(np.pi))
ax[0].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
ax[0].grid(color='k', linestyle='-', linewidth=0.1)
ax[0].xaxis.set_ticks_position('both')
ax[0].yaxis.set_ticks_position('both')

ax[1].plot(phase_psi_bras,2.0*log_psi_bras,'.r',markersize=0.5)
ax[1].scatter(phase_psi_bras[ind_log_bras],2.0*log_psi_bras[ind_log_bras],c='r',marker='x',)
ax[1].set_xlim([-2*np.pi+0.05,2*np.pi])
ax[1].set_ylim(ylim)
ax[1].set_xlabel('$\\varphi_s$')
ax[1].set_title("$s'$-configs")
ax[1].xaxis.set_major_locator(plt.MultipleLocator(np.pi))
ax[1].xaxis.set_major_formatter(plt.FuncFormatter(format_func))	
ax[1].grid(color='k', linestyle='-', linewidth=0.1)
ax[1].xaxis.set_ticks_position('both')
ax[1].yaxis.set_ticks_position('both')

ax[2].plot(phase_psi_sample,2.0*log_psi_sample,'.m',markersize=0.5)
ax[2].scatter(phase_psi_sample[ind_log_bras],2.0*log_psi_sample[ind_log_bras],c='m',marker='x',)
ax[2].set_xlim([-2*np.pi+0.05,2*np.pi])
ax[2].set_ylim(ylim)
ax[2].set_xlabel('$\\varphi_s$')
ax[2].set_title("$s'$-configs, exact")
ax[2].xaxis.set_major_locator(plt.MultipleLocator(np.pi))
ax[2].xaxis.set_major_formatter(plt.FuncFormatter(format_func))	
ax[2].grid(color='k', linestyle='-', linewidth=0.1)
ax[2].xaxis.set_ticks_position('both')
ax[2].yaxis.set_ticks_position('both')


plt.subplots_adjust(hspace=0,wspace=0,top=0.9, bottom=0.15, left=0.15, right=0.95)
#plt.tight_layout()


if save:
	plt.savefig('phase_distr--iter_{0:05d}.pdf'.format(iteration))
	plt.close()
else:
	plt.show()







