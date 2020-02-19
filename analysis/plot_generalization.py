import sys,os
import numpy as np 
import pickle, yaml

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


####################

L=4
N_batch=10
N_iter=480


# NN params
data_name='2020-02-14_09:47:56_NG/' # 4x4, MC
#data_name='2020-02-17_09:15:17_NG/' # 4x4, ED

load_dir='data/' + data_name 
if L==6:
	params_str='model_DNNcpx-mode_MC-L_6-J2_0.5-opt_NG-NNstrct_36--6-MCpts_20000-Nprss_130-NMCchains_1'
elif L==4:
	params_str='model_DNNcpx-mode_MC-L_4-J2_0.5-opt_NG-NNstrct_16--6-MCpts_200-Nprss_4-NMCchains_2'
	#params_str='model_DNNcpx-mode_exact-L_4-J2_0.5-opt_NG-NNstrct_16--6-MCpts_107-Nprss_1-NMCchains_2' # ED
else:
	'exiting'

# load cost functions
file_name=load_dir + 'cost_funcs--' + params_str + '.txt'
data=np.loadtxt(file_name,delimiter=':',)

# load energy
file_name=load_dir + 'data_files/energy--' + params_str + '.txt'
energy=np.loadtxt(file_name,delimiter=':',)


iterations=data[:,0].astype(int)

C_KL_div=data[:,1]
C_cross_entropy=data[:,2]

C_overlap=data[:,3]
C_psi_L2 = data[:,4]

C_log_psi_L2=data[:,5]
C_sign_psi_L2=data[:,6]

C_psi_L1_weighted=data[:,7]
C_log_psi_L1_weighted=data[:,8]
C_sign_psi_L1_weighted=data[:,9]


#plt.plot(iterations, energy[iterations,1], 'b', label='energy', )

plt.plot(iterations, C_KL_div, 'r', label='KL: $\\sum_s p^\\mathrm{GS}_s\\log(p^\\mathrm{GS}_s/p^\\mathrm{DNN}_s)$', )
plt.plot(iterations, C_cross_entropy, '--b', label='cross ent: $-\\sum_s p^\\mathrm{GS}_s\\log(p^\\mathrm{DNN}_s)$', )
plt.plot(iterations, C_overlap, '-.g', label='$1-|\\langle\\psi^\\mathrm{GS}|\\psi^\\mathrm{DNN}\\rangle|^2$', )

plt.xlabel('iteration')
plt.legend()

plt.grid()
plt.tight_layout()
plt.show()



plt.plot(iterations, C_psi_L2, 'g', label='$\\sqrt{\\sum_s c_s|\\psi^\\mathrm{GS}_s-\\psi^\\mathrm{DNN}_s|^2}$', )
plt.plot(iterations, C_log_psi_L2, '--r', label='$\\sqrt{N_\\mathrm{dim}^{-1}\\sum_s c_s|\\log(|\\psi^\\mathrm{GS}_s|)-\\log(|\\psi^\\mathrm{DNN}_s|)|^2}$', )
plt.plot(iterations, C_sign_psi_L2, '-.b', label='$1-|N_\\mathrm{dim}^{-1}\\sum_s c_s\\mathrm{sign}(\\psi^\\mathrm{GS}_s)\\mathrm{e}^{i \\phi^\\mathrm{DNN}_s}|$', )

plt.xlabel('iteration')
plt.legend()

plt.grid()
plt.tight_layout()
plt.show()



plt.plot(iterations, C_psi_L1_weighted, 'g', label='$\\sum_s p^\\mathrm{GS}_s |\\psi^\\mathrm{GS}_s-\\psi^\\mathrm{DNN}_s|$', )
plt.plot(iterations, C_log_psi_L1_weighted, '--r', label='$\\sum_s p^\\mathrm{GS}_s |\\log(|\\psi^\\mathrm{GS}_s|)-\\log(|\\psi^\\mathrm{DNN}_s|)|$', )
plt.plot(iterations, C_sign_psi_L1_weighted, '-.b', label='$1-|\\sum_s p^\\mathrm{GS}_s \\mathrm{sign}(\\psi^\\mathrm{GS}_s)\\mathrm{e}^{i \\phi^\\mathrm{DNN}_s}|$', )

plt.xlabel('iteration')
plt.legend()

plt.grid()
plt.tight_layout()
plt.show()





