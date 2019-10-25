import sys,os

import numpy as np 
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Qt5Agg')

os.environ["PATH"] += ':/usr/local/texlive/2015/bin/x86_64-darwin'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=16)
plt.tick_params(labelsize=18)



##############################




N_MC_points=200


file_markus_MC = './observables_MC_2.txt'
Markus_MC = np.loadtxt(file_markus_MC, delimiter=' ')


file_markus_ED = './observables_exact.txt'
Markus_ED = np.loadtxt(file_markus_ED, delimiter=' ')


file_marin_MC = './data/data_files/energy--model_DNNcpx-mode_MC-L_4-J2_0.5-opt_NG-NNstrct_16--4-MCpts_{0:d}.txt'.format(N_MC_points)
Marin_MC = np.loadtxt(file_marin_MC, delimiter=' : ')


file_marin_ED = './data/data_files/energy--model_DNNcpx-mode_exact-L_4-J2_0.5-opt_NG-NNstrct_16--4-MCpts_{0:d}.txt'.format(107)
Marin_ED = np.loadtxt(file_marin_ED, delimiter=' : ')



N_iter_max=-1


# plt.plot(Markus_MC[:N_iter_max,0],Markus_MC[:N_iter_max,2],'b',label='Markus, N-MC=200')
# plt.plot(Markus_ED[:N_iter_max,0],Markus_ED[:N_iter_max,2],'--b',label='Markus, ED')

# plt.plot(Marin_MC[:N_iter_max,0],Marin_MC[:N_iter_max,1],'r',label='Marin, N-MC={0:d}'.format(N_MC_points))
# plt.plot(Marin_ED[:N_iter_max,0],Marin_ED[:N_iter_max,1],'--r',label='Marin, ED')

# plt.legend()
# plt.show()


plt.plot(Markus_MC[:N_iter_max,0],Markus_MC[:N_iter_max,2]-Markus_ED[:N_iter_max,2],'b',label='Markus, N-MC=200')

plt.plot(Marin_MC[:N_iter_max,0],Marin_MC[:N_iter_max,1]-Marin_ED[:N_iter_max,1],'r',label='Marin, N-MC={0:d}'.format(N_MC_points))

plt.legend()
plt.show()


