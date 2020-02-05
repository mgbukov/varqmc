import sys,os
import numpy as np 
import pickle
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

################

#debug_data_strs=['Eloc_data','SF_data','intkets_data','modpsi_data','phasepsi_data']

data_name='2020-02-05_01:18:37_NG/'
#data_name='2020-02-04_19:20:06_NG/'

load_dir='data/' + data_name + 'debug_files/' + 'debug-'

params_str='--model_DNNcpx-mode_MC-L_6-J2_0.5-opt_NG-NNstrct_36--6-MCpts_20000-Nprss_130-NMCchains_1'



with open(load_dir + 'Eloc_data' + params_str + '.pkl', 'rb') as handle:
	Eloc_real, Eloc_imag = pickle.load(handle)

with open(load_dir + 'SF_data' + params_str + '.pkl', 'rb') as handle:
	S_lastiters, F_lastiters, delta = pickle.load(handle)

with open(load_dir + 'modpsi_data' + params_str + '.pkl', 'rb') as handle:
	modpsi_kets, = pickle.load(handle)

with open(load_dir + 'phasepsi_data' + params_str + '.pkl', 'rb') as handle:
	phasepsi_kets, = pickle.load(handle)

with open(load_dir + 'intkets_data' + params_str + '.pkl', 'rb') as handle:
	int_kets, = pickle.load(handle)


print(modpsi_kets[0,0], np.log(modpsi_kets[0,0]))
exit()


def int_to_spinconfig(s):
	S=np.array(list("{0:036b}".format(s)))
	print(S.reshape(6,6))


n=-3



print(np.max(np.abs(Eloc_real[n,:])))

inds=np.where(np.abs(Eloc_real[n,:])>500)[0]

print()


for ind in inds:
	s=int_kets[n,:][ind]
	print(s, Eloc_real[n,:][ind], modpsi_kets[n,:][ind])
	int_to_spinconfig(s)
	print()




