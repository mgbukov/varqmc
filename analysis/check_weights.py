import sys,os
import numpy as np 
import pickle

path = "../."
sys.path.insert(0,path)
from cpp_code import NN_Tree

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

n=-3 # steps before final blow-up
max_iter=228 # last iteration with saved E-data + 1
L=6
J2=0.5
opt='NG'
mode='MC'
NN_shape_str='36--8'
N_MC_points=20000
N_prss=130
NMCchains=1
sys_time='2020-02-24_23:19:34'


#### load debug data


data_name = sys_time + '--{0:s}-L_{1:d}-{2:s}/'.format(opt,L,mode)
load_dir='data/' + data_name 
data_params=(mode,L,J2,opt,NN_shape_str,N_MC_points,N_prss,NMCchains,)
params_str='model_DNNcpx-mode_{0:s}-L_{1:d}-J2_{2:0.1f}-opt_{3:s}-NNstrct_{4:s}-MCpts_{5:d}-Nprss_{6:d}-NMCchains_{7:d}'.format(*data_params)


with open(load_dir + 'debug_files/' + 'debug-' + 'params_update_data--' + params_str + '.pkl', 'rb') as handle:
	NN_params_update, = pickle.load(handle)
	NN_params_update[:-1,...]=NN_params_update[1:,...]
	NN_params_update[-1,...]=0.0


for n in range(-6,-1,1):


	iteration=max_iter+n+1

	file_name='NNparams'+'--iter_{0:05d}--'.format(iteration) + params_str

	with open(load_dir + 'NN_params/' +file_name+'.pkl', 'rb') as handle:
		NN_params, apply_fun_args, log_psi_shift = pickle.load(handle)

	Tree=NN_Tree(NN_params)
	NN_params_ravelled=Tree.ravel(NN_params)



	print("b-value:", NN_params_ravelled[-1], )
	print("b-update:", -1E-2*NN_params_update[n,:][-1], )


	plt.plot(NN_params_ravelled,'.b')
	plt.plot(NN_params_ravelled-1E-2*NN_params_update[n,:],'.r',markersize=1.)

	plt.show()



##########################################

mins=[]
maxs=[]
for params in NN_params[0]:

	mins.append(np.min(np.abs(params)) )
	maxs.append(np.max(np.abs(params)) )


W_real=NN_params[0][0]
W_imag=NN_params[0][1]



N_hidden=8
fig, axs = plt.subplots(2,N_hidden,figsize=(15,5))

for j in range(N_hidden):
	axs[0,j].imshow(W_real[:,j].reshape(6,6), vmin=-max(maxs), vmax=max(maxs) )
	axs[1,j].imshow(W_imag[:,j].reshape(6,6), vmin=-max(maxs), vmax=max(maxs) )

fig.suptitle('iter = {0:03d}'.format(iteration))
plt.show()





