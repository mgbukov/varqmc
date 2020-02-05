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

load_dir='data/2020-02-03_12:43:02_NG/'

params_str='model_DNNcpx-mode_MC-L_6-J2_0.5-opt_NG-NNstrct_36--6-MCpts_20000-Nprss_130-NMCchains_1'



fig, axs = plt.subplots(2,6,figsize=(15,5))


iteration=300



file_name='NNparams'+'--iter_{0:05d}--'.format(iteration) + params_str

with open(load_dir + 'NN_params/' +file_name+'.pkl', 'rb') as handle:
	NN_params = pickle.load(handle)

mins=[]
maxs=[]
for params in NN_params[0]:

	mins.append(np.min(np.abs(params)) )
	maxs.append(np.max(np.abs(params)) )


W_real=NN_params[0][0]
W_imag=NN_params[0][1]




for j in range(6):
	axs[0,j].imshow(W_real[:,j].reshape(6,6), vmin=-max(maxs), vmax=max(maxs) )
	axs[1,j].imshow(W_imag[:,j].reshape(6,6), vmin=-max(maxs), vmax=max(maxs) )

fig.suptitle('iter = {0:03d}'.format(iteration))


plt.show()

	# plt.draw() # draw frame
	# plt.pause(0.1) # pause frame
	# #plt.clf() # clear figure


