import sys,os
import numpy as np 
import pickle
import yaml 
#import jax

path = "../."
sys.path.insert(0,path)
from plot_lib import *
from eval_lib import *

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



###################


iteration=999
L=6
J2=0.5
opt='sgd_sgd' # 'RK_RK'
cost='SR_SR'
mode='MC' # 'exact' #
NN_dtype='real-decoupled'
NN_shape_str='({0:d}--12,{0:d}--24--12)'.format(L**2)
N_MC_points=10000 # 107 # 
N_prss=260 # 1 #  
NMCchains=1 # 
sys_time= '2020_03_30-12_34_34' 



#### load debug data


data_name = sys_time + '--{0:s}-{1:s}-L_{2:d}-{3:s}/'.format(opt,cost,L,mode)
load_dir='data/' + data_name  
data_params=(NN_dtype,mode,L,J2,opt,NN_shape_str,N_MC_points,N_prss,NMCchains,)
#params_str='--model_DNN{0:s}-mode_{1:s}-L_{2:d}-J2_{3:0.1f}-opt_{4:s}-NNstrct_{5:s}-MCpts_{6:d}-Nprss_{7:d}-NMCchains_{8:d}'.format(*data_params)
params_str=''



# load sample
n=-9
with open(load_dir + 'debug_files/' + 'debug-' + 'Eloc_data' + params_str + '.pkl', 'rb') as handle:
	Eloc_real, Eloc_imag = pickle.load(handle)


Eloc=(Eloc_real+1j*Eloc_imag)[n,:].ravel()

N_bootstrap=1000
N_batch=1000
Eloc_mean_s, Eloc_std_s = bootstrap_sample(Eloc, N_bootstrap, N_batch )


Eloc_real_min, Eloc_real_max = np.min(Eloc_mean_s.real), np.max(Eloc_mean_s.real)
Eloc_imag_min, Eloc_imag_max = np.min(Eloc_mean_s.imag), np.max(Eloc_mean_s.imag)
Eloc_imag_abs_min, Eloc_imag_abs_max = np.min(np.abs(Eloc_mean_s.imag)), np.max(np.abs(Eloc_mean_s.imag))

Eloc_std_min, Eloc_std_max = np.min(Eloc_std_s), np.max(Eloc_std_s)


print(Eloc_real_min,Eloc_real_max,  )
print(Eloc_imag_min,Eloc_imag_max)
print(Eloc_imag_abs_min, Eloc_imag_abs_max)
print(Eloc.mean())
print(Eloc_mean_s.mean())
print()
print(Eloc_std_min, Eloc_std_max, )
print(np.abs(Eloc).std())
print(Eloc_std_s.mean())










