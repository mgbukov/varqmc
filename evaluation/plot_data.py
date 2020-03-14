import sys,os
import numpy as np 
import pickle
import yaml 
#import jax

path = "../."
sys.path.insert(0,path)
from plot_lib import *

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

save=True #False # 



L=4
J2=0.5
opt='NG'
mode='MC'
NN_dtype='real-decoupled'
NN_shape_str='(16--12,16--24--12)'
N_MC_points=100000
N_prss=260
NMCchains=1
sys_time='2020_03_12-21_09_08' # '2020_03_12-21_08_52' # '2020_03_12-21_08_27' # '2020_03_12-21_08_03' #'2020_03_11-16_22_19'  # '2020_03_11-16_17_31'




#### load debug data


data_name = sys_time + '--{0:s}-L_{1:d}-{2:s}/'.format(opt,L,mode)
load_dir='data/' + data_name  + 'data_files/'
data_params=(NN_dtype,mode,L,J2,opt,NN_shape_str,N_MC_points,N_prss,NMCchains,)
params_str='model_DNN{0:s}-mode_{1:s}-L_{2:d}-J2_{3:0.1f}-opt_{4:s}-NNstrct_{5:s}-MCpts_{6:d}-Nprss_{7:d}-NMCchains_{8:d}'.format(*data_params)


plotfile_dir = 'data/' + data_name  + 'plots/'

if not os.path.exists(plotfile_dir):
	os.makedirs(plotfile_dir)




################

plot_acc_ratio(load_dir, plotfile_dir, params_str,L,J2, save=save)

plot_hist(load_dir, plotfile_dir, params_str,L,J2, save=save)

plot_NG( load_dir, plotfile_dir, params_str, L, J2, save=save)

plot_energy( load_dir, plotfile_dir, params_str, L, J2, save=save)

plot_loss( load_dir, plotfile_dir, params_str, L, J2, save=save)





