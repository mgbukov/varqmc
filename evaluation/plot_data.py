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



iterations=[200,400,600,800,1000]
J2=0.5
L=6
opt='RK_RK' # 'sgd_sgd' # 'sgd_sgd' #  
cost='SR_SR' # 'SR_SR' #
mode='MC' # 'exact' #

#sys_time = '2020_06_13-09_50_43' # '2020_06_08-22_23_05' #  '2020_06_05-21_55_27', '2020_06_08-22_22_13', '2020_06_08-22_22_42'    

for sys_time in ['2020_06_23-11_45_00']: #['2020_06_13-09_51_06', '2020_06_13-09_51_20', '2020_06_13-09_51_33', ]:

	#### load debug data

	data_name = sys_time + '--{0:s}-{1:s}-L_{2:d}-{3:s}/'.format(opt,cost,L,mode)
	#load_dir='data/' + data_name 
	#load_dir='data/paper_data/seeds/' + data_name 
	#load_dir='data/paper_data/MC_samples/' + data_name 
	load_dir='data/paper_data/local_sampling/' + data_name 

	#data_params=(NN_dtype,mode,L,J2,opt,NN_shape_str,N_MC_points,N_prss,NMCchains,)
	#params_str='--model_DNN{0:s}-mode_{1:s}-L_{2:d}-J2_{3:0.1f}-opt_{4:s}-NNstrct_{5:s}-MCpts_{6:d}-Nprss_{7:d}-NMCchains_{8:d}'.format(*data_params)
	params_str=''

	plotfile_dir = load_dir  + 'plots/'

	if os.path.exists(load_dir) and (not os.path.exists(plotfile_dir)):
		os.makedirs(plotfile_dir)




	################



	plot_sample(load_dir, plotfile_dir, params_str,L,J2, iterations, N_MC_points=1000, save=save)



	load_dir+= 'data_files/'

	plot_delta(load_dir, plotfile_dir, params_str, L, J2, save=save)

	if 'MC' in mode:
		plot_acc_ratio(load_dir, plotfile_dir, params_str,L,J2, save=save)

	plot_hist(load_dir, plotfile_dir, params_str,L,J2, save=save)

	plot_energy( load_dir, plotfile_dir, params_str, L, J2, save=save)

	plot_loss( load_dir, plotfile_dir, params_str, L, J2, save=save)

	if 'SR' in cost:
		plot_S_eigvals(load_dir, plotfile_dir, params_str, save=True)

		plot_overlap_VF(load_dir, plotfile_dir, params_str, save=True)

		plot_SNR(load_dir, plotfile_dir, params_str, iterations, save=True)


	#phase_movie(load_dir, plotfile_dir, params_str,L,J2, clear_data=True)



