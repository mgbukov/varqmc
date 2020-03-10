import os, sys
import numpy as np
import pickle
import csv

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



def plot_loss(load_dir, params_str,L,J2, save=True):

	file_name= load_dir + 'loss--' + params_str + '.txt'


	# preallocate lists
	iter_step=[]
	r2=[] 
	S_norm=[] 
	F_norm=[] 
	F_log_norm=[] 
	F_phase_norm=[] 
	S_logcond=[] 
	F_max=[]
	alpha_max=[]


	with open(file_name, 'r') as f:
		for j,row in enumerate(f):
			row_list=row.strip().split(" : ")
			
			iter_step.append(float(row_list[0]))
			r2.append(float(row_list[1])) 
			S_norm.append(float(row_list[2])) 
			F_norm.append(float(row_list[3]))
			F_log_norm.append(float(row_list[4])) 
			F_phase_norm.append(float(row_list[5]))
			S_logcond.append(float(row_list[6]))
			F_max.append(float(row_list[6]))
			alpha_max.append(float(row_list[7]))


	iter_step=np.array(iter_step)
	r2=np.array(r2)
	S_norm=np.array(S_norm)
	F_norm=np.array(F_norm)
	F_log_norm=np.array(F_log_norm)
	F_phase_norm=np.array(F_phase_norm)
	S_logcond=np.array(S_logcond)
	alpha_max=np.array(alpha_max)



	plt.plot(iter_step, r2, '.')
	plt.xlabel('iterations')
	plt.ylabel('$r^2$')
	plt.ylim(-0.01,1.01)
	#plt.legend()
	
	plt.grid()
	plt.tight_layout()
	plt.show()



def plot_energy(load_dir, params_str,L,J2, save=True):

	file_name= load_dir + 'energy--' + params_str + '.txt'


	if J2==0.0:
		E_GS_dict={"L=4":-11.228483, "L=6":-24.43939}
	elif J2==0.5:
		E_GS_dict={"L=4":-8.45792,   "L=6":-18.13716}



	# preallocate lists
	iter_step=[]
	Eave_real=[]
	Eave_imag=[]
	Estd=[]

	with open(file_name, 'r') as f:
		for j,row in enumerate(f):
			row_list=row.strip().split(" : ")

			iter_step.append(int(row_list[0]))
			Eave_real.append(float(row_list[1]))
			Eave_imag.append(float(row_list[2]))
			Estd.append(float(row_list[3]))


	iter_step=np.array(iter_step)
	Eave_real=np.array(Eave_real)
	Eave_imag=np.array(Eave_imag)
	Estd=np.array(Estd)


	E_GS=E_GS_dict['L={0:d}'.format(L)]
	Delta_E=np.abs((E_GS-Eave_real)/L**2)

	plt.plot(iter_step, Delta_E )
	plt.fill_between(iter_step, Delta_E-Estd, Delta_E+Estd, alpha=0.2)


	plt.xlabel('iterations')
	plt.ylabel('$|E - E_\\mathrm{GS}|/N$')

	plt.yscale('log')

	#plt.legend()
	plt.grid()
	plt.tight_layout()

	plt.show()



