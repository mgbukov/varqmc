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


L=6
J2=0.5
opt='NG' #'NG'
NN_struct=(36,6)
N_MC_points=20000


if J2==0.0:
	E_GS_dict={"L=4":-11.228483, "L=6":-24.43939}
elif J2==0.5:
	E_GS_dict={"L=4":-8.45792,   "L=6":-18.13716}

E_GS=E_GS_dict['L={0:d}'.format(L)]



data_tuple=(L,J2,opt,NN_struct[0],NN_struct[1],N_MC_points)
data_str='model_DNNcpx-mode_MC-L_{0:d}-J2_{1:0.1f}-opt_{2:s}-NNstrct_{3:d}--{4:d}-MCpts_{5:d}.txt'.format(*data_tuple)
path='./data/data_files/'


##########################

file_name = path + 'energy--'+data_str


# preallocate lists
iter_step=[]
Eave_real=[]
Eave_imag=[]

with open(file_name, 'r') as f:
	for j,row in enumerate(f):
		row_list=row.strip().split(" : ")

		iter_step.append(int(row_list[0]))
		Eave_real.append(float(row_list[1]))
		Eave_imag.append(float(row_list[2]))


iter_step=np.array(iter_step)
Eave_real=np.array(Eave_real)
Eave_imag=np.array(Eave_imag)


##########################


file_name = path + 'energy_std--'+data_str

# preallocate lists
E_std=[]

with open(file_name, 'r') as f:
	for j,row in enumerate(f):
		row_list=row.strip().split(" : ")
		E_std.append(float(row_list[1]))


E_std=np.array(E_std)


###########################




file_name = path + 'r2--'+data_str


# preallocate lists
r2=[]

with open(file_name, 'r') as f:
	for j,row in enumerate(f):
		row_list=row.strip().split(" : ")
		r2.append(float(row_list[1]))


r2=np.array(r2)


###########################


print('finished loading the data.\n')







plt.plot(iter_step, np.abs((E_GS-Eave_real)/L**2), label=opt + ', $L={0:d}, N_\\mathrm{{MC}}={1:d}$'.format(L,N_MC_points) )
plt.xlabel('iterations')
plt.ylabel('$|E - E_\\mathrm{GS}|/N$')
plt.yscale('log')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


plt.plot(iter_step, E_std, label=opt + ', $L={0:d}, N_\\mathrm{{MC}}={1:d}$'.format(L,N_MC_points) )
plt.xlabel('iterations')
plt.ylabel('$\\sigma(E)$')
plt.ylim(0,np.max(E_std))
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()



plt.plot(iter_step, r2, label=opt + ', $L={0:d}, N_\\mathrm{{MC}}={1:d}$'.format(L,N_MC_points) )
plt.xlabel('iterations')
plt.ylabel('$r^2$')
plt.ylim(-0.01,1.01)
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

