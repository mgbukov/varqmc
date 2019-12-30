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




data_tuple=(L,J2,opt,NN_struct[0],NN_struct[1],N_MC_points)
data_str='model_DNNcpx-mode_MC-L_{0:d}-J2_{1:0.1f}-opt_{2:s}-NNstrct_{3:d}--{4:d}-MCpts_{5:d}.txt'.format(*data_tuple)
path='./data/data_files/'


##########################

file_name = path + 'phases_histogram--'+data_str


# preallocate lists
iter_step=[]
hist_vals=[]

with open(file_name, 'r') as f:
	for j,row in enumerate(f):
		row_list=row.strip().split(" : ")

		iter_step.append(int(row_list[0]))
		hist_vals.append(np.array( list([float(elem.strip(',')) for elem in row_list[1].strip().split(", ")]) ))

iter_step=np.array(iter_step)
hist_vals=np.array(hist_vals).T


#####################################
N_iter=iter_step.shape[0]
n_bins=40
binned_phases=np.linspace(-np.pi,np.pi, n_bins, endpoint=True)



def format_func(value, tick_number):
	# find number of multiples of pi/2
	N = int(np.round(2 * value / np.pi))
	if N == 0:
		return "0"
	elif N == 1:
		return r"$\pi/2$"
	elif N == -1:
		return r"$-\pi/2$"
	elif N == 2:
		return r"$\pi$"
	elif N == -2:
		return r"$-\pi$"
	elif N % 2 > 0:
		return r"${0}\pi/2$".format(N)
	else:
		return r"${0}\pi$".format(N // 2)





fig, ax = plt.subplots(nrows=1, ncols=1)

im3 = ax.pcolor(iter_step,binned_phases, hist_vals, cmap='cool', norm=colors.LogNorm(vmin=1E-0, vmax=1E-4),label='phase distr.')
#ax3[0].set_ylabel('phase distribution')
ax.set_xlabel('training step')
ax.set_xlim([0,N_iter])
ax.set_ylim([-np.pi,np.pi])
#ax3.grid()
ax.legend(handlelength=0,loc='lower left')

cbar_ax = inset_axes(ax,
                width="2.5%",  # width = 50% of parent_bbox width
                height="100%",  # height : 5%
                loc='lower left',
                bbox_to_anchor=(1.025, 0., 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0,

                
                )

cbar_ax.set_yscale('log')

fig.colorbar(im3,cax=cbar_ax,ticks=[1E-4,1E-3,1E-2,1E-1,1E0])#

ax.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))


ax.grid(color='k', linestyle='-', linewidth=0.1)
ax.yaxis.set_ticks_position('both')


#fig.delaxes(ax)

plt.tight_layout(rect=(0,0,0.99,1))

plt.show()

