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




####################################


def plot_acc_ratio(load_dir, plotfile_dir, params_str,L,J2, save=True):

	file_name= load_dir + 'MC_data--' + params_str + '.txt'


	# preallocate lists
	iter_step=[]
	MC_acc_ratio=[]


	with open(file_name, 'r') as f:
		for j,row in enumerate(f):
			row_list=row.strip().split(" : ")

			iter_step.append(int(row_list[0]))
			MC_acc_ratio.append(float(row_list[1]))
		


	iter_step=np.array(iter_step)
	MC_acc_ratio=np.array(MC_acc_ratio)




	plt.plot(iter_step, MC_acc_ratio, '.m', markersize=2.0 )

	plt.xlabel('iteration')
	plt.ylabel('MC acceptance ratio')

	#plt.yscale('log')

	#plt.legend()
	plt.grid()
	plt.tight_layout()

	if save:
		plt.savefig(plotfile_dir + 'MC_acc_ratio.pdf')
		plt.close()
	else:
		plt.show()






def plot_hist(load_dir, plotfile_dir, params_str,L,J2, save=True):

	file_name= load_dir + 'phases_histogram--' + params_str + '.txt'


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




	fig, ax = plt.subplots(nrows=1, ncols=1)

	im3 = ax.pcolor(iter_step,binned_phases, hist_vals, cmap='cool', norm=colors.LogNorm(vmin=1E-0, vmax=1E-4),)
	#ax3[0].set_ylabel('phase distribution')
	ax.set_xlabel('training step')
	ax.set_xlim([0,N_iter])
	ax.set_ylim([-np.pi,np.pi])
	#ax3.grid()
	#ax.legend(handlelength=0,loc='lower left')

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

	plt.tight_layout(rect=(0,0,0.9,1))

	print('here')

	if save:
		plt.savefig(plotfile_dir + 'phase_hist.pdf')
		plt.close()
	else:
		plt.show()


def phase_movie(load_dir, plotfile_dir, params_str,L,J2, clear_data=True):

	file_name= load_dir + 'phases_histogram--' + params_str + '.txt'


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

	n_bins=40
	binned_phases=np.linspace(-np.pi,np.pi, n_bins, endpoint=True)



	#####################

	# energy data

	file_name= load_dir + 'energy--' + params_str + '.txt'

	# preallocate lists
	Eave_real=[]
	Eave_imag=[]
	Estd=[]

	with open(file_name, 'r') as f:
		for j,row in enumerate(f):
			row_list=row.strip().split(" : ")

			Eave_real.append(float(row_list[1]))
			Eave_imag.append(float(row_list[2]))
			Estd.append(float(row_list[3]))


	iter_step=np.array(iter_step)
	Eave_real=np.array(Eave_real)
	Eave_imag=np.array(Eave_imag)
	Estd=np.array(Estd)


	#####################


	#create temporary save directory
	save_dir = plotfile_dir + '/tmp_movie'
	if not os.path.exists(save_dir):
	    os.makedirs(save_dir)

	for it, hist in enumerate(hist_vals.T):

		ax=plt.gca()

		ax.set_xlim([-np.pi,np.pi])
		ax.set_ylim([-0.05,1.05])

		ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
		ax.xaxis.set_major_formatter(plt.FuncFormatter(format_func))

		ax.grid(color='k', linestyle='-', linewidth=0.1)
		ax.yaxis.set_ticks_position('both')
		
		ax.set_xlabel('wavefunction phase')
		ax.set_ylabel('probability')



		ax.plot(binned_phases, hist, '-b.')
		ax.set_title('iter={0:04d}, $E={1:0.6f}$'.format(it,Eave_real[it]))


		plt.tight_layout()
		#plt.show()
		#exit()

		fname=save_dir + '/phase_hist_frame_{0:d}.png'.format(it)
		plt.savefig(fname)

		plt.close()


	# create movie
	movie_name=plotfile_dir + 'phase_movie.mp4' # -loglevel panic -codec:v libx264
	cmd = "ffmpeg -framerate 15 -i " + save_dir + "/phase_hist_frame_%01d.png -r 30 -pix_fmt yuv420p "+movie_name
	# execute command cmd
	os.system(cmd)
	# remove temp directory
	os.system("rm -rf "+plotfile_dir+"/tmp_movie*")






def plot_NG(load_dir, plotfile_dir, params_str,L,J2, save=True):

	file_name= load_dir + 'opt_data--' + params_str + '.txt'


	# preallocate lists
	iter_step=[]
	NG_counter=[]
	NG_RK_step_size=[]
	NG_RK_time=[]
	NG_delta=[]
	NG_tol=[]

	with open(file_name, 'r') as f:
		for j,row in enumerate(f):
			row_list=row.strip().split(" : ")

			iter_step.append(int(row_list[0]))
			NG_counter.append(float(row_list[1]))
			NG_RK_step_size.append(float(row_list[2]))
			NG_RK_time.append(float(row_list[3]))
			NG_delta.append(float(row_list[4]))
			NG_tol.append(float(row_list[5]))


	iter_step=np.array(iter_step)
	NG_counter=np.array(NG_counter)
	NG_RK_step_size=np.array(NG_RK_step_size)
	NG_RK_time=np.array(NG_RK_time)
	NG_delta=np.array(NG_delta)
	NG_tol=np.array(NG_tol)



	plt.plot(iter_step, NG_delta )

	plt.xlabel('iteration')
	plt.ylabel('$\\delta_\\mathrm{SR}$')

	plt.yscale('log')

	#plt.legend()
	plt.grid()
	plt.tight_layout()

	if save:
		plt.savefig(plotfile_dir + 'delta_SR.pdf')
		plt.close()
	else:
		plt.show()









def plot_loss(load_dir, plotfile_dir, params_str,L,J2, save=True):

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
			F_max.append(float(row_list[7]))
			alpha_max.append(float(row_list[8]))


	iter_step=np.array(iter_step)
	r2=np.array(r2)
	S_norm=np.array(S_norm)
	F_norm=np.array(F_norm)
	F_log_norm=np.array(F_log_norm)
	F_phase_norm=np.array(F_phase_norm)
	S_logcond=np.array(S_logcond)
	F_max=np.array(F_max)
	alpha_max=np.array(alpha_max)


	### plot r2

	plt.plot(iter_step, r2, '.r')
	plt.xlabel('iteration')
	plt.ylabel('$r^2$')
	plt.ylim(-0.01,1.01)
	#plt.legend()
	
	plt.grid()
	plt.tight_layout()


	if save:
		plt.savefig(plotfile_dir + 'r2.pdf')
		plt.close()
	else:
		plt.show()


	### plot alpha

	plt.plot(iter_step, alpha_max,'.b' )
	plt.xlabel('iteration')
	plt.ylabel('$\\mathrm{max}_k(\\alpha_k)$')
	#plt.ylim(-0.01,1.01)
	#plt.legend()
	
	plt.grid()
	plt.tight_layout()


	if save:
		plt.savefig(plotfile_dir + 'alpha_max.pdf')
		plt.close()
	else:
		plt.show()


	### plot S norm

	plt.plot(iter_step, S_norm, '.r', )
	plt.xlabel('iteration')
	plt.ylabel('$||S||$')
	#plt.ylim(-0.01,1.01)
	#plt.legend()
	
	plt.grid()
	plt.tight_layout()


	if save:
		plt.savefig(plotfile_dir + 'S_norm.pdf')
		plt.close()
	else:
		plt.show()


	### plot S cont

	plt.plot(iter_step, np.exp(S_logcond), '.b', )
	plt.xlabel('iteration')
	plt.ylabel('$\\mathrm{cond}(S)$')
	#plt.ylim(-0.01,1.01)
	#plt.legend()
	
	plt.grid()
	plt.tight_layout()


	if save:
		plt.savefig(plotfile_dir + 'S_cond.pdf')
		plt.close()
	else:
		plt.show()


	### plot F vector

	plt.plot(iter_step, F_log_norm, '.b', label='$||F^\\mathrm{log}||$')
	plt.plot(iter_step, F_phase_norm, '.g', label='$||F^{\\phi}||$')
	plt.plot(iter_step, F_norm, '.r', label='$||F||$', markersize=2.0)
	plt.xlabel('iteration')
	#plt.ylabel('')
	#plt.ylim(-0.01,1.01)
	plt.legend()
	
	plt.grid()
	plt.tight_layout()


	if save:
		plt.savefig(plotfile_dir + 'F_vector.pdf')
		plt.close()
	else:
		plt.show()


	### plot max_F

	plt.plot(iter_step, F_max, '.c', label='$||\\mathrm{max}_k(F_k)$')
	plt.xlabel('iteration')
	plt.ylabel('$\\mathrm{max}_k(F_k)$')
	#plt.ylim(-0.01,1.01)
	#plt.legend()
	
	plt.grid()
	plt.tight_layout()


	if save:
		plt.savefig(plotfile_dir + 'F_max.pdf')
		plt.close()
	else:
		plt.show()




def plot_energy(load_dir, plotfile_dir, params_str,L,J2, save=True):

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

	plt.plot(iter_step, Delta_E,'b' )
	plt.fill_between(iter_step, Delta_E-Estd, Delta_E+Estd,color='b', alpha=0.2)


	plt.xlabel('iteration')
	plt.ylabel('$|E - E_\\mathrm{GS}|/N$')

	plt.yscale('log')

	#plt.legend()
	plt.grid()
	plt.tight_layout()

	if save:
		plt.savefig(plotfile_dir + 'energy.pdf')
		plt.close()
	else:
		plt.show()



