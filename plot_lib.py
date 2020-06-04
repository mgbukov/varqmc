import os, sys
import numpy as np
import pickle
import csv

from eval_lib import *
from cpp_code import integer_to_spinstate

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



def _load_data(file_name,N_variables):

	# preallocate lists
	data=tuple([] for _ in range(N_variables))

	with open(file_name, 'r') as f:
		for j,row in enumerate(f):
			for k, var in enumerate(row.strip().split(":"), ):

				#data[k].append(np.float64(var))
				data[k].append(np.array([np.float64(v) for v in var.split(",") if v !='']))

	data_array=()
	for var in data:
		data_array+=(np.array(var).squeeze(), )

	return data_array


####################################


def _plot_SNR(plotfile_dir,it,SNR_exact,SNR_gauss,VF_overlap,S_eigvals,save,net_type='log',):

	plt.plot(SNR_exact[it,:],'b', label='SNR exact')
	plt.plot(SNR_gauss[it,:],'r', label='SNR gauss')
	plt.plot(np.ones_like(SNR_exact[it,:]),'-k', label='1.0' )
	plt.plot(np.abs(VF_overlap[it,:]), 'm', label='$V^\\dagger F$')
	plt.plot(S_eigvals[it,:], 'c', label='$\\sigma^2$', )
	
	plt.yscale('log')
	plt.xlabel('$k$')
	plt.title('iteration={0:d}'.format(it))


	plt.grid()

	plt.legend()
	

	plt.tight_layout()

	if net_type=='log':
		save_name=plotfile_dir + 'SNRs_log--iter_{0:05d}.pdf'.format(it)
	elif net_type=='phase':
		save_name=plotfile_dir + 'SNRs_phase--iter_{0:05d}.pdf'.format(it)


	if save:
		plt.savefig(save_name)
		plt.close()
	else:
		plt.show()


def plot_SNR(load_dir, plotfile_dir, params_str, iterations, save=True):

	N_variables=2

	file_name= load_dir + 'overlap_VF_log' + params_str + '.txt'
	iter_step, overlap_VF_log = _load_data(file_name,N_variables)

	file_name= load_dir + 'overlap_VF_phase' + params_str + '.txt'
	iter_step, overlap_VF_phase = _load_data(file_name,N_variables)



	N_variables=2

	file_name= load_dir + 'eigvals_S_matrix_log' + params_str + '.txt'
	iter_step, S_eigvals_log = _load_data(file_name,N_variables)

	file_name= load_dir + 'eigvals_S_matrix_phase' + params_str + '.txt'
	iter_step, S_eigvals_phase = _load_data(file_name,N_variables)



	N_variables=2

	file_name= load_dir + 'SNR_exact_log' + params_str + '.txt'
	iter_step, SNR_exact_log = _load_data(file_name,N_variables)

	file_name= load_dir + 'SNR_exact_phase' + params_str + '.txt'
	iter_step, SNR_exact_phase = _load_data(file_name,N_variables)

	file_name= load_dir + 'SNR_gauss_log' + params_str + '.txt'
	iter_step, SNR_gauss_log = _load_data(file_name,N_variables)

	file_name= load_dir + 'SNR_gauss_phase' + params_str + '.txt'
	iter_step, SNR_gauss_phase = _load_data(file_name,N_variables)


	for it in iterations:
		_plot_SNR(plotfile_dir,it,SNR_exact_log,SNR_gauss_log,overlap_VF_log,S_eigvals_log,save,net_type='log')
		_plot_SNR(plotfile_dir,it,SNR_exact_phase,SNR_gauss_phase,overlap_VF_phase,S_eigvals_phase,save,net_type='phase')





def plot_sample(load_dir, plotfile_dir, params_str,L,J2, iterations, N_MC_points=1000, save=True):

	for iteration in iterations:

		file_name= 'NNparams--iter_{0:05d}'.format(iteration) 


		with open(load_dir + 'NN_params/' +file_name+'.pkl', 'rb') as handle:
			params_log, params_phase, apply_fun_args_log, apply_fun_args_phase, log_psi_shift = pickle.load(handle)

		# sample points
		MC_tool = MC_sample(load_dir, params_log, N_MC_points=N_MC_points, reps=True)


		rep_spin_configs_ints=compute_reps(MC_tool.ints_ket,L)

		log_psi, phase_psi = evaluate_DNN(load_dir, params_log, params_phase, rep_spin_configs_ints, )
		sign_psi = np.exp(-1j*phase_psi)


		log_psi, phase_psi,  phase_psi_bras, log_psi_bras = evaluate_sample(load_dir,params_log, params_phase, rep_spin_configs_ints, log_psi,phase_psi,log_psi_shift=0.0)

		# wrap phases
		phase_psi = (phase_psi+np.pi)%(2*np.pi) - np.pi
		phase_psi_bras = (phase_psi_bras+np.pi)%(2*np.pi) - np.pi

		# shift phases
		ind=np.argmax(log_psi)
		a=phase_psi[ind]
		b=log_psi[ind]

		phase_psi=phase_psi-a
		phase_psi_bras=phase_psi_bras-a

		log_psi=log_psi-b
		log_psi_bras=log_psi_bras-b


		#print(phase_psi)

		############

		fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True)

		xlim=[-2*np.pi,2*np.pi]
		ylim=[-20.0,1.0]

		ax[0].plot(phase_psi,2.0*log_psi,'.b',markersize=0.5)
		ax[0].set_xlim(xlim)
		ax[0].set_ylim(ylim)
		ax[0].set_xlabel('$\\varphi_s$')
		ax[0].set_ylabel('$2\\log|\\psi_s|$')
		ax[0].set_title('$s$-configs')
		ax[0].xaxis.set_major_locator(plt.MultipleLocator(np.pi))
		ax[0].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
		ax[0].grid(color='k', linestyle='-', linewidth=0.1)
		ax[0].xaxis.set_ticks_position('both')
		ax[0].yaxis.set_ticks_position('both')

		ax[1].plot(phase_psi_bras,2.0*log_psi_bras,'.r',markersize=0.5)
		ax[1].set_xlim(xlim)
		ax[1].set_ylim(ylim)
		ax[1].set_xlabel('$\\varphi_s$')
		ax[1].set_title("$s'$-configs")
		ax[1].xaxis.set_major_locator(plt.MultipleLocator(np.pi))
		ax[1].xaxis.set_major_formatter(plt.FuncFormatter(format_func))
		ax[1].grid(color='k', linestyle='-', linewidth=0.1)
		ax[1].xaxis.set_ticks_position('both')
		ax[1].yaxis.set_ticks_position('both')

		plt.tight_layout()


		if save:
			plt.savefig(plotfile_dir + 'configs--iter_{0:05d}.png'.format(iteration))
			plt.close()
		else:
			plt.show()





def plot_acc_ratio(load_dir, plotfile_dir, params_str,L,J2, save=True):

	file_name= load_dir + 'MC_data' + params_str + '.txt'


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

	plt.ylim([0.0,1.0])

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

	file_name= load_dir + 'phases_histogram' + params_str + '.txt'


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

	if save:
		plt.savefig(plotfile_dir + 'phase_hist.pdf')
		plt.close()
	else:
		plt.show()



	######################################################

	# plot against eigentime

	N_variables=8
	file_name= load_dir + 'opt_data_phase' + params_str + '.txt'
	iter_step_phase, delta_phase, tol_phase, SNR_weight_sum_exact_phase	, SNR_weight_sum_gauss_phase, counter_phase, step_size_phase, time_phase = _load_data(file_name,N_variables)


	fig, ax = plt.subplots(nrows=1, ncols=1)

	im3 = ax.pcolor(time_phase,binned_phases, hist_vals, cmap='cool', norm=colors.LogNorm(vmin=1E-0, vmax=1E-4),)
	#ax3[0].set_ylabel('phase distribution')
	ax.set_xlabel('time')
	ax.set_xlim([0,time_phase.max()])
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

	if save:
		plt.savefig(plotfile_dir + 'phase_hist_eigentime.pdf')
		plt.close()
	else:
		plt.show()



def phase_movie(load_dir, plotfile_dir, params_str,L,J2, clear_data=True):

	file_name= load_dir + 'phases_histogram' + params_str + '.txt'


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

	file_name= load_dir + 'energy' + params_str + '.txt'
	iter_step, Eave_real, Eave_imag, Evar, Estd = _load_data(file_name,N_variables)



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








def plot_delta(load_dir, plotfile_dir, params_str,L,J2, save=True):

	
	N_variables=8

	file_name= load_dir + 'opt_data_log' + params_str + '.txt'
	iter_step_log, delta_log, tol_log, SNR_weight_sum_exact_log, SNR_weight_sum_gauss_log, counter_log, step_size_log, time_log = _load_data(file_name,N_variables)


	file_name= load_dir + 'opt_data_phase' + params_str + '.txt'
	iter_step_phase, delta_phase, tol_phase, SNR_weight_sum_exact_phase	, SNR_weight_sum_gauss_phase, counter_phase, step_size_phase, time_phase = _load_data(file_name,N_variables)


	plt.plot(iter_step_log, delta_log, 'r', label='log net' )
	plt.plot(iter_step_phase, delta_phase, 'b', label='phase net' )

	plt.xlabel('iteration')
	plt.ylabel('$\\delta_\\mathrm{SR}$')

	plt.yscale('log')

	plt.legend()
	plt.grid()
	plt.tight_layout()

	if save:
		plt.savefig(plotfile_dir + 'delta_SR.pdf')
		plt.close()
	else:
		plt.show()









def plot_loss(load_dir, plotfile_dir, params_str,L,J2, save=True):


	N_variables=8

	file_name= load_dir + 'loss_log' + params_str + '.txt'
	iter_step_log, r2_log, max_grad_log, dE_log, curv_log, F_norm_log, S_norm_log, S_logcond_log = _load_data(file_name,N_variables)

	file_name= load_dir + 'loss_phase' + params_str + '.txt'
	iter_step_phase, r2_phase, max_grad_phase, dE_phase, curv_phase, F_norm_phase, S_norm_phase, S_logcond_phase = _load_data(file_name,N_variables)


	### plot r2

	plt.plot(iter_step_log, r2_log, '.r', label='log net')
	plt.plot(iter_step_phase, r2_phase, '.b', label='phase net')
	plt.plot(iter_step_phase, r2_log + r2_phase - 1.0, '.g', label='$r^2_\\mathrm{log}\\!+\\!r^2_\\mathrm{phase}\\!-\\!1$')
	plt.xlabel('iteration')
	plt.ylabel('$r^2$')
	plt.ylim(-0.01,1.01)
	plt.legend()
	
	plt.grid()
	plt.tight_layout()


	if save:
		plt.savefig(plotfile_dir + 'r2.pdf')
		plt.close()
	else:
		plt.show()


	###############


	plt.plot(iter_step_phase, r2_log + r2_phase - 1.0, '.g', )
	plt.xlabel('iteration')
	plt.ylabel('$r^2_\\mathrm{log}\\!+\\!r^2_\\mathrm{phase}\\!-\\!1$')
	#plt.ylim(-0.01,1.01)
	plt.yscale('log')
	#plt.legend()
	
	plt.grid()
	plt.tight_layout()


	if save:
		plt.savefig(plotfile_dir + 'r2_total.pdf')
		plt.close()
	else:
		plt.show()


	### plot alpha

	plt.plot(iter_step_log, max_grad_log,'.r', label='log net' )
	#plt.plot(iter_step_phase, max_grad_phase,'.b', label='phase net' )
	plt.xlabel('iteration')
	plt.ylabel('$\\mathrm{max}_k|\\alpha_k|$')
	#plt.ylim(-0.01,1.01)
	plt.legend()
	
	plt.grid()
	plt.tight_layout()


	if save:
		plt.savefig(plotfile_dir + 'alpha_max_log.pdf')
		plt.close()
	else:
		plt.show()


	#plt.plot(iter_step_log, max_grad_log,'.r', label='log net' )
	plt.plot(iter_step_phase, max_grad_phase,'.b', label='phase net' )
	plt.xlabel('iteration')
	plt.ylabel('$\\mathrm{max}_k|\\alpha_k|$')
	#plt.ylim(-0.01,1.01)
	plt.legend()
	
	plt.grid()
	plt.tight_layout()


	if save:
		plt.savefig(plotfile_dir + 'alpha_max_phase.pdf')
		plt.close()
	else:
		plt.show()


	### plot S norm

	plt.plot(iter_step_log, S_norm_log, '.r', label='log net'  )
	plt.plot(iter_step_phase, S_norm_phase,'.b', label='phase net' )
	plt.xlabel('iteration')
	plt.ylabel('$||S||$')
	#plt.ylim(-0.01,1.01)
	plt.legend()
	
	plt.grid()
	plt.tight_layout()


	if save:
		plt.savefig(plotfile_dir + 'S_norm.pdf')
		plt.close()
	else:
		plt.show()


	### plot S cont

	plt.plot(iter_step_log, S_logcond_log, '.r', label='log net' )
	plt.plot(iter_step_phase, S_logcond_phase,'.b', label='phase net' )
	plt.xlabel('iteration')
	plt.ylabel('$\\log(\\mathrm{cond}(S))$')
	#plt.ylim(-0.01,1.01)
	plt.legend()
	
	plt.grid()
	plt.tight_layout()


	if save:
		plt.savefig(plotfile_dir + 'S_cond.pdf')
		plt.close()
	else:
		plt.show()


	### plot F vector

	plt.plot(iter_step_log, F_norm_log, '.r', label='log net')
	plt.plot(iter_step_phase, F_norm_phase, '.b', label='phase net')
	plt.xlabel('iteration')
	plt.ylabel('$||F_k||$')
	#plt.ylim(-0.01,1.01)
	plt.legend()
	
	plt.grid()
	plt.tight_layout()


	if save:
		plt.savefig(plotfile_dir + 'F_vector.pdf')
		plt.close()
	else:
		plt.show()



	### plot dE vector
	plt.plot(iter_step_log, curv_log, '.r', label='log', )
	plt.plot(iter_step_phase, curv_phase, '.b', label='phase net', )
	plt.xlabel('iteration')
	plt.ylabel('$\\sqrt{\\dot\\alpha^t S \\dot\\alpha}$')
	#plt.ylim(-0.01,1.01)
	plt.legend()
	
	plt.grid()
	plt.tight_layout()


	if save:
		plt.savefig(plotfile_dir + 'curvatures.pdf')
		plt.close()
	else:
		plt.show()






def plot_energy(load_dir, plotfile_dir, params_str,L,J2, save=True):

	if J2==0:
		E_GS_dict={'L=4':-11.228483 ,'L=6':-24.43939}
	else:
		E_GS_dict={'L=4':-8.45792 ,'L=6':-18.13716}

	N_variables=5

	file_name= load_dir + 'energy' + params_str + '.txt'
	iter_step, Eave_real, Eave_imag, Evar, Estd = _load_data(file_name,N_variables)


	E_GS=E_GS_dict['L={0:d}'.format(L)]
	Delta_E=np.abs((E_GS-Eave_real)/L**2)

	plt.plot(iter_step, Delta_E,'b' )
	plt.fill_between(iter_step, Delta_E-Estd, Delta_E+Estd,color='b', alpha=0.2)

	plt.gca().yaxis.set_ticks_position('both')

	plt.xlabel('iteration')
	plt.ylabel('$|E - E_\\mathrm{GS}|/N$')

	plt.yscale('log')

	plt.ylim([1E-3,2E0])

	#plt.legend()
	plt.grid()
	plt.tight_layout()

	if save:
		plt.savefig(plotfile_dir + 'energy.pdf')
		plt.close()
	else:
		plt.show()


def _plot_S_eigvals(iter_step, S_eigvals, tol, plotfile_dir, net_type, save):

	S_max=np.tile(np.max(np.abs(S_eigvals), axis=1) , [S_eigvals.shape[1],1]).T

	plt.plot(iter_step, np.abs(S_eigvals)/S_max, linewidth=0.5 )
	plt.plot(iter_step, tol, '-k', linewidth=1.5 )


	plt.gca().yaxis.set_ticks_position('both')

	plt.xlabel('iteration')
	plt.ylabel('$\\sigma^2_k$')

	plt.ylim(1E-18, 1E1)

	plt.yscale('log')

	if net_type=='log':
		plt.title("$\\mathrm{log\\ net}$")
	else:
		plt.title("$\\mathrm{phase\\ net}$")
	
	plt.tight_layout()

	if net_type=='log':
		name_str='S_eigvals_log'
	else:
		name_str='S_eigvals_phase'


	if save:
		plt.savefig(plotfile_dir + name_str + '.png')
		plt.close()
	else:
		plt.show()


def plot_S_eigvals(load_dir, plotfile_dir, params_str, save=True):

	
	N_variables=2

	file_name= load_dir + 'eigvals_S_matrix_log' + params_str + '.txt'
	iter_step, S_eigvals_log = _load_data(file_name,N_variables)

	file_name= load_dir + 'eigvals_S_matrix_phase' + params_str + '.txt'
	iter_step, S_eigvals_phase = _load_data(file_name,N_variables)



	N_variables=8

	file_name= load_dir + 'opt_data_log' + params_str + '.txt'
	iter_step_log, delta_log, tol_log, SNR_weight_sum_exact_log, SNR_weight_sum_gauss_log, counter_log, step_size_log, time_log = _load_data(file_name,N_variables)

	file_name= load_dir + 'opt_data_phase' + params_str + '.txt'
	iter_step_phase, delta_phase, tol_phase, SNR_weight_sum_exact_phase	, SNR_weight_sum_gauss_phase, counter_phase, step_size_phase, time_phase = _load_data(file_name,N_variables)


	_plot_S_eigvals(iter_step, S_eigvals_log, tol_log, plotfile_dir, net_type='log', save=save)
	_plot_S_eigvals(iter_step ,S_eigvals_phase, tol_phase, plotfile_dir ,net_type='phase', save=save)
	



def _plot_overlap_VF(data, plotfile_dir, net_type, save):


	fig, ax = plt.subplots(nrows=1, ncols=1)

	im3 = ax.pcolor(np.abs(data)+1E-15, cmap='hot', norm=colors.LogNorm(vmin=1E-0, vmax=1E-10),)
	
	
	cbar_ax = inset_axes(ax,
	                width="2.5%",  # width = 50% of parent_bbox width
	                height="100%",  # height : 5%
	                loc='lower left',
	                bbox_to_anchor=(1.025, 0., 1, 1),
	                bbox_transform=ax.transAxes,
	                borderpad=0,           
	                )

	cbar_ax.set_yscale('log')

	fig.colorbar(im3,cax=cbar_ax,ticks=[1E-9,1E-7,1E-5,1E-3,1E-1])#

	
	plt.gca().yaxis.set_ticks_position('both')

	if net_type=='log':
		ax.set_title("$\\mathrm{log\\ net}$")
	else:
		ax.set_title("$\\mathrm{phase\\ net}$")
	
	ax.set_xlabel('$k$')
	ax.set_ylabel('iteration')
	cbar_ax.set_xlabel('$|V^\\dagger F|$',fontsize=16)
	cbar_ax.xaxis.set_label_position('top') 

	
	plt.tight_layout(rect=(0,0,0.9,1))

	if net_type=='log':
		name_str='overlap_VF_log'
	else:
		name_str='overlap_VF_phase'

	if save:
		plt.savefig(plotfile_dir + name_str + '.png')
		plt.close()
	else:
		plt.show()


def plot_overlap_VF(load_dir, plotfile_dir, params_str, save=True):

	
	N_variables=2

	file_name= load_dir + 'overlap_VF_log' + params_str + '.txt'
	iter_step, overlap_VF_log = _load_data(file_name,N_variables)


	file_name= load_dir + 'overlap_VF_phase' + params_str + '.txt'
	iter_step, overlap_VF_phase = _load_data(file_name,N_variables)

	_plot_overlap_VF(overlap_VF_log, plotfile_dir, net_type='log', save=save)
	_plot_overlap_VF(overlap_VF_phase, plotfile_dir ,net_type='phase', save=save)
	




