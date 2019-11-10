import os, sys
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





class data():

	def __init__(self,cwd,model_params,N_MC_points,N_epochs,extra_label=''):

		self.model_params=model_params
		self._create_name(extra_label)
		self._define_variables(N_MC_points,N_epochs)

		self.cwd=cwd
		self._create_directories()


	def _create_directories(self):	
		# create a data directory
		self.data_dir=self.cwd+'/data/checkpoints/'
		if not os.path.exists(self.data_dir):
		    os.makedirs(self.data_dir)

		# create a plots directory
		self.plots_dir=self.cwd+'/data/plots/'
		if not os.path.exists(self.plots_dir):
		    os.makedirs(self.plots_dir)



	def _define_variables(self,N_MC_points,N_epochs):
		self.N_epochs=N_epochs
		self.N_MC_points=N_MC_points

		self.excess_energy=np.zeros((N_epochs,2),dtype=np.complex128)
		self.SdotS=np.zeros((N_epochs,2),dtype=np.complex128)

		self.loss=np.zeros((N_epochs,2),dtype=np.float64)
		self.r2=np.zeros(N_epochs,dtype=np.float64)

		self.phase_psi=np.zeros((N_epochs,N_MC_points),dtype=np.float64)
		self.mod_psi=np.zeros((N_epochs,N_MC_points),dtype=np.float64)


	def _create_name(self,extra_label):
		file_name = ''
		for key,value in self.model_params.items():
			file_name += ( key+'_{}'.format(value)+'-' )
		file_name=file_name[:-1]
		self.file_name=file_name+extra_label

	
	def save(self,NN_params=None):
		with open(self.data_dir+self.file_name+'.pkl', 'wb') as handle:
			pickle.dump([self.excess_energy,self.SdotS,self.loss,self.r2,self.phase_psi,self.mod_psi,self.file_name,], 
						handle, protocol=pickle.HIGHEST_PROTOCOL)

		if NN_params is not None:
			with open(self.data_dir+'NNparams--'+self.file_name+'.pkl', 'wb') as handle:
				pickle.dump([NN_params,], 
							handle, protocol=pickle.HIGHEST_PROTOCOL)

		print("data saved as:\n {}\n".format(self.file_name) )

	def load(self):
		handle=open(self.data_dir+self.file_name+'.pkl','rb')
		self.excess_energy,self.SdotS,self.loss,self.r2,self.phase_psi,self.mod_psi,_ = pickle.load(handle)
		print("data loaded from:\n {}\n".format(self.file_name)  )
		
	def load_weights(self):
		handle=open(self.data_dir+'NNparams--'+self.file_name+'.pkl','rb')
		NN_params = pickle.load(handle)[0]
		print("NN parameters loaded from:\n {}\n".format(self.file_name)  )
		return NN_params

	def compute_phase_hist(self,n_bins=40):

		self.phase_hist=np.zeros((n_bins,self.N_epochs))
		self.binned_phases=np.linspace(-np.pi,np.pi, n_bins, endpoint=True)

		for j,(phases,amplds) in enumerate(zip(self.phase_psi,self.mod_psi)):
			# shift phases
			phases = (phases+np.pi)%(2*np.pi) - np.pi
			hist, bin_edges = np.histogram(phases ,bins=n_bins,range=(-np.pi,np.pi), density=True, weights=amplds**2)
			self.phase_hist[:,j] = hist*np.diff(bin_edges)
				


	def plot(self,save=False):

		epochs=list(range(self.N_epochs))

		fig, (ax1, ax2, ax3) = plt.subplots(nrows=3,ncols=2,sharex=True)
		fig.set_size_inches(18,9)


		ax1[0].plot(epochs,self.loss[:,0]/np.max(self.loss[:,0]),'b',label='$\\sim \\mathrm{max}\\ |F_k|$' )
		ax1[0].plot(epochs,self.loss[:,1]/np.max(self.loss[:,1]),'r',label='$\\sim \\mathrm{max}\\ |\\dot\\alpha_k|$' )
		ax1[0].grid()
		#ax1.set_ylabel('pseudo loss')
		ax1[0].legend()
		ax1[0].set_xlim([0,self.N_epochs])
		#ax1.set_ylim([-0.05,0.05])
		ax1[0].yaxis.set_ticks_position('both')


		ax1[1].plot(epochs,self.r2,'r',label='$r^2_\\mathrm{SR}$' )
		ax1[1].grid()
		#ax1.set_ylabel('pseudo loss')
		ax1[1].legend()
		ax1[1].set_xlim([0,self.N_epochs])
		ax1[1].set_ylim([-0.01,1.01])
		#ax1[1].set_yscale('log')
		ax1[1].yaxis.set_ticks_position('both')



		ax2[0].plot(epochs, np.abs(self.excess_energy[:,0].real), label='$\\frac{1}{N}|E - E_\\mathrm{GS}|$' )
		error=self.excess_energy[:,1].real
		ax2[0].fill_between(epochs, np.abs(self.excess_energy[:,0].real)-error, np.abs(self.excess_energy[:,0].real)+error, alpha=0.2)
		ax2[0].grid(which='major')
		ax2[0].grid(which='minor',linewidth=0.01)
		ax2[0].legend()
		ax2[0].set_xlim([0,self.N_epochs])
		ax2[0].set_yscale('log')
		#ax2[0].set_ylim([1E-3,1E-0])
		ax2[0].yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, ), numdecs=4, numticks=10))
		ax2[0].yaxis.set_minor_locator(LogLocator(base=10.0, subs=(range(1,10,1) ), numdecs=4, numticks=10))
		ax2[0].yaxis.set_ticks_position('both')
		#ax2[0].set_ylabel('$|E_\\mathrm{var} - E_\\mathrm{GS}|/N$')
	
	
		ax2[1].plot(epochs,self.SdotS[:,0].real,'g',label='$\\frac{1}{S_\\mathrm{max}(S_\\mathrm{max}+1)}\\sum_{i,j}\\langle\\vec{S}_i\\cdot\\vec{S}_j\\rangle$' )
		#ax2[1].plot(epochs,self.SdotS[:,0].real,'g',label='$\\sum_{i,j}\\langle\\vec{S}_i\\cdot\\vec{S}_j\\rangle$' )
		error=self.SdotS[:,1].real
		ax2[1].fill_between(epochs, self.SdotS[:,0].real-error, self.SdotS[:,0].real+error,color='g', alpha=0.2)
		ax2[1].grid(which='major')
		#ax2[1].grid(which='minor',linewidth=0.01)
		#ax2.set_ylabel('pseudo loss')
		ax2[1].legend()
		#ax2[1].yaxis.set_major_locator(LinearLocator(numticks=4))
		#ax2[1].set_yticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
		ax2[1].yaxis.set_ticks_position('both')
		ax2[1].xaxis.set_tick_params(labelbottom=True)
		ax2[1].set_xlim([0,self.N_epochs])
		ax2[1].set_ylim([-0.01,self.SdotS[:,0].real.max()+0.01])
		#ax2[1].set_ylim([-0.01,1.01])
		




		im3 = ax3[0].pcolor(epochs,self.binned_phases, self.phase_hist, cmap='cool', norm=colors.LogNorm(vmin=1E-0, vmax=1E-4),label='phase distr.')
		#ax3[0].set_ylabel('phase distribution')
		ax3[0].set_xlabel('training step')
		ax3[0].set_xlim([0,self.N_epochs])
		ax3[0].set_ylim([-np.pi,np.pi])
		#ax3.grid()
		ax3[0].legend(handlelength=0,loc='lower left')

		cbar_ax = inset_axes(ax3[0],
	                    width="2.5%",  # width = 50% of parent_bbox width
	                    height="100%",  # height : 5%
	                    loc='lower left',
	                    bbox_to_anchor=(1.025, 0., 1, 1),
	                    bbox_transform=ax3[0].transAxes,
	                    borderpad=0,

	                    
	                    )
		
		cbar_ax.set_yscale('log')
		
		fig.colorbar(im3,cax=cbar_ax,ticks=[1E-4,1E-3,1E-2,1E-1,1E0])#

		ax3[0].yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
		ax3[0].yaxis.set_major_formatter(plt.FuncFormatter(format_func))

	
		ax3[0].grid(color='k', linestyle='-', linewidth=0.1)
		ax3[0].yaxis.set_ticks_position('both')


		fig.delaxes(ax3[1])
		
		plt.tight_layout(rect=(0,0,0.99,1))

		if save==False:
			plt.show()
		else:
			plt.savefig(self.plots_dir+self.file_name+'.pdf',format='pdf')





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


