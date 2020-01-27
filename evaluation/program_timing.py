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

#import PyQt5

#matplotlib.use('Qt5Agg')

#os.environ["PATH"] += ':/usr/local/texlive/2015/bin/x86_64-darwin'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=18)
plt.tick_params(labelsize=20)


def find_between( s, first, last ):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""

def find_between_r( s, first, last ):
    try:
        start = s.rindex( first ) + len( first )
        end = s.rindex( last, start )
        return s[start:end]
    except ValueError:
        return ""


##################################

mode='NG'
date='2020-01-24' 

directory='../simulation/data/' + date + '_' + mode +'/'


N_prss=[]
M_prss_ave=[]
j=0
#for filename in os.listdir(directory):	
#	if filename.endswith(".txt") and filename.startswith("simulation_time"): 

for k in range(26):

		filename="simulation_time--model_DNNcpx-mode_MC-L_6-J2_0.5-opt_NG-NNstrct_36--6-MCpts_4000-Nprss_{0:d}-NMCchains_1.txt".format(k+1)
		
		M = np.loadtxt(directory+filename, delimiter=',')
		
		if M.ndim>1:
			M_prss_ave.append(np.mean(M,axis=1))
			N_prss.append(M.shape[1])
		else:
			M_prss_ave.append(M.copy())
			N_prss.append(1)

		j+=1

		simulation_data = find_between(filename,'time--','-MCpts')


M_prss_ave=np.array(M_prss_ave)
N_prss=np.array(N_prss)



M_prss_iter_ave=np.zeros((M_prss_ave.shape[0],3))
M_prss_iter_ave[:,0]=M_prss_ave[:,0]
M_prss_iter_ave[:,1]=np.mean(M_prss_ave[:,1:-1],axis=1)
M_prss_iter_ave[:,2]=M_prss_ave[:,-1]

#
#print(M_prss_iter_ave)
#exit()

# 1 core time / N core time vs N
plt.plot(N_prss, M_prss_iter_ave[0,0]/M_prss_iter_ave[:,0],'sr', label='jit iteration')
plt.plot(N_prss, M_prss_iter_ave[0,1]/M_prss_iter_ave[:,1],'ob', label='ave iteration')
plt.plot(N_prss, M_prss_iter_ave[0,2]/M_prss_iter_ave[:,2],'dg', label='total time')
plt.legend()
plt.grid()

plt.ylabel('1-prss time / N-prss time')
plt.xlabel('$N_\\mathrm{MPI}$')
plt.title('$'+date + ',\\ ' + mode+'$')

plt.tight_layout()


plt.show()

plt.savefig('timing_'+date + '_' + simulation_data+'.pdf')



