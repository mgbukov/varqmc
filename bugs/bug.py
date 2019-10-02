import sys,os

os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel
os.environ['OMP_NUM_THREADS']='1' # set number of MKL threads to run in parallel

#quspin_path = os.path.join(os.path.expanduser('~'),"quspin/QuSpin/")
#quspin_path = os.path.join(os.path.expanduser('~'),"quspin/QuSpin_dev/")
#sys.path.insert(0,quspin_path)

from quspin.operators import hamiltonian

import numpy as np

L=4

###### define model parameters ######
Lx=L
Ly=Lx # linear dimension of spin 1 2d lattice
N_sites = Lx*Ly # number of sites for spin 1
#
###### setting up user-defined symmetry transformations for 2d lattice ######
sites = np.arange(N_sites,dtype=np.int32) # sites [0,1,2,....]

x = sites%Lx # x positions for sites
y = sites//Lx # y positions for sites

T_x = (x+1)%Lx + Lx*y # translation along x-direction
T_y = x +Lx*((y+1)%Ly) # translation along y-direction



###### setting up hamiltonian ######
J1=1.0 # spin=spin interaction
sign=-1.0





###### setting up Hamiltonian site-coupling lists
J1_pm_list=[[sign*0.5*J1,i,T_x[i]] for i in range(N_sites)] + [[sign*0.5*J1,i,T_y[i]] for i in range(N_sites)]

H=hamiltonian([ ["+-",J1_pm_list],["-+",J1_pm_list] ], [], N=16) #
