import sys,os

os.environ['KMP_DUPLICATE_LIB_OK']='True' # uncomment this line if omp error occurs on OSX for python 3
os.environ['OMP_NUM_THREADS']='1' # set number of OpenMP threads to run in parallel
os.environ['MKL_NUM_THREADS']='1' # set number of MKL threads to run in parallel

quspin_path = os.path.join(os.path.expanduser('~'),"quspin/QuSpin_dev/")
sys.path.insert(0,quspin_path)

from quspin.basis import spin_basis_general
from quspin.operators import hamiltonian
import numpy as np


L=4
J1=1.0
J2=0.5
sign=-1.0

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

T_a = (x+1)%Lx + Lx*((y+1)%Ly) # translation along anti-diagonal
T_d = (x-1)%Lx + Lx*((y+1)%Ly) # translation along diagonal


###### setting up Hamiltonian site-coupling lists
J1_list=[[J1,i,T_x[i]] for i in range(N_sites)] + [[J1,i,T_y[i]] for i in range(N_sites)]
J1_pm_list=[[sign*0.5*J1,i,T_x[i]] for i in range(N_sites)] + [[sign*0.5*J1,i,T_y[i]] for i in range(N_sites)]
J2_list=[[J2,i,T_d[i]] for i in range(N_sites)] + [[J2,i,T_a[i]] for i in range(N_sites)]
J2_pm_list=[[0.5*J2,i,T_d[i]] for i in range(N_sites)] + [[0.5*J2,i,T_a[i]] for i in range(N_sites)]
#

static=[ ["+-",J1_pm_list],["-+",J1_pm_list], ["+-",J2_pm_list],["-+",J2_pm_list], ["zz",J1_list],["zz",J2_list] ]


basis = spin_basis_general(N_sites, pauli=False, Nup=N_sites//2)
no_checks=dict(check_herm=False,check_symm=False,check_pcon=False)
H=hamiltonian(static, [], basis=basis,dtype=np.float64,**no_checks) 


E, psi = H.eigsh(k=1,which='SA')
print(E[0], H.expt_value(psi))


#######
i=0
SS_value=np.zeros(N_sites)
for j in range(N_sites):
	Sz0_Sz1=hamiltonian([['zz',[[1.0,i,j]]]], [], basis=basis,dtype=np.float64,**no_checks) 
	SS_value[j]=Sz0_Sz1.expt_value(psi).real


SS_value_uq, index, inv_index, count=np.unique(np.round(SS_value,13), return_index=True, return_inverse=True, return_counts=True)

#print(SS_value_uq)
#print(index)


#######
unique_sites=[1,] #[1,2,5,6,10]




J1_list_unique=[[J1,0,1]]
J1_pm_list_unique=[[sign*0.5*J1,0,1]]
J1_unique_static=[['zz',J1_list_unique],['+-',J1_pm_list_unique],['-+',J1_pm_list_unique]]
HJ1=hamiltonian(J1_unique_static, [], basis=basis,dtype=np.float64,**no_checks) 

J2_list_unique=[[J2,0,7]]
J2_pm_list_unique=[[0.5*J2,0,7]]
J2_unique_static=[['zz',J2_list_unique],['+-',J2_pm_list_unique],['-+',J2_pm_list_unique]]
HJ2=hamiltonian(J2_unique_static, [], basis=basis,dtype=np.float64,**no_checks)

HJ_unique=hamiltonian(J1_unique_static+J2_unique_static, [], basis=basis,dtype=np.float64,**no_checks)

print(len(J1_list), len(J2_list))

print(2*N_sites*(HJ1.expt_value(psi) + HJ2.expt_value(psi))   )
print(2*N_sites*HJ_unique.expt_value(psi))
