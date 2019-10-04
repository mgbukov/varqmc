import numpy as np
from mpi4py import MPI
import time

seed=1
np.random.seed(seed)


# mpiexec -n 2 python main.py 


#########################################

N=4000000


x=np.random.uniform(size=N)
y=np.random.uniform(size=N)
z=np.zeros_like(x)


comm=MPI.COMM_WORLD
comm_size=comm.Get_size()
rank=comm.Get_rank()

n=N//comm_size

######
ti = time.time()


# local variables
x_loc=x[rank*n : (rank+1)*n]
y_loc=y[rank*n : (rank+1)*n]

# x_loc=np.zeros(n)
# y_loc=np.zeros(n)
z_loc=np.zeros_like(x_loc)


# comm.Scatter([x,  MPI.DOUBLE], [x_loc, MPI.DOUBLE])
# comm.Scatter([y,  MPI.DOUBLE], [y_loc, MPI.DOUBLE])

###
#ti = time.time()

# for i in range(x_loc.shape[0]):
# 	z_loc[i]=x_loc[i]*y_loc[i]

for _ in range(400):
	z_loc[:]=x_loc*y_loc

#tf = time.time()
###

comm.Allgather([z_loc,  MPI.DOUBLE], [z, MPI.DOUBLE])
#z=comm.allgather(z_loc)


tf = time.time() 
######


# if rank==0:
# 	print(z)

print("world {0:d} calculation took {1:.4f} sec.".format(rank,tf-ti))



