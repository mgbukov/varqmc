1. MPI, osx: https://stackoverflow.com/questions/54811518/mpi-gather-call-hangs-for-large-ish-arrays

#$ brew unlink open-mpi
#$ brew install mpich
#$ pip uninstall mpi4py
#$ pip install mpi4py --no-cache-dir
$ conda install mpi4py # mpiexec

Then, I had to edit /etc/hosts and add the line

127.0.0.1     <mycomputername> # see below
import socket
print(socket.gethostname()) # mycomputername


2. yaml: 

conda install yaml
pip install pyyaml