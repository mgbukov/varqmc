#!/bin/bash -login
#SBATCH --qos=regular
#SBATCH --time=08:03:00
#SBATCH --constraint=haswell
#SBATCH --nodes=5
#SBATCH --tasks-per-node=26
#SBATCH --job-name=job_VMC
module purge
module load cray-hdf5
source activate jax-noGPU
mpiexec -np 130 /global/homes/m/mgbukov/miniconda3/envs/jax-noGPU/bin/python ./main.py
