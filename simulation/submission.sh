#!/bin/bash -login
#SBATCH --qos=debug
#SBATCH --time=00:03:00
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --tasks-per-node=32
#SBATCH --job-name=job_VMC
module purge
module load cray-hdf5
source activate jax-noGPU
srun --cpu_bind=cores /global/homes/m/mgbukov/miniconda3/envs/jax-noGPU/bin/python ./main.py
