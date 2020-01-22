#!/bin/bash -login
#SBATCH --qos=debug
#SBATCH --time=00:05:00
#SBATCH --constraint=haswell
#SBATCH --nodes=2
#SBATCH --tasks-per-node=32
module purge
module load cray-hdf5
source activate jax-noGPU
srun /global/homes/m/mgbukov/miniconda3/envs/jax-noGPU/bin/python ./main.py
