#!/bin/bash -login
#$ -P f-dmrg
#$ -N job_VMC
#$ -l h_rt=00:01:00
#$ -pe mpi_28_tasks_per_node 28
#$ -m n
module purge
module load gcc/5.5.0
module load python3/3.7.5
module load miniconda/4.7.5
conda activate jax-debug
mpiexec -np 24 /projectnb/f-dmrg/mbukov/.conda/envs/jax-debug/bin/python -W ignore ./bug.py
