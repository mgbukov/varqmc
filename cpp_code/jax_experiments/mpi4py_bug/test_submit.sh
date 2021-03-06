#!bin/bash
let "N_mpi=26" # 130
let "N_omp=1"
let "N_tot=28" # 140

if [ -e  submission.sh ]
then
rm submission.sh
fi

echo "#!/bin/bash -login" > submission.sh
echo "#$ -P f-dmrg" >> submission.sh
echo "#$ -N job_VMC" >> submission.sh # Specify parameters in the job name. Don't specify the labels for k and SGE_TASK_ID 
echo "#$ -l h_rt=00:01:00" >> submission.sh

#echo "#$ -pe omp ${N_omp}" >> submission.sh # more processors

echo "#$ -pe mpi_28_tasks_per_node ${N_tot}" >> submission.sh

#echo "#$ -l mem_per_core=12G" >> submission.sh # memory
echo "#$ -m n" >> submission.sh
echo "module purge" >> submission.sh
echo "module load gcc/5.5.0" >> submission.sh
echo "module load python3/3.7.5" >> submission.sh
#echo "module load openmpi/3.1.4" >> submission.sh
echo "module load miniconda/4.7.5" >> submission.sh
#echo "module load hdf5/1.8.21" >> submission.sh
echo "conda activate jax-debug" >> submission.sh


echo mpiexec -np ${N_mpi} /projectnb/f-dmrg/mbukov/.conda/envs/jax-debug/bin/python -W ignore ./bug.py >> submission.sh

#qsub submission.sh

