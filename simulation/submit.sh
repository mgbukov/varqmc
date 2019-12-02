#!bin/bash

echo "#!/bin/bash -login" >> submission.sh
echo "#$ -P f-dmrg" >> submission.sh
echo "#$ -N job_VMC" >> submission.sh # Specify parameters in the job name. Don't specify the labels for k and SGE_TASK_ID 
echo "#$ -l h_rt=12:00:00" >> submission.sh
echo "#$ -pe omp 8" >> submission.sh # more processors
echo "#$ -pe mpi_4_tasks_per_node 40" >> submission.sh
#echo "#$ -l mem_per_core=16G" >> submission.sh # memory
echo "#$ -m n" >> submission.sh

#echo mpirun -np $NSLOTS /projectnb/f-dmrg/mbukov/.conda/envs/jax/bin/python main.py >> submission.sh
echo mpirun -np $NSLOTS ~/.conda/envs/jax-noGPU/bin/python main.py >> submission.sh



qsub submission.sh

rm submission.sh

