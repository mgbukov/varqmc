#!bin/bash
let "N_mpi=12"
let "N_omp=1"
let "N_tot=12"

echo "#!/bin/bash -login" >> submission.sh
echo "#$ -P f-dmrg" >> submission.sh
echo "#$ -N job_VMC" >> submission.sh # Specify parameters in the job name. Don't specify the labels for k and SGE_TASK_ID 
echo "#$ -l h_rt=2:00:00" >> submission.sh

echo "#$ -pe omp ${N_omp}" >> submission.sh # more processors

echo "#$ -pe mpi_4_tasks_per_node ${N_tot}" >> submission.sh

#echo "#$ -l mem_per_core=12G" >> submission.sh # memory
echo "#$ -m n" >> submission.sh


#echo mpirun -np ${N_mpi} /projectnb/f-dmrg/mbukov/.conda/envs/jax/bin/python main.py >> submission.sh
echo mpirun -np ${N_mpi} ~/.conda/envs/jax-noGPU/bin/python ./main.py >> submission.sh

qsub submission.sh

rm submission.sh

