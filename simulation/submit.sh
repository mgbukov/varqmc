#!bin/bash

echo "#!/bin/bash -login" >> submission.sh
echo "#$ -P f-dmrg" >> submission.sh
echo "#$ -N job_VMC" >> submission.sh # Specify parameters in the job name. Don't specify the labels for k and SGE_TASK_ID 
echo "#$ -l h_rt=12:00:00" >> submission.sh
echo "# -pe omp 40" >> submission.sh # more processors
#echo "#$ -l mem_per_core=16G" >> submission.sh # memory
echo "#$ -m n" >> submission.sh

echo /projectnb/f-dmrg/mbukov/.conda/envs/jax/bin/python main.py >> submission.sh

qsub submission.sh

rm submission.sh

