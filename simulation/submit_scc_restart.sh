#!bin/bash
let "N_mpi=448" # 130
let "N_omp=1"
#let "N_tot=280" # 140

if [ -e  submission.sh ]
then
rm submission.sh
fi

echo "#!/bin/bash -login" > submission.sh
echo "#$ -P f-dmrg" >> submission.sh
echo "#$ -N job_VMC" >> submission.sh # Specify parameters in the job name. Don't specify the labels for k and SGE_TASK_ID 
echo "#$ -l h_rt=48:00:00" >> submission.sh

#echo "#$ -pe omp ${N_omp}" >> submission.sh # more processors

echo "#$ -pe mpi_28_tasks_per_node ${N_mpi}" >> submission.sh

#echo "#$ -l mem_per_core=12G" >> submission.sh # memory
echo "#$ -m n" >> submission.sh
echo "module purge" >> submission.sh
echo "module load gcc/5.5.0" >> submission.sh
echo "module load python3/3.7.5" >> submission.sh
#echo "module load openmpi/3.1.4" >> submission.sh
echo "module load miniconda/4.7.5" >> submission.sh
echo "module load hdf5/1.8.21" >> submission.sh
echo "conda activate jax-noGPU" >> submission.sh

data_dir="data/2020_03_15-19_12_08--NG-L_6-MC_restart"

### CPU
#echo mpirun -np ${N_mpi} ~/.conda/envs/jax-noGPU/bin/python ./main.py ${data_dir} >> submission.sh
echo mpiexec -np ${N_mpi} ~/.conda/envs/jax-noGPU/bin/python ./main.py ${data_dir} >> submission.sh

qsub submission.sh
