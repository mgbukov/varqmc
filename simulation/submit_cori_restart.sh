#!bin/bash -l
let "N_mpi=260" #26*N_nodes
let "N_nodes=10" # 5

if [ -e  submission.sh ]
then
rm submission.sh
fi

echo "#!/bin/bash -login" > submission.sh

#echo "#SBATCH --account = <NERSC Repository>" >> submission.sh
#echo "#SBATCH --qos=debug" >> submission.sh
#echo "#SBATCH --qos=regular" >> submission.sh 
echo "#SBATCH --qos=premium" >> submission.sh
echo "#SBATCH --time=24:00:00" >> submission.sh

echo "#SBATCH --constraint=haswell" >> submission.sh # haswell: 2 sockets x 16 cores per node and 2 threads per core
#echo "#SBATCH --constraint=knl" >> submission.sh # knl: 68 cores per node and 4 threads per core

echo "#SBATCH --nodes=${N_nodes}" >> submission.sh
#echo "#SBATCH --tasks=4" >> submission.sh # total number of tasks
#echo "#SBATCH --cpus-per-task=272" >> submissio.sh # #OMP processes

echo "#SBATCH --tasks-per-node=26" >> submission.sh # 26/32
#echo "#SBATCH --tasks-per-node=68" >> submission.sh



echo "#SBATCH --job-name=job_VMC" >> submission.sh
#echo "#SBATCH --mail-user=mgbukov@berkeley.edu" >> submission.sh


echo "module purge" >> submission.sh
echo "module load cray-hdf5" >> submission.sh
echo "source activate jax-noGPU" >> submission.sh


data_dir="data/2020_03_15-19_12_08--NG-L_6-MC_restart"


### CPU
#echo ~/miniconda3/envs/jax-noGPU/bin/python ./main.py ${data_dir} >> submission.sh

echo mpirun -np ${N_mpi} ~/miniconda3/envs/jax-noGPU/bin/python ./main.py ${data_dir} >> submission.sh



sbatch submission.sh


