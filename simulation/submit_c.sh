#!bin/bash
let "N_mpi=2"
let "N_omp=1"
let "N_tot=2"

if [ -e  submission.sh ]
then
rm submission.sh
fi

echo "#!/bin/bash -login" >> submission.sh
echo "#SBATCH --qos=debug" >> submission.sh
echo "#SBATCH --time=00:05:00" >> submission.sh


echo "#SBATCH --constraint=haswell" >> submission.sh 
echo "#SBATCH --nodes=${N_tot}" >> submission.sh
echo "#SBATCH --tasks-per-node=32" >> submission.sh

echo "#SBATCH --job-name=job_VMC" >> submission
#echo "#SBATCH --mail-user=mgbukov@berkeley.edu" >> submission.sh


echo "module purge" >> submission.sh
echo "module load cray-hdf5" >> submission.sh
echo "source activate jax-noGPU" >> submission.sh


### CPU
#echo srun mpiexec -np ${N_mpi} ~/miniconda3/envs/jax-noGPU/bin/python ./main.py >> submission.sh
echo srun ~/miniconda3/envs/jax-noGPU/bin/python ./main.py >> submission.sh

#echo  mpiexec -np ${N_mpi} ~/miniconda3/envs/jax-noGPU/bin/python ./main.py >> submission.sh

sbatch submission.sh
#sh submission.sh

