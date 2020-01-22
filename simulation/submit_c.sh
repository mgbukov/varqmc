#!bin/bash
let "N_mpi=32"
let "N_omp=1"
let "N_tot=32"

if [ -e  submission.sh ]
then
rm submission.sh
fi

echo "#!/bin/bash -login" >> submission.sh
echo "#SBATCH --qos=debug" >> submission.sh
echo "#SBATCH --time=00:30:00" >> submission.sh


echo "#SBATCH --constraint=haswell" >> submission.sh 
echo "#SBATCH --tasks-per-node=${N_tot}" >> submission.sh
echo "#SBATCH --nodes=2" >> submission.sh

echo "#SBATCH --job-name=job_VMC" >> submission

#echo "#SBATCH --mail-type=end,fail" >> submission
#echo "#SBATCH --mail-user=mgbukov@berkeley.edu" >> submission.sh


echo "module purge" >> submission.sh
echo "module load cray-hdf5" >> submission.sh
echo "conda activate jax-noGPU" >> submission.sh


### CPU
#echo mpirun -np ${N_mpi} ~/.conda/envs/jax-noGPU/bin/python ./main.py >> submission.sh
echo srun mpiexec -np ${N_mpi} ~/.conda/envs/jax-noGPU/bin/python ./main.py >> submission.sh


sbatch submission.sh

