#!bin/bash -l
let "N_mpi=32"
let "N_nodes=1"

for N_mpi in $(seq 1 1 26)
	do

	if [ -e  submission.sh ]
	then
	rm submission.sh
	fi


	echo "#!/bin/bash -login" >> submission.sh

	echo "#SBATCH --qos=regular" >> submission.sh
	echo "#SBATCH --time=00:40:00" >> submission.sh
	echo "#SBATCH --nodes=${N_nodes}" >> submission.sh


	echo "#SBATCH --constraint=haswell" >> submission.sh # haswell: 2 sockets x 16 cores per node and 2 threads per core
	#echo "#SBATCH --constraint=knl" >> submission.sh # knl: 68 cores per node and 4 threads per core

	echo "#SBATCH --tasks-per-node=32" >> submission.sh
	#echo "#SBATCH --tasks-per-node=68" >> submission.sh

	echo "#SBATCH --job-name=job_VMC_{N}" >> submission.sh


	echo "module purge" >> submission.sh
	echo "module load cray-hdf5" >> submission.sh
	echo "source activate jax-noGPU" >> submission.sh


	### CPU
	#echo mpiexec -np ${N_mpi} ~/miniconda3/envs/jax-noGPU/bin/python ./main.py >> submission.sh
	#echo ~/miniconda3/envs/jax-noGPU/bin/python ./main.py >> submission.sh

	echo mpiexec -np ${N_mpi} ~/miniconda3/envs/jax-noGPU/bin/python ./main.py >> submission.sh

	sbatch submission.sh


	# wait 1 secs
	sleep 1


done

