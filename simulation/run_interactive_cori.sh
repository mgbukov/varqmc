N_nodes=32
N_mpi=$(( 32*${N_nodes} ))

data_dir="$(/global/cfs/cdirs/m3444/.conda/envs/jax-noGPU/bin/python make_data_file_linux.py)"

srun -c 2 --cpu_bind=cores -n ${N_mpi} /global/cfs/cdirs/m3444/.conda/envs/jax-noGPU/bin/python ./main.py ${data_dir}

