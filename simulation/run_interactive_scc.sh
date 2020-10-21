N_nodes=28
N_mpi=$(( 16*${N_nodes} ))

data_dir="$(~/.conda/envs/jax-noGPU/bin/python make_data_file_linux.py)"

mpiexec -np ${N_mpi} ~/.conda/envs/jax-noGPU/bin/python ./main.py ${data_dir}