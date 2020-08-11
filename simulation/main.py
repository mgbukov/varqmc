import os,sys

if sys.platform == "linux" or sys.platform == "linux2": # linux
    from make_data_file_linux import create_params_file

elif sys.platform == "darwin": # OS X
	from make_data_file import create_params_file


sys.path.append("..")

from VMC_class import VMC
import yaml 

# export KMP_DUPLICATE_LIB_OK=TRUE
# export OMP_NUM_THREADS=4
# mpiexec -n 4 python main.py --test

# check memory of finished job
# sacct -j "" --format="JobID,MaxRSS,AveRSS,MaxRSSNode"

# request interactive node
# salloc -N 32 -t 120 -C haswell -q interactive

# qrsh -now n -l h_rt=12:00:00 -pe mpi_28_tasks_per_node 280


if len(sys.argv)>1:

	if '--test' in sys.argv:
		# initialize new simulation
	    params = yaml.load(open('config_params_test.yaml'),Loader=yaml.FullLoader)
	    data_dir = create_params_file(params)
	else:
		# load simulation
		data_dir = sys.argv[1]

else:
	# initialize new simulation
	params = yaml.load(open('config_params.yaml'),Loader=yaml.FullLoader)
	data_dir = create_params_file(params)


DNN_psi=VMC(data_dir)




