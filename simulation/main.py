import os,sys

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
# salloc -N 1 -t 30 -C knl -q interactive
# salloc -N 1 -t 30 -C haswell -q interactive


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




