import os,sys

from make_data_file import create_params_file

sys.path.append("..")

from VMC_class import VMC
import yaml 

# export KMP_DUPLICATE_LIB_OK=TRUE
# export OMP_NUM_THREADS=4
# mpiexec -n 4 python main.py --test

# sacct -j "" --format="JobID,MaxRSS,AveRSS,MaxRSSNode"


if '--test' in sys.argv:
    params = yaml.load(open('config_params_test.yaml'),Loader=yaml.FullLoader)
    data_dir=create_params_file(params)
else:
	#params = yaml.load(open('config_params.yaml'),Loader=yaml.FullLoader)
	data_dir=sys.argv[1]


DNN_psi=VMC(data_dir)




