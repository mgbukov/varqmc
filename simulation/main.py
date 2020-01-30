import sys
sys.path.append("..")

from VMC_class import VMC
import yaml 

# export KMP_DUPLICATE_LIB_OK=TRUE
# export OMP_NUM_THREADS=4
# mpiexec -n 4 python main.py 


if str(sys.argv[1])=='--test':
    params = yaml.load(open('config_params_test.yaml'),Loader=yaml.FullLoader)
else:
    params = yaml.load(open('config_params.yaml'),Loader=yaml.FullLoader)

DNN_psi=VMC(params)




