import sys
sys.path.append("..")

from VMC_class import VMC
import yaml 
from threadpoolctl import threadpool_limits

# export KMP_DUPLICATE_LIB_OK=TRUE
# export OMP_NUM_THREADS=4
# mpiexec -n 4 python main.py 



params = yaml.load(open('config_params.yaml'),Loader=yaml.FullLoader)

with threadpool_limits(limits=1):
	DNN_psi=VMC(params)




