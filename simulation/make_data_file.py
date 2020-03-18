import sys, os
from mpi4py import MPI
import datetime
import yaml


		

def create_params_file(params):

	comm=MPI.COMM_WORLD

	# system time
	
	sys_data=''

	if comm.Get_rank()==0:
		sys_time=datetime.datetime.now()
		sys_data="{0:d}_{1:02d}_{2:02d}-{3:02d}_{4:02d}_{5:02d}--".format(sys_time.year, sys_time.month, sys_time.day, sys_time.hour, sys_time.minute, sys_time.second)
	
	# broadcast sys_data
	sys_data = comm.bcast(sys_data, root=0)


	sys_time=sys_data + params['optimizer'] + '-L_{0:d}-{1:s}'.format(params['L'],params['mode'])


	data_dir=os.getcwd()+'/data/'+sys_time
	#params['data_dir']=data_dir


	if (not os.path.exists(data_dir)) and comm.Get_rank()==0:
		os.makedirs(data_dir)
	comm.Barrier()


	# check if file exists and create it if not
	#if not os.path.isfile(data_dir + '/config_params_init.yaml'):

	config_params_init = open(data_dir + '/config_params_init.yaml', 'w')
	yaml.dump(params, config_params_init)
	config_params_init.close()


	return data_dir



if __name__ == "__main__":

	if '--test' in sys.argv:
		params = yaml.load(open('config_params_test.yaml'),Loader=yaml.FullLoader)
	else:
		params = yaml.load(open('config_params.yaml'),Loader=yaml.FullLoader)

	data_dir = create_params_file(params)
	print(data_dir)




