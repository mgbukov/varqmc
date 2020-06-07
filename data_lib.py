import sys,os,warnings
import numpy as np
import yaml
import pickle


def create_open_file(file_name,load_data,binary=True):
	# open log_file
	if os.path.exists(file_name):
		if load_data:
		    append_write = 'a+' # append if already exists
		else:
			append_write = 'w' # make a new file if not
	else:
		append_write = 'w+' # append if already exists

	if binary:
		append_write+='b'

	return open(file_name, append_write)


def truncate_file(file, start_iter):

	file.close()

	with open(file.name) as f:	
		lines=f.readlines()
		keep_lines=[]
		for i,line in enumerate(lines):
			if i < start_iter:
				keep_lines.append(line)

	with open(file.name, 'wb') as f:
		for line in keep_lines:
			f.write(line.encode('utf8'))

	#return open(file.name, 'ab+')


def flush_all_datafiles(data_files):
	for file in data_files:
		file.flush()	


def close_all_datafiles(data_files):
	for file in data_files:
		file.close()





def read_str(tuple_str):

	shape_tuple=()

	tuple_str=tuple_str.replace('(','')
	tuple_str=tuple_str.replace(')','')
	tuple_str=tuple_str.split(',')


	for NN_str in tuple_str:
		shape_tuple+=(NN_str,)

	return shape_tuple, len(tuple_str)


def load_opt_data(opt,file_name,start_iter):

	with open(file_name) as file:
		for i in range(start_iter):
			opt_data_str = file.readline().rstrip().split(' : ')

	print(start_iter, opt_data_str)	

	opt.iteration=int(opt_data_str[0])+1
	opt.time=np.float64(opt_data_str[7])

	if opt.cost=='SR':
		opt.NG.iteration=int(opt_data_str[0])+1
		opt.NG.delta=np.float64(opt_data_str[1])
		opt.NG.tol=np.float64(opt_data_str[2])

		opt.NG.SNR_weight_sum_exact=np.float64(opt_data_str[3])
		opt.NG.SNR_weight_sum_gauss=np.float64(opt_data_str[4])


	if opt.opt=='RK':
		opt.Runge_Kutta.counter=int(opt_data_str[5])
		opt.Runge_Kutta.step_size=np.float64(opt_data_str[6])
		opt.Runge_Kutta.time=np.float64(opt_data_str[7])
		opt.Runge_Kutta.iteration=int(opt_data_str[0])+1
	else:
		opt.step_size=np.float64(opt_data_str[6])




def store_loss(iteration,r2,grads_max,file_loss,opt):
	data_tuple=(iteration, r2, grads_max, )
	if opt.cost=='SR':
		data_tuple+= (opt.NG.dE, opt.NG.curvature, opt.NG.F_norm, opt.NG.S_norm, opt.NG.S_logcond, )
	else:
		data_tuple+= (0.0, 0.0, 0.0, 0.0, 0.0)
	file_loss.write("{0:d} : {1:0.14f} : {2:0.14f} : {3:0.14f} : {4:0.14f} : {5:0.14f} : {6:0.10f} : {7:0.10f}\n".format(*data_tuple).encode('utf8'))
	

def store_opt_data(iteration,file_opt_data,opt):

	if opt.cost=='SR':
		data_cost=(opt.NG.delta, opt.NG.tol, opt.NG.SNR_weight_sum_exact, opt.NG.SNR_weight_sum_gauss,) 
	else:
		data_cost=(0.0,0.0,0.0,0.0)

	if opt.opt=='RK':
		data_opt=(opt.Runge_Kutta.counter, opt.Runge_Kutta.step_size, opt.Runge_Kutta.time, )
	else:
		data_opt=(opt.iteration,opt.step_size,opt.time,)

	data_tuple=(iteration,)+data_cost+data_opt
	file_opt_data.write("{0:d} : {1:0.14f} : {2:0.14f} : {3:0.14f} : {4:0.14f} : {5:d} : {6:0.14f} : {7:0.14f}\n".format(*data_tuple).encode('utf8'))


def store_debug_helper_data(debug_file_SF,opt,):

	if opt.cost=='SR':
		with open(debug_file_SF+'.pkl', 'wb') as handle:
			pickle.dump([opt.NG.S_lastiters,   opt.NG.F_lastiters,   opt.NG.delta, ], 
							handle, protocol=pickle.HIGHEST_PROTOCOL
						)

	


