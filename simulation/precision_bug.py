import pickle
import numpy as np 



file_name='./bug'
with open(file_name+'.pkl', 'rb') as handle:
	dlog_psi, OO_expt, O_expt2, O_expt, Fisher, grad = pickle.load(handle)


print(grad)
