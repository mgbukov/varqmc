import numpy as np
from data_analysis import data





N_MC_points=107
N_epochs=20

model_params=dict(model='RBMcpx',
				  mode='exact', 
				  symm=1,
				  L=4,
				  J2=0.5,
				  opt='RK', #'NG', #'adam',#
				  NNstrct=tuple(tuple(shape) for shape in [[2,16],[2,16]]),
				  epochs=N_epochs,
				  MCpts=N_MC_points,
				  
				)

extra_label='' #'-unique_configs' #
data_structure=data(model_params,N_MC_points,N_epochs,extra_label=extra_label)

data_structure.load()


data_structure.compute_phase_hist()
data_structure.plot(save=0)



