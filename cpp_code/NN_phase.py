from jax.config import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp

from .DNN_architectures_common import *
from .DNN_architectures_real import *
#from .DNN_architectures_cpx import *




def NN_phase_arch(net_str, shapes, input_shape, reduce_shape, output_shape, scale ):

	NN_archs=dict()
	n_layers=len(shapes)

	if 'DNN' in net_str:

		if n_layers==1:
			pass

		elif n_layers==2:

			NN_archs['DNN_2']=	{
								'layer_1': GeneralDense(1, shapes['layer_1'][1], input_shape[-1], ignore_b=True, init_value_W=scale ),  
								'nonlin_1': elementwise(logcosh),
								'layer_2': GeneralDense(shapes['layer_2'][0], shapes['layer_2'][1], 1,ignore_b=False, init_value_W=scale, init_value_b=scale ),
								'nonlin_2': elementwise(xtanh),
								# 'symm': elementwise(symmetrize, reduce_shape=reduce_shape, ),
								# 'pool': elementwise(uniform_pool, output_shape=output_shape ),
								'pool': elementwise(symmetric_pool, reduce_shape=reduce_shape, output_shape=output_shape ),					
								}


	##########################################################################
	##########################################################################
	##########################################################################


	elif 'CNN' in net_str:	

		dim_nums=('NCHW', 'OIHW', 'NCHW') # default CNN

		if n_layers==1:
			pass

		elif n_layers==2:

			NN_archs['CNN_as_dnn_2']= {
									'layer_1': GeneralConvPeriodic(dim_nums, shapes['layer_1'][1], shapes['layer_1'][0], dense_output=True, ignore_b=True, init_value_W=scale, ), 
									'nonlin_1': elementwise(logcosh),
									'layer_2': GeneralDense(shapes['layer_1'][1], shapes['layer_2'][1], 1, ignore_b=False, init_value_W=scale, init_value_b=scale ), 
									'nonlin_2': elementwise(xtanh),
									'pool': elementwise(symmetric_pool, reduce_shape=reduce_shape, output_shape=output_shape ),					
									}


			NN_archs['CNN_pure_2']=	{
									'layer_1': GeneralConvPeriodic(dim_nums, shapes['layer_1'][1], shapes['layer_1'][0], ignore_b=True, init_value_W=scale, ), 
									'nonlin_1': elementwise(logcosh),
									'layer_2': GeneralConvPeriodic(dim_nums, shapes['layer_2'][1], shapes['layer_2'][0], ignore_b=False, init_value_W=scale, init_value_b=scale, ), 
									'nonlin_2': elementwise(xtanh),
									'pool': elementwise(symmetric_pool, reduce_shape=reduce_shape, output_shape=output_shape ),
									},
			

		elif n_layers==3:

			NN_archs['CNN_mixed_3']= {
									'layer_1': GeneralConvPeriodic(dim_nums, shapes['layer_1'][1], shapes['layer_1'][0], dense_output=True, ignore_b=True, init_value_W=scale, ), 
									'nonlin_1': elementwise(logcosh),

									'layer_2': GeneralDense(shapes['layer_1'][1], shapes['layer_2'][1], 1, ignore_b=False, init_value_W=scale, init_value_b=scale ),  # 1E-1
									'nonlin_2': elementwise(xtanh),

									'layer_3': GeneralDense(shapes['layer_2'][1], shapes['layer_3'][1], 1, ignore_b=False, init_value_W=scale, init_value_b=scale ),  # 1E-1
									'nonlin_3': elementwise(xtanh),

									'symm': elementwise(symmetrize, reduce_shape=reduce_shape, ),
									'pool': elementwise(uniform_pool, output_shape=output_shape ),

									#'pool': elementwise(symmetric_pool, reduce_shape=reduce_shape, output_shape=output_shape ),
									}


		


		

	return NN_archs[net_str]

