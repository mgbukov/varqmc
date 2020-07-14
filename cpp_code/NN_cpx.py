from jax.config import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp

from .DNN_architectures_common import *
#from .DNN_architectures_real import *
from .DNN_architectures_cpx import *


def NN_cpx_arch(net_str, shapes, input_shape, reduce_shape, output_shape, scale ):

	NN_archs=dict()
	n_layers=len(shapes)

	if 'DNN' in net_str:

		if n_layers==1:
			
			NN_archs['DNN_1']= {

								'layer_1': GeneralDenseComplex(1, shapes['layer_1'][1], input_shape[-1], ignore_b=True, init_value_W=scale), 
								'nonlin_1': elementwise(logcosh_cpx),
								'symmpool': elementwise(symmetric_pool_cpx, reduce_shape=reduce_shape, output_shape=output_shape, norm=jnp.sqrt(output_shape[1]*(reduce_shape[1]+reduce_shape[3])) ),
							#	'reg': RegularizationComplex(a=8.0),

								}

		elif n_layers==2:
			
			NN_archs['DNN_2']=	{

								'layer_1': GeneralDenseComplex(1, shapes['layer_1'][1], input_shape[-1], ignore_b=True, init_value_W=scale),  
								'nonlin_1': elementwise(poly_cpx),
								'layer_2': GeneralDenseComplex(shapes['layer_2'][0], shapes['layer_2'][1], 1 , ignore_b=False, init_value_W=scale, init_value_b=scale ),  
								'nonlin_2': elementwise(poly_cpx),
								'symm': elementwise(symmetrize_cpx, reduce_shape=reduce_shape, ),
								'pool': elementwise(uniform_pool_cpx, output_shape=output_shape ),
								'reg': RegularizationComplex(a=8.0),

								}


	##########################################################################
	##########################################################################
	##########################################################################


	elif 'CNN' in net_str:	

		dim_nums=('NCHW', 'OIHW', 'NCHW') # default CNN

		if n_layers==1:

			NN_archs['CNN_1']= {

										'layer_1': GeneralConvPeriodicComplex(dim_nums, shapes['layer_1'][1], shapes['layer_1'][0], dense_output=True, ignore_b=True, init_value_W=scale, ), 
										'nonlin_1': elementwise(logcosh_cpx),
										'symmpool': elementwise(symmetric_pool_cpx, reduce_shape=reduce_shape, output_shape=output_shape, norm=jnp.sqrt(output_shape[1]*(reduce_shape[1]+reduce_shape[3])) ),
										
										}

		elif n_layers==2:

			NN_archs['CNN_as_dnn_2']= {

										'layer_1': GeneralConvPeriodicComplex(dim_nums, shapes['layer_1'][1], shapes['layer_1'][0], dense_output=True, ignore_b=True, init_value_W=scale, ), 
										'nonlin_1': elementwise(poly_cpx),
										'layer_2': GeneralDenseComplex(shapes['layer_1'][1], shapes['layer_2'][1], 1, ignore_b=False, init_value_W=scale, init_value_b=scale ), 
										'nonlin_2': elementwise(poly_cpx),
										'symm': elementwise(symmetrize_cpx, reduce_shape=reduce_shape, norm=jnp.sqrt(reduce_shape[1]+reduce_shape[3])),
										'pool': elementwise(uniform_pool_cpx, output_shape=output_shape, norm=jnp.sqrt(output_shape[1]) ),
										'reg': RegularizationComplex(a=8.0),
										
										}
		
		elif n_layers==3:

			NN_archs['CNN_mixed_3']= {

										'layer_1': GeneralConvPeriodicComplex(dim_nums, shapes['layer_1'][1], shapes['layer_1'][0], ignore_b=True, init_value_W=scale, ), 
										'nonlin_1': elementwise(poly_cpx),

										'layer_2': GeneralConvPeriodicComplex(dim_nums, shapes['layer_2'][1], shapes['layer_2'][0], dense_output=True, ignore_b=False, init_value_W=scale, init_value_b=scale, ), 
										'nonlin_2': elementwise(poly_cpx),

										'layer_3': GeneralDenseComplex(shapes['layer_2'][1], shapes['layer_3'][1], 1, ignore_b=False, init_value_W=scale, init_value_b=scale, ), 
										'nonlin_3': elementwise(poly_cpx),

										'symm': elementwise(symmetrize_cpx, reduce_shape=reduce_shape, norm=jnp.sqrt(reduce_shape[1]+reduce_shape[3])),
										'pool': elementwise(uniform_pool_cpx, output_shape=output_shape, norm=jnp.sqrt(output_shape[1]) ),
										
										'reg': RegularizationComplex(a=8.0),
										
										}


	return NN_archs[net_str]

