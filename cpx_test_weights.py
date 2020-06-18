from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp


C0=jnp.array([
			0.000635608395631, -9.33115842289e-06,
			0.000413545684415, 0.000351460936065,
			0.000447646468557, -0.000536498609612,
			-0.000200352150557, -0.00059763677585,
			0.000416180696879, 0.000645131218371,
			2.62465575302e-05, 2.13328601761e-06,
			0.000294973573169, -0.000528538964996,
			0.000724107145635, 0.000514039015103,
			-0.000894360038816, 0.0003369063605,
			-0.000128385813564, 0.000490693590965,
			5.72395943483e-05, 0.000776345000926,
			0.00076261663827, -6.17585863106e-05,
			-7.42560579852e-05, 5.8514286075e-05,
			0.00056462739014, 0.00013798818975,
			-0.000126367193502, 2.75500005495e-05,
			0.000850500892914, 0.000396817677652,

		], dtype=jnp.float64).reshape(16,2)


C1=jnp.array([

			-9.31314111548e-05, -2.42592890187e-05,
			-0.000240257098128, 5.75583399852e-05,
			0.00029485294998, -0.000553410610877,
			-0.000547768933723, -0.000720583341629,
			-0.000378281353432, 1.19889190242e-05,
			-0.000755627453134, 0.000358207983367,
			0.000203520629652, -3.18391096394e-05,
			-1.10708545835e-05, 0.000139257906033,
			-1.8185463441e-05, 0.000198630891069,
			-0.00035730842921, -0.000199209558387,
			0.000481088831946, 0.000802086076111,
			-2.46519683597e-05, 4.61029174849e-05,
			0.000213439618518, 9.73172758336e-05,
			-0.000582727615438, -0.000537339478462,
			2.23728102549e-05, -0.000248183757149,
			-0.000802390555031, -0.000131297394696,


		], dtype=jnp.float64).reshape(16,2)



C2=jnp.array([

			-5.4128099131e-05, 0.000232410510017,
			9.92818842442e-05, 0.000403352960664,
			0.000407588039615, 0.00065624854078,
			-0.000161892830272, -0.000299708600458,
			0.000258552246897, 0.000941893654745,
			-0.000414402254748, -0.000199934126203,
			-0.000376987569299, 4.80106631154e-05,
			-0.000380400626572, 2.25537208287e-05,
			-0.00015848092339, 5.3567743424e-05,
			-0.000649704488197, -0.000130521710288,
			-0.000474874139211, -0.000736036015942,
			-9.29290920283e-06, -4.5985123388e-05,
			0.000142640605463, -0.000127915763952,
			0.000348765684719, 0.000648500683816,
			-0.000808401650057, 0.000499533668765,
			0.000519392969883, 0.000219206692482,


		], dtype=jnp.float64).reshape(16,2)



C3=jnp.array([

			-6.11412943478e-05, 0.000568210031618,
			6.21920580717e-05, 0.000127065033461,
			0.00089305026692, -0.000158100168139,
			0.000592218866909, -0.000338513770326,
			-0.000493369981135, 0.000183308524139,
			0.000463922440081, 0.000163338103119,
			-0.000608507418895, 0.000295867245697,
			-0.000210632405838, -0.000318144375732,
			0.000586594842876, -0.000649181035031,
			0.000301762111075, 0.000260578273102,
			0.000422149450425, 7.815672151e-06,
			-0.000436954170368, -0.000491788511388,
			-3.86142882591e-06, -8.58457153634e-05,
			0.00022318582446, 0.000121065079667,
			0.000148622774305, -6.58323386933e-05,
			-1.43576525884e-05, -0.000144256928413,


		], dtype=jnp.float64).reshape(16,2)


W_real=jnp.array([C0[:,0],C1[:,0],C2[:,0],C3[:,0]],) #+ 1E-15
W_imag=jnp.array([C0[:,1],C1[:,1],C2[:,1],C3[:,1]],) #- 1E-15

NN_type='CNN' # 'DNN' # 

if NN_type=='CNN':
	W_real=W_real.reshape(4,1,4,4)
	W_imag=W_imag.reshape(4,1,4,4)
else:
	W_real=W_real.T
	W_imag=W_imag.T


# print(jnp.array([C0[:,0],C1[:,0],C2[:,0],C3[:,0]],).shape)
# exit()

# print(W_real.shape)
# exit()

#W_real=jnp.transpose(W_real,(0,1,3,2))
#W_imag=jnp.transpose(W_imag,(0,1,3,2))
