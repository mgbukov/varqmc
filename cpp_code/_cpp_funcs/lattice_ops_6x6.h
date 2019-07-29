#include "numpy/ndarraytypes.h"

#define cyclicity_factor 576 //6*6*2*2*2*2
#define N_symms 288 //6*6*2*2*2 // no Z symemtry


inline npy_uint64 shift_x(npy_uint64 x){
	return ((x & 0x0000000041041041) << 5) | ((x & 0x0000000fbefbefbe) >> 1);
}


inline npy_uint64 shift_y(npy_uint64 x){
	return ((x & 0x000000000000003f) << 30) | ((x & 0x0000000fffffffc0) >> 6);
}


inline npy_uint64 flip_x(npy_uint64 x){
	return (((x & 0x0000000104104104) << 1) \
		  	| ((x & 0x0000000082082082) << 3) \
		  	| ((x & 0x0000000041041041) << 5) \
		  	| ((x & 0x0000000820820820) >> 5) \
		  	| ((x & 0x0000000410410410) >> 3) \
		  	| ((x & 0x0000000208208208) >> 1) \
			);
}


inline npy_uint64 flip_y(npy_uint64 x){
	return  (((x & 0x000000000003f000) << 6) \
  			| ((x & 0x0000000000000fc0) << 18) \
  			| ((x & 0x000000000000003f) << 30) \
  			| ((x & 0x0000000fc0000000) >> 30) \
  			| ((x & 0x000000003f000000) >> 18) \
 			| ((x & 0x0000000000fc0000) >> 6) \
			);
}


inline npy_uint64 flip_d(npy_uint64 x){
	x = (((x & 0x0000000204081000) << 15) \
		| ((x & 0x0000000810204081) << 25) \
		| ((x & 0x0000000408102040) << 20) \
		| ((x & 0x0000000020408102) << 30) \
		| (((x & 0x0000000183820838) * 0x0004210800000420) & 0x07820c1820000000) \
		| (((x & 0x0000000040050604) * 0x0000010800000401) & 0x000c102050000000));
	return (x << 39) | (x >> 25); // x = rol(x, 39);
}


inline npy_uint64 inv_spin(npy_uint64 x){
	return  (0x0000000FFFFFFFFF ^ x);
}





int check_state(npy_uint64 s){
	npy_uint64  t = s;
	int norm=0;

	for(int i=0;i<2;i++){
			for(int j=0;j<2;j++){
				for(int k=0;k<2;k++){
					for(int l=0;l<2;l++){
						for(int m=0;m<_L;m++){
							for(int n=0;n<_L;n++){
								if(t == s) norm++;
								t = shift_x(t);
								if(t > s) return -1;
							}
							t = shift_y(t);
							if(t > s) return -1;
						}
						t = inv_spin(t);
						if(t > s) return -1;
					}
					t = flip_x(t);
					if(t > s) return -1;
				}
				t = flip_y(t);
				if(t > s) return -1;
			}
			t = flip_d(t);
			if(t > s) return -1;
		}

	return norm;
}



npy_uint64 ref_state(npy_uint64 s){
	npy_uint64  t = s;
	npy_uint64  r = s;
	for(int i=0;i<2;i++){
			for(int j=0;j<2;j++){
				for(int k=0;k<2;k++){
					for(int l=0;l<2;l++){
						for(int m=0;m<_L;m++){
							for(int n=0;n<_L;n++){
								if(t > r) r = t;
	
								t = shift_x(t);
							}
							t = shift_y(t);
						}
						t = inv_spin(t);
					}
					t = flip_x(t);
				}
				t = flip_y(t);
			}
			t = flip_d(t);
		}

	return r;
}

