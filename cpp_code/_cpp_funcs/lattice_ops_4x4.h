#include "numpy/ndarraytypes.h"
#include "perm_ops.h"



const uint ld_bits = 4;
const uint bits (1 << ld_bits);

#define ld_bits 4
#define bits (1 << ld_bits)  
#define cyclicity_factor 256 //4*4*2*2*2*2
#define N_symms 128 //4*4*2*2*2 // no Z symemtry



const npy_uint16 a_bfly_mask[]={
  // 0..ld_bits
  // For butterfly ops
  // = all_bits / ((1 << (1 << i)) + 1)
  (npy_uint16) 0x5555,   // 0
  (npy_uint16) 0x3333,   // 1
  (npy_uint16) 0x0f0f,   // 2
  (npy_uint16) 0x00ff,   // 3
  (npy_uint16) 0xffff};  // 4




inline npy_uint16 shift_x(npy_uint16 x){
	return ((x & 0x1111) << 3) | ((x & 0xeeee) >> 1);
}

inline npy_uint16 shift_y(npy_uint16 x){
	return rol(x, 12, bits);
}


inline npy_uint16 flip_x(npy_uint16 x){
	return rol(x & 0xf0f0, 4, bits) | rol(x & 0x0f0f, 12, bits);
}


inline npy_uint16 flip_y(npy_uint16 x){
	x = bit_permute_step_simple<npy_uint16>(x, 0x5555, 1);
	x = bit_permute_step_simple<npy_uint16>(x, 0x3333, 2);
	return x;
}


inline npy_uint16 rot_90(npy_uint16 x){
	x = bswap(x, a_bfly_mask, ld_bits);
	x = bit_permute_step<npy_uint16>(x, 0x0a0a, 3);
	x = bit_permute_step<npy_uint16>(x, 0x0033, 10);
	x = bit_permute_step_simple<npy_uint16>(x, 0x0f0f, 4);
	return x;
}

inline npy_uint16 flip_d(npy_uint16 x){
	x = bit_permute_step<npy_uint16>(x, 0x0a0a, 3);
	x = bit_permute_step<npy_uint16>(x, 0x00cc, 6);
	return x;
}


inline npy_uint16 inv_spin(npy_uint16 x){
	return (0xFFFF ^ x);
}




int check_state(npy_uint16 s){
	npy_uint16  t = s;
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



npy_uint16 ref_state(npy_uint16 s){
	npy_uint16  t = s;
	npy_uint16  r = s;
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






