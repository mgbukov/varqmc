#include "numpy/ndarraytypes.h"
#include "perm_ops.h"

#define cyclicity_factor 1024 //6*6*2*2*2*2

const uint ld_bits = 6;
const uint bits (1 << ld_bits);

#define ld_bits 6
#define bits (1 << ld_bits)  

//#define cyclicity_factor 1024 //8*8*2*2*2*2
//#define N_symms 512 //8*8*2*2*2 // no Z symemtry



const npy_uint64 a_bfly_mask[]={
  // 0..ld_bits
  // For butterfly ops
  // = all_bits / ((1 << (1 << i)) + 1)
  0x5555555555555555,   // 0
  0x3333333333333333,   // 1
  0x0f0f0f0f0f0f0f0f,   // 2
  0x00ff00ff00ff00ff,   // 3
  0x0000ffff0000ffff,   // 4
  0x00000000ffffffff,   // 5
  0xffffffffffffffff};  // 6



inline npy_uint64 shift_x(npy_uint64 x){
	return ((x & 0x0101010101010101) << 7) | ((x & 0xfefefefefefefefe) >> 1);
}


inline npy_uint64 shift_y(npy_uint64 x){
	return rol(x, 56, bits);
}


inline npy_uint64 flip_x(npy_uint64 x){
	return bswap(x, a_bfly_mask, ld_bits);
}


inline npy_uint64 flip_y(npy_uint64 x){
	x = bit_permute_step_simple<npy_uint64>(x, 0x5555555555555555, 1);
	x = bit_permute_step_simple<npy_uint64>(x, 0x3333333333333333, 2);
	x = bit_permute_step_simple<npy_uint64>(x, 0x0f0f0f0f0f0f0f0f, 4);
	return x;
}


inline npy_uint64 flip_d(npy_uint64 x){
	x = bit_permute_step<npy_uint64>(x, 0x00aa00aa00aa00aa, 7);
	x = bit_permute_step<npy_uint64>(x, 0x0000cccc0000cccc, 14);
	x = bit_permute_step<npy_uint64>(x, 0x00000000f0f0f0f0, 28);
	return x;
}


inline npy_uint64 inv_spin(npy_uint64 x){
	return  (0xFFFFFFFFFFFFFFFF ^ x);
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

