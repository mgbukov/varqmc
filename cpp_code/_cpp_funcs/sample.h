
#if _L==4
	#include "./lattice_ops_4x4.h"
#elif _L==6
	#include "./lattice_ops_6x6.h"
#elif _L==8
	#include "./lattice_ops_8x8.h"
#endif

#include "symmetrized/common_funcs.h"