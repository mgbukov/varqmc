import numpy as np 
from scipy.special import comb

np.random.seed(2)

N=16
m=N//2

print(N,m)
print()

s=np.random.randint(comb(N,m))


def decode(ones, ordinal):
	"""
	https://cs.stackexchange.com/questions/67664/prng-for-generating-numbers-with-n-set-bits-exactly

	"""

	bits = 0;
	for bit in range(N-1,-1,-1):
		nCk = int(comb(bit, ones));
		if (ordinal >= nCk):
	  
			ordinal -= nCk;
			bits |= (1 << bit);
			ones-=1;

	return bits;

s0=decode(m,s)

print(s0, '{0:0{1:d}b}'.format(s0,N))
