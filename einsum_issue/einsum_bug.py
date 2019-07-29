import sys
import numpy as np
print(np.__version__, sys.version)

# the data files are attached below
complex_vecs=np.loadtxt('einsum_bug_vec.txt', delimiter=',').view(complex)   # (107,32) array
real_weights=np.loadtxt('einsum_bug_weights.txt', delimiter=',')   # (107,) array

print(complex_vecs.shape, real_weights.shape)

#
A_loop=np.zeros((32,32),dtype=np.complex128)
for s in range(107):
	for i in range(32):
		for j in range(32):
			A_loop[i,j]+=real_weights[s]*complex_vecs[s,i].conj()*complex_vecs[s,j]

print(np.linalg.norm(A_loop-A_loop.T.conj())) # 0.0, as it should be
print(np.max(np.abs(A_loop)),np.min(np.abs(A_loop)))
exit()


# pedestrian calculation: gives a hermitian matrix
A=np.zeros((32,32),dtype=np.complex128)
scaled_vecs=np.zeros_like(complex_vecs)
for s,vec in enumerate(complex_vecs):
	A+=real_weights[s]*np.einsum('i,j->ij',vec.conj(),vec) 

	scaled_vecs[s]=real_weights[s]*vec

scaled_vecs2 = np.einsum('s,si->si',real_weights,complex_vecs)

A_tensordot=np.tensordot(complex_vecs.conj(),scaled_vecs2,axes=([0,],[0,]))


#print(np.linalg.norm(scaled_vecs-scaled_vecs2))


# einsum calculation: erroneously gives a non-hermitian matrix
A_einsum=np.einsum('s,si,sj->ij',real_weights,complex_vecs.conj(),complex_vecs,optimize=False)

A_einsum2=np.einsum('si,sj->ij',complex_vecs.conj(), np.einsum('s,si->si',real_weights,complex_vecs) )

A_einsum3=np.einsum('s,sij->ij',real_weights, np.einsum('si,sj->sij',complex_vecs.conj(),complex_vecs) ) 

print(np.linalg.norm(A_einsum), np.linalg.norm(complex_vecs))

print(np.linalg.norm(A-A.T.conj())) # 0.0, as it should be
print(np.linalg.norm((A_einsum-A_einsum.T.conj())/np.linalg.norm(A_einsum) )) # 0.19771954288603175, this is suspicious
exit()
print(np.linalg.norm(A_einsum2-A_einsum2.T.conj())) # 0.19771954288603175, this is suspicious
print(np.linalg.norm(A_tensordot-A_tensordot.T.conj())) # 0.19771954288603175, this is suspicious
print(np.linalg.norm(A_einsum3-A_einsum3.T.conj())) # 0.19771954288603175, this is suspicious

print()

print(np.linalg.norm(A-A_einsum)) # 0.254093277317957, the two matrices should be equal
print(np.linalg.norm(A-A_einsum3))
