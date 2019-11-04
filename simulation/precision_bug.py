import pickle
import numpy as np 
from scipy.linalg import eigh,eig



file_name='./bug'
with open(file_name+'.pkl', 'rb') as handle:
	dlog_psi_osx, OO_expt_osx, O_expt2_osx, O_expt_osx, Fisher_osx, grad_osx = pickle.load(handle)

norm_osx=np.linalg.norm(Fisher_osx)


file_name='./bug_linux'
with open(file_name+'.pkl', 'rb') as handle:
	dlog_psi_linux, OO_expt_linux, O_expt2_linux, O_expt_linux, Fisher_linux, grad_linux = pickle.load(handle)


norm_linux=np.linalg.norm(Fisher_linux)


def max_mismatch(x,y):
	norm_x=np.linalg.norm(x)
	norm_y=np.linalg.norm(y)
	return np.max(np.abs(np.abs(x/norm_x) - np.abs(y/norm_y)))

print(
	max_mismatch(grad_linux,grad_osx),
	max_mismatch(Fisher_linux,Fisher_osx),
	max_mismatch(O_expt_linux,O_expt_osx),
	max_mismatch(O_expt2_linux,O_expt2_osx),
	max_mismatch(OO_expt_linux,OO_expt_osx),
	max_mismatch(dlog_psi_linux,dlog_psi_osx),
	'\n'
	)


E_osx = eigh(Fisher_osx/norm_osx,eigvals_only=True)
E_linux = eigh(Fisher_linux/norm_linux,eigvals_only=True)
E_linux2, _ = eig(Fisher_linux/norm_linux)
E_linux2=np.sort(E_linux2.real)

print(np.linalg.cond(Fisher_linux), np.linalg.cond(Fisher_linux/norm_linux))
print(np.linalg.cond(Fisher_osx), np.linalg.cond(Fisher_osx/norm_osx))
print()


print(E_linux2)




