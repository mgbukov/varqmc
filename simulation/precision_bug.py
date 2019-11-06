import pickle
import numpy as np 
from scipy.linalg import eigh,eig
from scipy.sparse.linalg import cg



file_name='./bug'
with open(file_name+'.pkl', 'rb') as handle:
	dlog_psi_osx, OO_expt_osx, O_expt2_osx, O_expt_osx, Fisher_osx, grad_osx = pickle.load(handle)



file_name='./bug_linux'
with open(file_name+'.pkl', 'rb') as handle:
	dlog_psi_linux, OO_expt_linux, O_expt2_linux, O_expt_linux, Fisher_linux, grad_linux = pickle.load(handle)


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



norm_osx=np.linalg.norm(Fisher_osx)
norm_linux=np.linalg.norm(Fisher_linux)

E_osx = eigh(Fisher_osx/norm_osx,eigvals_only=True)
E_linux = eigh(Fisher_linux/norm_linux,eigvals_only=True)
E_linux2, _ = eig(Fisher_linux/norm_linux)
E_linux2=np.sort(E_linux2.real)

#print(E_osx)
#exit()


E_linux = eigh(Fisher_linux/norm_linux +1E-15*np.eye(Fisher_osx.shape[0])/norm_linux,eigvals_only=True)
#print(E_linux)
#exit()

np.linalg.cholesky(Fisher_osx+1E-15*np.eye(Fisher_osx.shape[0]))
#exit()



tol=1E-5
delta=1E-7	
nat_grad_guess=np.zeros_like(grad_osx)

# Fisher_osx_r   = Fisher_osx   + delta**np.diag(np.diag(Fisher_osx))
# Fisher_linux_r = Fisher_linux + delta**np.diag(np.diag(Fisher_linux))

Fisher_osx_r   = Fisher_osx   + delta*np.eye(Fisher_osx.shape[0]) 
Fisher_linux_r = Fisher_linux + delta*np.eye(Fisher_linux.shape[0]) 

norm_osx_r=np.linalg.norm(Fisher_osx_r)
norm_linux_r=np.linalg.norm(Fisher_linux_r)


print(np.linalg.cond(Fisher_osx_r)  , np.linalg.cond(Fisher_osx_r/norm_osx_r))
print(np.linalg.cond(Fisher_linux_r), np.linalg.cond(Fisher_linux_r/norm_linux_r))
print()

nat_grad_osx, _	  = cg(Fisher_osx_r,   grad_osx,   x0=nat_grad_guess,  maxiter=1E4, atol=tol, tol=tol)
nat_grad_linux, _ = cg(Fisher_linux_r, grad_linux, x0=nat_grad_guess,  maxiter=1E4, atol=tol, tol=tol)


print('osx  ',grad_osx[0], Fisher_osx_r[-1,-1], nat_grad_guess[0], nat_grad_osx[0])
print('linux',grad_linux[0], Fisher_linux_r[-1,-1], nat_grad_guess[0], nat_grad_linux[0])

print(nat_grad_linux[0] - nat_grad_osx[0])


