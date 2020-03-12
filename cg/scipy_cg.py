"""Iterative methods for solving linear systems"""

from __future__ import division, print_function, absolute_import

__all__ = ['cg',]

import warnings
import numpy as np

from . import _iterative

from scipy.sparse.linalg.interface import LinearOperator
from .utils import make_system
from scipy._lib._util import _aligned_zeros
from scipy._lib._threadsafety import non_reentrant

_type_conv = {'f':'s', 'd':'d', 'F':'c', 'D':'z'}





def _stoptest(residual, atol):
    """
    Successful termination condition for the solvers.
    """
    resid = np.linalg.norm(residual)
    if resid <= atol:
        return resid, 1
    else:
        return resid, 0


def _get_atol(tol, atol, bnrm2, get_residual, routine_name):
    """
    Parse arguments for absolute tolerance in termination condition.
    Parameters
    ----------
    tol, atol : object
        The arguments passed into the solver routine by user.
    bnrm2 : float
        2-norm of the rhs vector.
    get_residual : callable
        Callable ``get_residual()`` that returns the initial value of
        the residual.
    routine_name : str
        Name of the routine.
    """

    if atol is None:
        warnings.warn("scipy.sparse.linalg.{name} called without specifying `atol`. "
                      "The default value will be changed in a future release. "
                      "For compatibility, specify a value for `atol` explicitly, e.g., "
                      "``{name}(..., atol=0)``, or to retain the old behavior "
                      "``{name}(..., atol='legacy')``".format(name=routine_name),
                      category=DeprecationWarning, stacklevel=4)
        atol = 'legacy'

    tol = float(tol)

    if atol == 'legacy':
        # emulate old legacy behavior
        resid = get_residual()
        if resid <= tol:
            return 'exit'
        if bnrm2 == 0:
            return tol
        else:
            return tol * float(bnrm2)
    else:
        return max(float(atol), tol * float(bnrm2))




# def set_docstring(header, Ainfo, footer='', atol_default='0'):
#     def combine(fn):
#         fn.__doc__ = '\n'.join((header, common_doc1,
#                                 '    ' + Ainfo.replace('\n', '\n    '),
#                                 common_doc2, footer))
#         return fn
#     return combine



# @set_docstring('Use Conjugate Gradient iteration to solve ``Ax = b``.',
#                'The real or complex N-by-N matrix of the linear system.\n'
#                '``A`` must represent a hermitian, positive definite matrix.\n'
#                'Alternatively, ``A`` can be a linear operator which can\n'
#                'produce ``Ax`` using, e.g.,\n'
#                '``scipy.sparse.linalg.LinearOperator``.')
@non_reentrant()
def cg(A, b, x0=None, tol=1e-5, maxiter=None, M=None, callback=None, atol=None):
    A, M, x, b, postprocess = make_system(A, M, x0, b)

    n = len(b)
    if maxiter is None:
        maxiter = n*10

    matvec = A.matvec
    psolve = M.matvec
    ltr = _type_conv[x.dtype.char]
    revcom = getattr(_iterative, ltr + 'cgrevcom')

    get_residual = lambda: np.linalg.norm(matvec(x) - b)
    atol = _get_atol(tol, atol, np.linalg.norm(b), get_residual, 'cg')
    if atol == 'exit':
        return postprocess(x), 0

    resid = atol
    ndx1 = 1
    ndx2 = -1
    # Use _aligned_zeros to work around a f2py bug in Numpy 1.9.1
    work = _aligned_zeros(4*n,dtype=x.dtype)
    ijob = 1
    info = 0
    ftflag = True
    iter_ = maxiter
    while True:
        olditer = iter_
        x, iter_, resid, info, ndx1, ndx2, sclr1, sclr2, ijob = \
           revcom(b, x, work, iter_, resid, info, ndx1, ndx2, ijob)
        if callback is not None and iter_ > olditer:
            callback(x)
        slice1 = slice(ndx1-1, ndx1-1+n)
        slice2 = slice(ndx2-1, ndx2-1+n)
        if (ijob == -1):
            if callback is not None:
                callback(x)
            break
        elif (ijob == 1):
            work[slice2] *= sclr2
            work[slice2] += sclr1*matvec(work[slice1])
        elif (ijob == 2):
            work[slice1] = psolve(work[slice2])
        elif (ijob == 3):
            work[slice2] *= sclr2
            work[slice2] += sclr1*matvec(x)
        elif (ijob == 4):
            if ftflag:
                info = -1
                ftflag = False
            resid, info = _stoptest(work[slice1], atol)
            if info == 1 and iter_ > 1:
                # recompute residual and recheck, to avoid
                # accumulating rounding error
                work[slice1] = b - matvec(x)
                resid, info = _stoptest(work[slice1], atol)
        ijob = 2

    if info > 0 and iter_ == maxiter and not (resid <= atol):
        # info isn't set appropriately otherwise
        info = iter_

    return postprocess(x), info, iter_, resid

