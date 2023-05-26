"""
Modified from scipy stats

"""

from cpython cimport bool
from libc cimport math
cimport cython
cimport numpy as np
from numpy.math cimport PI
from numpy cimport ndarray, int64_t, float64_t, intp_t
import warnings
import numpy as np
import scipy.stats, scipy.special
#cimport scipy.special.cython_special as cs
np.import_array()

from scipy.linalg import solve_triangular

ctypedef fused real:
    float
    double
    long double


@cython.wraparound(False)
@cython.boundscheck(False)
def gaussian_aw_kernel_estimate(points, values, xi, precision, inv_gamma,dtype, real _=0):
    """
    def gaussian_kernel_estimate(points, real[:, :] values, xi, precision)
    Evaluate a multivariate Gaussian kernel estimate.
    Parameters
    ----------
    points : array_like with shape (n, d)
        Data points to estimate from in d dimensions.
    values : real[:, :] with shape (n, p)
        Multivariate values associated with the data points.
    xi : array_like with shape (m, d)
        Coordinates to evaluate the estimate at in d dimensions.
    precision : array_like with shape (d, d)
        Precision matrix for the Gaussian kernel.
    Returns
    -------
    estimate : double[:, :] with shape (m, p)
        Multivariate Gaussian kernel estimate evaluated at the input coordinates.
    """
    cdef:
        real[:, :] points_, xi_, values_, estimate, whitening
        real[:] inv_gamma_
        int i, j, k
        int n, d, m, p
        real arg, residual, norm

    n = points.shape[0]
    d = points.shape[1]
    m = xi.shape[0]
    p = values.shape[1]

    if xi.shape[1] != d:
        raise ValueError("points and xi must have same trailing dim")
    if precision.shape[0] != d or precision.shape[1] != d:
        raise ValueError("precision matrix must match data dims")

    # Rescale the data
    whitening = np.linalg.cholesky(precision).astype(dtype, copy=False)
    points_ = np.dot(points, whitening).astype(dtype, copy=False)
    xi_ = np.dot(xi, whitening).astype(dtype, copy=False)
    values_ = values.astype(dtype, copy=False)

    # Evaluate the normalisation
    #norm = math.pow((2 * PI) ,(- d / 2))  # This causes trouble!!
    norm = math.pow((2 * PI) ,(- d / 2.))
    for i in range(d):
        norm *= whitening[i, i]
    
    # Create the result array and evaluate the weighted sum
    estimate = np.zeros((m, p), dtype)
    inv_gamma_ = inv_gamma.astype(dtype, copy=False)    
    
    for i in range(n):
        for j in range(m):
            arg = 0
            for k in range(d):
                residual = (points_[i, k] - xi_[j, k])
                arg += residual * residual

            arg *= inv_gamma_[i]
            
            arg = math.exp(-arg / 2.) * norm * inv_gamma_[i]
          
            
            for k in range(p):
                estimate[j, k] += values_[i, k] * arg

    return np.asarray(estimate)