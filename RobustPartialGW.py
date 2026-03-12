import numpy as np
import matplotlib.pyplot as plt
import ot
import random

def density_based_weighting(dist_mat,radius = 1):

    weights_init = np.zeros(len(dist_mat))

    for i in range(len(dist_mat)):
        weights_init[i] = sum(dist_mat[i,:]<radius)

    weights = weights_init/np.sum(weights_init)

    return weights

def robust_mGW(D1,D2,p1,p2,k=1,t=1e-2, max_iter = 100):

    """
    Find parameter m in [0,1] such that
    ot.gromov.partial_gromov_wasserstein2(D1, D2, p1, p2, mass = m) is approximately zero.

    Parameters
    ----------
    D1, D2, p1, p2 : np.ndarray. Two kernel matrices and probability distributions
    t : float, optional
        Tolerance for the zero test. The search stops when |f(...)| <= t.
    max_iter : int, optional
        Maximum number of iterations to perform.

    Returns
    -------
    m : float
        The parameter m that makes f(...) close to zero, within given tolerance.
    dist : float, robust mGW distance
    log: dict, data from the optimal partial GW distance computation
    """

    low, high = 0.01, 0.99
    dist_low, log = ot.gromov.partial_gromov_wasserstein2(D1, D2, p1, p2, m = 1-low, log = True)
    f_low = dist_low - k*low
    dist_high, log = ot.gromov.partial_gromov_wasserstein2(D1, D2, p1, p2, m = 1-high, log = True)
    f_high = dist_high - k*high

    # Quick checks if endpoints are already within tolerance
    if abs(f_low) <= t:
        return low, dist_low, log
    if abs(f_high) <= t:
        return high, dist_high, log

    # If f_low and f_high are not of opposite signs, the root might still exist,
    # but the bisection method typically relies on a sign change.
    # If you know a root exists, you can proceed anyway or raise an exception.
    # For robustness, let's just proceed under assumption that a root is in [0,1].

    for i in range(max_iter):
        mid = 0.5 * (low + high)
        dist_mid, log = ot.gromov.partial_gromov_wasserstein2(D1, D2, p1, p2, m = 1-mid, log = True)
        f_mid = dist_mid - k*mid

        # Check if mid is close enough to zero
        if abs(f_mid) <= t:
            return mid, dist_mid, log

        # Decide which half to keep
        # If f_low and f_mid have opposite signs, root is in [low, mid]
        if f_low * f_mid < 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    # If we reach here, we didn't find a value within tolerance, return best guess
    return 1-mid, dist_mid, log
