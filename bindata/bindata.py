"""  Created on 14/08/2021::
------------- bindata.py -------------

This is a script that implements functions for generating correlated binary data using
the method described in Leisch et al. (1998).

This software is a python implementation of the algorithm outlined in:

On the Generation of Correlated Artificial Binary Data
Friedrich Leisch, Andreas Weingessel, Kurt Hornik (1998)

**Authors**: L. Mingarelli
"""

import numpy as np
import pickle
from scipy.stats import multivariate_normal
from scipy import interpolate
from bindata.check_commonprob import check_commonprob, _check_against_simulvals
from scipy.stats import norm

from bindata._simuvals_class import SimulVals

def commonprob2sigma(commonprob, simulvals=None):
    """
    Computes a covariance matrix for a normal distribution
    which corresponds to a binary distribution with marginal probabilities given by
    diag(commonprob)
    and pairwise probabilities given by
    commonprob.
    For the simulations, the values of simulvals are used.
    If a non-valid covariance matrix is the result, the program stops with an error in the case of NA
    arguments and yields are warning message if the matrix is not positive definite
    Args:
        commonprob (numpy.array): The joint probabilities matrix.

    Returns:
        A covariance matrix is returned with the same dimensions as commonprob.
    """
    commonprob = np.array(commonprob)
    N = commonprob.shape[0]

    if simulvals is None:
        simulvals = SimulVals

    _check_against_simulvals(x=np.diagonal(commonprob), simulvals=simulvals)

    Σ = np.diag(np.ones(N))
    N_Σ = Σ.shape[0]

    for i, j in zip(*np.triu_indices(N_Σ, k=1)):
        r, jp = simulvals[tuple(sorted((round(commonprob[i, i], 10), round(commonprob[j, j], 10))))]
        func = interpolate.interp1d(jp, r)
        Σ[i, j] = Σ[j, i] = func(commonprob[i, j])
    return Σ

def bincorr2commonprob(margprob, bincorr):
    margprob, bincorr = np.array(margprob), np.array(bincorr)
    sqrtprod = np.sqrt(np.multiply.outer(margprob * (1 - margprob), margprob * (1 - margprob)))
    return bincorr * sqrtprod + np.multiply.outer(margprob, margprob)

def ra2ba(x):
    """
    Converts all values of the real valued array x to binary values by thresholding at 0.
    """
    return x>0

def condprob(x):
    """
    Returns a matrix containing the conditional probabilities P(xi = 1|xj = 1)
    where xi corresponds to the i-th column of x.
    Args:
        x: matrix of binary data with rows corresponding to cases and columns corresponding to variables.

    """
    x = np.array(x)
    nc = x.shape[1]
    mask = x!=0
    retval = np.zeros((nc, nc))
    for k in range(nc):
        retval[k, :] = np.mean(x[mask[:, k] != 0, :], axis=0)

    return retval

def rmvbin(margprob=None, sigma=None, bincorr=None,  commonprob=None, N=100,
           simulvals=None):
    """
    Creates correlated multivariate binary random variables by thresholding a normal distribution.
    The correlations of the components can be specified either as common probabilities,
    correlation matrix for the binary distribution,
    or covariance matrix of the normal distribution.
    Args:
        margprob:  marginal probabilities.
        commonprob:  matrix of probabilities whose components i and j are simultaneously 1.
        bincorr:  matrix of binary correlations.
        sigma:  covariance matrix for the normal distribution.
        N (int):  sample size to be generated.
        simulvals:  result from simul.commonprob, a default data array is automatically loaded if this argument is omitted.
    """
    if simulvals is None:
        simulvals = SimulVals

    if margprob is not None and (isinstance(margprob, float) or len(margprob) == 1):
        return np.random.binomial(1, margprob)

    if sigma is None:
        if commonprob is not None:
            _flag, msg = check_commonprob(commonprob=commonprob)
            if not _flag:
                raise ValueError(f"Joint probability (commonprob) is not admissable\n{msg}")
            if margprob is None:
                margprob = np.diagonal(commonprob)
            sigma = commonprob2sigma(commonprob, simulvals)
        elif bincorr is not None:
            commonprob = bincorr2commonprob(margprob, bincorr)
            sigma = commonprob2sigma(commonprob, simulvals)
        else:
            sigma = np.eye(len(margprob))
    elif (np.linalg.eig(sigma)[0]<0).any():
        raise ValueError("sigma is not positive definite.")

    # Sample from a multivariate normal
    µ, Σ = norm.ppf(margprob), sigma
    sample = multivariate_normal.rvs(µ, Σ, size=N)

    return ra2ba(sample)



