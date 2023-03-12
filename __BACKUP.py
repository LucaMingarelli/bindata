"""  Created on 12/03/2023::
------------- __BACKUP.py -------------

**Authors**: L. Mingarelli
"""


def __commonprob2sigma(commonprob, simulvals=None):

    commonprob = np.array(commonprob)

    if simulvals is None:
        simulvals = SimulVals

    margprob = np.diag(commonprob)

    # Checking for commonprob's admissibility
    _flag, msg = check_commonprob(commonprob)
    if not _flag:
        raise ValueError(f"Matrix commonprob not admissible.\n{msg}")

    sigma = np.eye(commonprob.shape[0])

    for m in range(commonprob.shape[1]-1):
        for n in range(m+1, commonprob.shape[0]):
            x = np.vstack((margprob[m], margprob[n], np.array(simulvals)))
            y = RegularGridInterpolator((range(margprob.size), range(margprob.size), range(simulvals.size)), x.T)
            f = lambda z: y((m,n,z))
            sigma[m,n] = sigma[n,m] = f(commonprob[m,n])

    if np.any(np.isnan(sigma)):
        raise ValueError("Extrapolation occurred ... margprob and commonprob not compatible?")
    if np.min(eigvals(sigma)) < 0:
        print("Warning: Resulting covariance matrix is not positive definite.")
        print(f"         Smallest eigenvalue equals {np.min(eigvals(sigma))}.")
        print("         Please check whether the results are still useful.")

    return sigma
