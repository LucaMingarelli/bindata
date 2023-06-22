"""  Created on 12/03/2023::
------------- _generate_simulvals.py -------------

**Authors**: L. Mingarelli
"""
import pickle
import numpy as np
from bindata.simul_commonprob import simul_commonprob

margprob = [0]*2 + list(np.linspace(0, 1, 21)) + [1]*2
corr     = [-1]*2 + list(np.linspace(-1, 1, 41)) + [1]*2

margprob[1:3] = 0.001, 0.01
margprob[-3:-1] = 1-margprob[2], 1-margprob[1]

corr[1:3] = -0.999, -0.99
corr[-3:-1] = -corr[2], -corr[1]

Z = simul_commonprob(margprob=margprob,
                     corr=corr,
                     method="monte carlo",
                     n1=10 ** 4,
                     n2=100,
                     pbar=True)


with open('NEW_simulvals.pickle', 'wb') as f:
    pickle.dump(Z, f, protocol=pickle.HIGHEST_PROTOCOL)

