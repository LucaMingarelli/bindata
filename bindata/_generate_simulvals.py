"""  Created on 12/03/2023::
------------- _generate_simulvals.py -------------

**Authors**: L. Mingarelli
"""
import numpy as np
from bindata.simul_commonprob import simul_commonprob


margprob = np.linspace(0, 1, 201)
corr = np.linspace(-1, 1, 201)


Z = simul_commonprob(margprob=margprob,
                     corr=corr,
                     method="monte carlo",
                     n1=10 ** 4,
                     n2=100,
                     pbar=True)