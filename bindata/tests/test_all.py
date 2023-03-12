"""  Created on 23/07/2022::
------------- test_all.py -------------

**Authors**: L. Mingarelli
"""
import numpy as np

from bindata.check_commonprob import check_commonprob
from bindata import (commonprob2sigma,
                     condprob,
                     bincorr2commonprob,
                     ra2ba,
                     rmvbin
                     )
from bindata.simul_commonprob import simul_commonprob


class Tests:
    def test_check_commonprob(self):
        flag, msg = check_commonprob([[0.5, 0.4],
                                      [0.4, 0.8]])
        assert flag

        flag, msg = check_commonprob([[0.5, 0.25],
                                      [0.25, 0.8]])
        assert not flag
        assert msg[0].startswith('Error in Element (0, 1): Admissible values are')

        flag, msg = check_commonprob([[0.5, 0, 0],
                                      [0, 0.5, 0],
                                      [0, 0, 0.5]])
        assert not flag
        assert msg[0].startswith('The sum of the common probabilities of 0, 1, 2')

    def test_bincorr2commonprob(self):
        margprob = np.array([0.3, 0.9])
        bincorr = np.eye(len(margprob))
        commonprob = bincorr2commonprob(margprob, bincorr)
        assert np.isclose(commonprob, np.array([[0.3 , 0.27],
                                                [0.27, 0.9 ]])).all()

    def test_commonprob2sigma(self):
        m = [[1/2, 1/5, 1/6],
             [1/5, 1/2, 1/6],
             [1/6, 1/6, 1/2]]
        Σ = commonprob2sigma(commonprob=m)
        assert (Σ == np.array([[ 1.        , -0.3122489280072217, -0.5038009378508685],
                               [-0.3122489280072217,  1.        , -0.5038009378508685],
                               [-0.5038009378508685, -0.5038009378508685,  1.        ]])).all()

    def test_condprob(self):
        x = np.array([[0,1], [1,1], [0,0], [0,0], [1,0], [1,1]])
        expected_res = np.array([[1, 2/3],
                                 [2/3, 1]])
        assert np.isclose(condprob(x), expected_res).all()

        np.random.seed(0)
        x = np.random.binomial(1, 0.5, (1_000_000, 2))
        expected_res = np.array([[1, 0.5013397165515436],
                                 [ 0.5011774904572338, 1]])
        assert np.isclose(condprob(x), expected_res).all()

    def test_ra2ba(self):
        np.random.seed(0)
        x = np.random.normal(0,1, (2, 5))
        expected_res = np.array([[ True,  True,  True,  True,  True],
                                [False,  True, False, False,  True]])
        assert (ra2ba(x)==expected_res).all()

    def test_rmvbin(self):
        corr = np.array([[1., -0.25, -0.0625],
                         [-0.25, 1., 0.25],
                         [-0.0625, 0.250, 1.]])
        commonprob = bincorr2commonprob(margprob=[0.2, 0.5, 0.8], bincorr=corr)

        sample = rmvbin(margprob=np.diag(commonprob), commonprob=commonprob, N=10_000_000)
        realised_corr = np.corrcoef(sample, rowvar=False)
        np.abs(corr - realised_corr)
        assert np.isclose(corr, realised_corr, rtol=1e-4, atol=2e-3).all()

    def test_rmvbin2(self):
        N = 10_000_000
        # Uncorrelated columns:
        margprob = [0.3, 0.9]
        X = rmvbin(N=N, margprob=margprob)
        assert np.isclose(X.mean(0), margprob, rtol=1e-4, atol=2e-3).all()
        assert np.isclose(np.corrcoef(X, rowvar=False), np.eye(2), rtol=1e-4, atol=2e-3).all()

        # Correlated columns
        m = [[1/2, 1/5, 1/6],
             [1/5, 1/2, 1/6],
             [1/6, 1/6, 1/2]]
        X = rmvbin(N=N, commonprob=m)
        assert np.isclose(X.mean(0), np.diagonal(m), rtol=1e-4, atol=2e-3).all()
        assert np.isclose(np.corrcoef(X, rowvar=False),
                          np.array([[ 1.        , -0.20189503, -0.33612065],
                                    [-0.20189503,  1.        , -0.33655543],
                                    [-0.33612065, -0.33655543,  1.        ]]),
                          rtol=1e-4, atol=2e-3).all()

        # Same as the example above, but faster if the same probabilities are
        # used repeatedly
        sigma = commonprob2sigma(m)
        X = rmvbin(N=N, margprob=np.diagonal(m), sigma=sigma)
        assert np.isclose(X.mean(0), np.diagonal(m), rtol=1e-4, atol=2e-3).all()
        assert np.isclose(np.corrcoef(X, rowvar=False),
                          np.array([[ 1.        , -0.20189503, -0.33612065],
                                    [-0.20189503,  1.        , -0.33655543],
                                    [-0.33612065, -0.33655543,  1.        ]]),
                          rtol=1e-4, atol=2e-3).all()

    def test_rmvbin3(self):
        N = 10_000
        p_d = 0.1
        corr = 0.1
        a, b = rmvbin(N=N, margprob=[p_d, p_d],
                      bincorr=[[1, corr],
                               [corr, 1]]).T

    def test_simul_commonprob(self):
        margprob = np.arange(0, 1.5, 0.5)
        corr = np.arange(-1, 1.5, 0.5)
        np.random.seed(0)
        Z = simul_commonprob(margprob=margprob,
                             corr=corr,
                             method="monte carlo", n1=10**4)
        expected_Z = {(0.0, 0.0): np.array([[-1. , -0.5,  0. ,  0.5,  1. ],
                                            [ 0. ,  0. ,  0. ,  0. ,  0. ]]),
                      (0.0, 0.5): np.array([[-1. , -0.5,  0. ,  0.5,  1. ],
                                            [ 0. ,  0. ,  0. ,  0. ,  0. ]]),
                      (0.0, 1.0): np.array([[-1. , -0.5,  0. ,  0.5,  1. ],
                                             [ 0. ,  0. ,  0. ,  0. ,  0. ]]),
                      (0.5, 0.5): np.array([[-1.     , -0.5    ,  0.     ,  0.5    ,  1.     ],
                                            [ 0.     ,  0.16769,  0.25   ,  0.3354 ,  0.5    ]]),
                      (0.5, 1.0): np.array([[-1. , -0.5,  0. ,  0.5,  1. ],
                                            [ 0.5,  0.5,  0.5,  0.5,  0.5]]),
                      (1.0, 1.0): np.array([[-1. , -0.5,  0. ,  0.5,  1. ],
                                            [ 1. ,  1. ,  1. ,  1. ,  1. ]])
                      }
        for c, eZ in expected_Z.items():
            assert np.isclose(Z[c], eZ).all().all()



