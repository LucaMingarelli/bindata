# bindata <img src="https://raw.githubusercontent.com/LucaMingarelli/bindata/master/bindata/res/binary.png"  width="80">

[![CircleCI](https://circleci.com/gh/LucaMingarelli/bindata.svg?style=svg&circle-token=cd9c300380d25c24c66cd6637693cc50a7e00248)](https://app.circleci.com/pipelines/github/LucaMingarelli/bindata)
[![version](https://img.shields.io/badge/version-0.9.5-success.svg)](#)
[![PyPI Latest Release](https://img.shields.io/pypi/v/bindata.svg)](https://pypi.org/project/bindata/)
[![License](https://img.shields.io/pypi/l/bindata.svg)](https://github.com/LucaMingarelli/bindata/blob/master/LICENSE.txt)

[//]: # ([![Downloads]&#40;https://static.pepy.tech/personalized-badge/bindata?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads&#41;]&#40;https://pepy.tech/project/bindata&#41;)



A python replication of the omonymous R library 
[`bindata`](https://cran.r-project.org/web/packages/bindata/bindata.pdf),
 based on the paper 
*"Generation of correlated artificial binary data."*, 
by Friedrich Leisch, Andreas Weingessel, and Kurt Hornik.

The library fully replicates the existing R-package 
with the following functions:
* `bincorr2commonprob`
* `check_commonprob` (`check.commonprob` in R)
* `commonprob2sigma`
* `condprob`
* `ra2ba`
* `rmvbin`
* `simul_commonprob` (`simul.commonprob` in R)

Precomputed (via Monte Carlo simulations) `SimulVals` are also available.

## Installation
**bindata** can be installed with pip as:

```pip install bindata```

## How to


### Generate *uncorrelated* variates
```python
import bindata as bnd

margprob = [0.3, 0.9]

X = bnd.rmvbin(N=100_000, margprob=margprob)
```

Now let's verify the sample marginals and correlations:

```python
import numpy as np

print(X.mean(0))
print(np.corrcoef(X, rowvar=False))
```

```
[0.30102 0.9009 ]
[[ 1.         -0.00101357]
 [-0.00101357  1.        ]]
```

### Generate *correlated* variates

#### From a correlation matrix
```python
corr = np.array([[1., -0.25, -0.0625],
                 [-0.25,   1.,  0.25],
                 [-0.0625, 0.25, 1.]])
commonprob = bnd.bincorr2commonprob(margprob=[0.2, 0.5, 0.8], 
                                        bincorr=corr)

X = bnd.rmvbin(margprob=np.diag(commonprob), 
                   commonprob=commonprob, N=100_000)
print(X.mean(0))
print(np.corrcoef(X, rowvar=False))
```

```
[0.1996  0.50148 0.80076]
[[ 1.         -0.25552    -0.05713501]
 [-0.25552     1.          0.24412401]
 [-0.05713501  0.24412401  1.        ]]
```

#### From a joint probability matrix

```python
commonprob = [[1/2, 1/5, 1/6],
              [1/5, 1/2, 1/6],
              [1/6, 1/6, 1/2]]
X = bnd.rmvbin(N=100_000, commonprob=commonprob)

print(X.mean(0))
print(np.corrcoef(X, rowvar=False))
```

```
[0.50076 0.50289 0.49718]
[[ 1.         -0.20195239 -0.33343712]
 [-0.20195239  1.         -0.34203855]
 [-0.33343712 -0.34203855  1.        ]]
```

For a more comprehensive documentation please consult 
the [documentation](https://cran.r-project.org/web/packages/bindata/bindata.pdf).

## Acknowledgements

* *"Generation of correlated artificial binary data."*, 
by Friedrich Leisch, Andreas Weingessel, and Kurt Hornik.

* <a href="https://www.flaticon.com/free-icons/code" title="code icons">Icon created by Freepik - Flaticon</a>

## Author

Luca Mingarelli, 2022




