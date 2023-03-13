"""  Created on 13/03/2023::
------------- _simuvals_class.py -------------

**Authors**: L. Mingarelli
"""
import pickle
import numpy as np

try:
    with open(f"{'/'.join(__file__.split('/')[:-1])}/res/simulvals.pickle", 'rb') as f:
        _SimulVals = pickle.load(f)
except:
    with open("./bindata/res/SimulVals.pickle", 'rb') as f:
        _SimulVals = pickle.load(f)



def interpolate_matrix(D, target_coord):
    """Weighted mean bilinear interpolation: https://en.wikipedia.org/wiki/Bilinear_interpolation"""

    available_x_crds, available_y_crds = zip(*D.keys())
    available_x_crds, available_y_crds = (np.array(sorted(set(available_x_crds))),
                                          np.array(sorted(set(available_y_crds)))
                                          )

    x, y = target_coord
    xid = np.argmax(available_x_crds >= x)
    yid = np.argmax(available_y_crds >= y)

    x1, x2 = available_x_crds[xid], available_x_crds[min(xid+1, len(available_x_crds)-1)]
    y1, y2 = available_y_crds[yid], available_y_crds[min(yid+1, len(available_y_crds)-1)]

    Q11 = tuple(sorted((x1, y1)))
    Q12 = tuple(sorted((x1, y2)))
    Q21 = tuple(sorted((x2, y1)))
    Q22 = tuple(sorted((x2, y2)))

    dx, dy = (x2-x1), (y2-y1)

    w11 = (x2-x) * (y2-y)
    w12 = (x2-x) * (y-y1)
    w21 = (x-x1) * (y2-y)
    w22 = (x-x1) * (y-y1)

    if dx==0:  # Just a linear interpolation (along y)
        w1, w2 = (y2 - y), (y - y1)
        D_interp = (w1 * D[Q11] + w2 * D[Q12]) / dy
    elif dy==0: # Just a linear interpolation (along x)
        w1, w2 = (x2-x), (x-x1)
        D_interp = (w1*D[Q11] + w2*D[Q21]) / dx
    else:
        D_interp = (w11*D[Q11] + w12*D[Q12] + w21*D[Q21] + w22*D[Q22]) / (dx * dy)

    return D_interp


class SimuValsClass():
    _SimulVals = _SimulVals

    def __getitem__(self, arg):
        if arg in self._SimulVals.keys():
            return self._SimulVals[arg]
        else:
            return interpolate_matrix(D=self._SimulVals, target_coord=arg)

    def keys(self):
        return self._SimulVals.keys()

    def values(self):
        return self._SimulVals.values()

    def items(self):
        return self._SimulVals.items()

    def __repr__(self):
        return self._SimulVals.__repr__()


SimulVals = SimuValsClass()