import numpy as np


def bb_hw(a):
    return np.array([a[1], a[0], a[3]-a[1]+1, a[2]-a[0]+1])

