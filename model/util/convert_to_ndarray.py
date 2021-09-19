import numpy as np


def convert_to_ndarray(obj):
    if isinstance(obj, list):
        return np.asarray([convert_to_ndarray(o) for o in obj])
    else:
        return obj
