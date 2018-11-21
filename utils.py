import numpy as np
from scipy.io import loadmat


def read_bach10_F0s(F0):
    f = np.round(loadmat(F0)['GTF0s'] - 21).astype(int)
    index = np.where(f >= 0)
    pianoroll = np.zeros((88, f.shape[1]))
    for i, frame in zip(index[0], index[1]):
        pianoroll[f[i, frame], frame] = 1
    return pianoroll


def multipitch_evaluation(estimation, truth, raw_value=False):
    TP = np.count_nonzero(truth)
    diff = truth - estimation
    FN = np.where(diff == 1)[0].shape[0]
    FP = np.where(diff < 0)[0].shape[0]
    TP -= FN

    if raw_value:
        return TP, FP, FN
    else:
        p = TP / (TP + FP)
        r = TP / (TP + FN)
        f1 = 2 * p * r / (p + r)
        return p, r, f1
