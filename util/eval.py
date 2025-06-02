import numpy as np
import torch
from sklearn.metrics.cluster import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

###

def neighbor_precision(n, y): # torch tensor required
    total_precision = 0

    for i in range(n.size(0)):
        precision = torch.sum(y[n[i]] == y[i]) / n[i].size(0) * 100

        total_precision = total_precision + precision

    precision = total_precision / n.size(0)

    return precision

###

def clustering_acc(p, y): # numpy array required
    d = max(np.max(p), np.max(y)) + 1
    w = np.zeros([d, d])

    for i in range(p.shape[0]):
        w[p[i], y[i]] = w[p[i], y[i]] + 1

    ind_row, ind_col = linear_sum_assignment(np.max(w) - w)

    acc = np.sum(w[ind_row, ind_col]) / p.shape[0] * 100

    return acc

###

def clustering_nmi(p, y): # numpy array required
    nmi = normalized_mutual_info_score(p, y) * 100

    return nmi

###
