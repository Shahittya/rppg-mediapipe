import numpy as np

def mae(pred, gt):
    return np.mean(np.abs(pred - gt))

def rmse(pred, gt):
    return np.sqrt(np.mean((pred - gt) ** 2))