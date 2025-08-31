import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_pred-y_true,2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / np.size(y_pred)

def binary_cross_entrophy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred)+(1-y_true)*np.log(1-y_pred))

def binary_cross_entrophy_prime(y_true, y_pred):
    return ((1-y_true)/(1-y_pred) - y_true/y_pred) / np.size(y_true)