"""Module responsible for the kernels"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

def poly_kernel(x: np.ndarray, y: np.ndarray, degree: int=1, gamma: float=1.0, coef0: float=0.0):
    """
    Compute the polynomial kernel between x and y::
        K(x, y) = (gamma * <x, y> + coef0)^degree

    :param x: ndarray of shape (n_samples_1, n_features)
    :param y: ndarray of shape (n_samples_2, n_features)
    :param degree: see formula
    :param gamma: see formula
    :param coef0: see formula
    :return: Gram matrix : array of shape (n_samples_1, n_samples_2)
    """

    K = np.dot(x, y.T)
    K *= gamma
    K += coef0
    K **= degree
    return K


def rbf(a: np.ndarray, b: np.ndarray, length: float, sigma_f: float):
    """
    Radial basis kernel: Gaussian distance kernel

    :param a: design matrix of shape (nexamples, nfeatures)
    :param b: design matrix of shape (nexamples, nfeatures)
    :param length: Kernel parameter: vertical length
    :param sigma_f: Kernel parameter: horizontal length
    :return: Gram Matrix of shape (nexamples, nexamples)
    """
    sqdist = np.sum(a**2, 1).reshape(-1, 1) + np.sum(b**2, 1) - 2 * np.dot(a, b.T)
    return np.exp(-0.5 * (1 / length) * sqdist) * sigma_f
