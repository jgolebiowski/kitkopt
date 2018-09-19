import numpy as np
import scipy.stats
from typing import Callable, Iterable, Any

SINGULARITY_PREVENTION = 1e-10


class GaussianProcessRegression(object):
    noise: float
    kernel: Callable[[np.ndarray, np.ndarray, Any], np.ndarray]
    kernel_args: Iterable

    mu: np.ndarray
    cov: np.ndarray
    sigmas: np.ndarray

    def __init__(self, kernel: Callable[[np.ndarray, np.ndarray, Any], np.ndarray],
                 *kernel_args: Any, noise: float = 1e-8):
        """
        Standard Gaussian Process Regression

        :param kernel: Kernel used ot generate the Gram matrixes for GP regression
        :param kernel_args: Parameters for the kernel
        :param noise: optional sigma^2 noise for of the measurements
        """
        self.noise = noise
        self.kernel = kernel
        self.kernel_args = kernel_args

        self.mu = None
        self.cov = None
        self.sigmas = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_predict: np.ndarray) -> None:
        """
        Train the model on data

        :param x_train: Train data in dims (nexamples, nfeatures)
        :param y_train: Labels in dims od (nexamples, 1)
        :param x_predict: Prediction points in dims (nexamples, nfeatures)
        """
        npoints_train = x_train.shape[0]

        # ------ Prepare Gram matrices
        K = self.kernel(x_train, x_train, *self.kernel_args) + np.eye(npoints_train) * self.noise
        Ks = self.kernel(x_train, x_predict, *self.kernel_args)
        Kss = self.kernel(x_predict, x_predict, *self.kernel_args)

        # Fit the GP
        alpha = np.linalg.solve(K, y_train)
        v = Ks.T @ np.linalg.solve(K, Ks)

        self.mu = Ks.T @ alpha
        self.cov = Kss - v
        self.sigmas = np.sqrt(np.diag(self.cov)).reshape(-1, 1)

        self.cov += np.eye(self.cov.shape[0]) * SINGULARITY_PREVENTION

    def initialize(self, x_predict: np.ndarray) -> None:
        """
        Initialize the model with no data

        :param x_predict: Prediction points in dims (nexamples, nfeatures)
        """

        self.mu = np.zeros((x_predict.shape[0], 1))
        self.cov = self.kernel(x_predict, x_predict, *self.kernel_args)
        self.sigmas = np.sqrt(np.diag(self.cov)).reshape(-1, 1)
        self.cov += np.eye(self.cov.shape[0]) * SINGULARITY_PREVENTION

    def predict(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Get predictions

        :return: mean value at each prediction point
        :return: uncertianty at each point
        :return: covariance matrix
        """
        return self.mu, self.sigmas, self.cov

    def sample(self) -> np.ndarray:
        """
        Get a sample from the Multi varied Gaussian on the points defined in

        :return: Sample from a multivariate Gaussian with dims (nexamples, 1)
        """
        if self.cov is None:
            raise RuntimeError("Must run self.predict() before sampling")

        return scipy.stats.multivariate_normal.rvs(self.mu.flatten(), self.cov)
