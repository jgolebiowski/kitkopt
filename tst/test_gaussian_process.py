import numpy as np
import unittest
from blackop.kernels import rbf
from blackop.gaussian_process import GaussianProcessRegression


class GaussianProcessTest(unittest.TestCase):
    train = np.array([[0.54172578, 0.27191608, 0.94019346],
                      [0.38345828, 0.6140292, 0.60123052],
                      [0.91552568, 0.29013107, 0.46425194],
                      [0.23228642, 0.30616566, 0.94467399],
                      [0.06702568, 0.08163386, 0.52261632]])
    values = np.array([[0.58296797],
                       [0.87302992],
                       [0.43924792],
                       [0.77782836],
                       [0.15247357]])
    test = np.array([[0.96380476, 0.83472723, 0.13314283],
                     [0.99353882, 0.07326699, 0.91934838],
                     [0.77286147, 0.97525652, 0.57766557],
                     [0.30939199, 0.12165684, 0.75722879]])

    def test_fit(self):
        kp = (1.0, 1.0)
        noise = 1e-6
        gp = GaussianProcessRegression(rbf, *kp, noise=noise)
        gp.fit(self.train, self.values, self.test)
        mu, sigma, cov = gp.predict()

        target_mu = np.array([[0.77044],
                              [0.17201],
                              [0.9521],
                              [0.38181]])
        target_sigma = np.array([[0.39728],
                                 [0.21463],
                                 [0.32675],
                                 [0.0823]])
        target_cov = np.array([[0.15783, -0.01669, 0.08743, -0.00217],
                               [-0.01669, 0.04607, -0.00378, -0.00412],
                               [0.08743, -0.00378, 0.10677, -0.00557],
                               [-0.00217, -0.00412, -0.00557, 0.00677]])

        np.testing.assert_almost_equal(mu, target_mu, decimal=5)
        np.testing.assert_almost_equal(cov, target_cov, decimal=5)
        np.testing.assert_almost_equal(sigma, target_sigma, decimal=5)

    def test_init(self):
        kp = (1.0, 1.0)
        noise = 1e-6
        gp = GaussianProcessRegression(rbf, *kp, noise=noise)
        gp.initialize(self.test)
        mu, sigma, cov = gp.predict()

        target_mu = np.array([[0.0],
                              [0.0],
                              [0.0],
                              [0.0]])
        target_sigma = np.array([[1.0],
                                 [1.0],
                                 [1.0],
                                 [1.0]])
        target_cov = np.array([[1., 0.54913, 0.88082, 0.51525],
                               [0.54913, 1., 0.61292, 0.78009],
                               [0.88082, 0.61292, 1., 0.61395],
                               [0.51525, 0.78009, 0.61395, 1.]])

        np.testing.assert_almost_equal(mu, target_mu, decimal=5)
        np.testing.assert_almost_equal(cov, target_cov, decimal=5)
        np.testing.assert_almost_equal(sigma, target_sigma, decimal=5)

    def test_sample(self):
        kp = (1.0, 1.0)
        noise = 1e-6
        gp = GaussianProcessRegression(rbf, *kp, noise=noise)
        gp.fit(self.train, self.values, self.test)
        mu, sigma, cov = gp.predict()

        N = 10000
        draws = np.empty((N, 4))
        for idx in range(N):
            draws[idx, :] = gp.sample().T

        target_mu = np.mean(draws, axis=0)
        target_sigma = np.std(draws, axis=0)

        np.testing.assert_almost_equal(mu.ravel(), target_mu, decimal=2)
        np.testing.assert_almost_equal(sigma.ravel(), target_sigma, decimal=2)
