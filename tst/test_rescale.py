import numpy as np
import unittest
from bayesian_optimizer.rescale import rescale_vector, rescale_hypergrid


class RescaleTest(unittest.TestCase):


    def test_vector(self):
        source = np.array([1.0, 2, 3, 4])
        result = rescale_vector(source, 0, 4, 0, 1)
        target = np.array([0.25, 0.5, 0.75, 1])
        np.testing.assert_almost_equal(result, target, decimal=5)

        source = np.array([0.1, 0.2, 0.3, 0.4])
        result = rescale_vector(source, 0, 0.4, 0, 1)
        target = np.array([0.25, 0.5, 0.75, 1])
        np.testing.assert_almost_equal(result, target, decimal=5)

        source = np.array([0.1, 0.2, 0.3, 0.4])
        result = rescale_vector(source, 0, 0.4, -1, 1)
        target = np.array([-0.5, 0, 0.5, 1.0])
        np.testing.assert_almost_equal(result, target, decimal=5)

        source = np.array([2.0, 0, 0, 0])
        result = rescale_vector(source, source.min(), source.max(), -1, 1)
        target = np.array([1, -1, -1, -1])
        np.testing.assert_almost_equal(result, target, decimal=5)
