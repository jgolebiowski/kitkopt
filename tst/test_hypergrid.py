import numpy as np
import unittest

from blackop.hyper_parameter import HyperParameter
from blackop.hypergrid import not_in_array, get_hypergrid, prune_hypergrid
from blackop.utilities import debugtool, OptimizerError


class HypergridTest(unittest.TestCase):
    def test_not_in_array(self):
        vector = np.array([0, 1, 2], dtype=float)
        array = np.arange(3 * 3, dtype=float).reshape((3, 3))
        self.assertFalse(not_in_array(vector, array))

        vector = np.array([4, 3, 2], dtype=float)
        array = np.arange(3 * 3, dtype=float).reshape((3, 3))
        self.assertTrue(not_in_array(vector, array))

        vector = np.array([2, 3, 4])
        array = np.arange(2, 11, dtype=float).reshape((3, 3))
        self.assertFalse(not_in_array(vector, array))

    def test_get_hypergrid(self):
        hyperparam_config = [
            HyperParameter(0, 4, 1),
            HyperParameter(0, 5, 2)
        ]
        grid = get_hypergrid(hyperparam_config)
        target = np.array([[0., 0.],
                           [0., 2.],
                           [0., 4.],
                           [1., 0.],
                           [1., 2.],
                           [1., 4.],
                           [2., 0.],
                           [2., 2.],
                           [2., 4.],
                           [3., 0.],
                           [3., 2.],
                           [3., 4.],
                           [4., 0.],
                           [4., 2.],
                           [4., 4.]])
        np.testing.assert_almost_equal(grid, target, decimal=5)

    def test_prune_hypergrid(self):
        hyperparam_config = [
            HyperParameter(0, 4, 1),
            HyperParameter(0, 5, 2)
        ]
        tested_points = np.array([
            [0, 4.0],
            [1, 0],
            [3, 2],
            [4, 2]
        ])
        fullgrid = get_hypergrid(hyperparam_config)
        grid = prune_hypergrid(fullgrid, tested_points=tested_points)
        target = np.array([[0., 0.],
                           [0., 2.],
                           [1., 2.],
                           [1., 4.],
                           [2., 0.],
                           [2., 2.],
                           [2., 4.],
                           [3., 0.],
                           [3., 4.],
                           [4., 0.],
                           [4., 4.]])
        np.testing.assert_almost_equal(grid, target, decimal=5)
