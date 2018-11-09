import numpy as np
import unittest

from blackop.hyper_parameter import HyperParameter
from blackop.random_optimizer import propose_points, \
    _get_new_unique_point, minimize_function
from blackop.hypergrid import not_in_array, get_hypergrid, prune_hypergrid
from blackop.utilities import debugtool, OptimizerError


class RandomOptimizerTest(unittest.TestCase):

    def test_minimize(self):
        def funct(x):
            return np.sum(np.square(x))

        hyperparam_config = [
            HyperParameter(-5, 5, 1),
            HyperParameter(-5, 5, 1)
        ]

        best_point, best_value = minimize_function(funct, hyperparam_config,
                                                   extra_function_args=(),
                                                   tolerance=1e-2,
                                                   seed=123)
        np.testing.assert_allclose(best_point, np.array([0, 0]), atol=1e-5)
        np.testing.assert_allclose(best_value, np.array([0]), atol=1e-5)

    def test_propose_points(self):
        hyperparam_config = [
            HyperParameter(0, 4, 1),
            HyperParameter(0, 5, 2)
        ]
        tested_points = np.array([
            [0, 4],
            [1, 0],
            [3, 2],
            [4, 2]
        ])
        target = np.array([[1., 2.],
                           [2., 4.],
                           [0., 2.],
                           [1., 4.],
                           [4., 4.],
                           [4., 0.],
                           [0., 0.],
                           [2., 0.],
                           [3., 0.],
                           [3., 4.],
                           [2., 2.]])
        result = propose_points(tested_points, None, hyperparam_config, 11, seed=123)
        # print(repr(result))
        np.testing.assert_almost_equal(result, target, decimal=5)

        target = np.array([[1., 2.],
                           [2., 4.],
                           [0., 2.],
                           [1., 4.],
                           [4., 4.],
                           [4., 0.]])
        result = propose_points(tested_points, None, hyperparam_config, 6, seed=123)
        # print(repr(result))
        np.testing.assert_almost_equal(result, target, decimal=5)

        # Check error
        with self.assertRaises(OptimizerError):
            propose_points(tested_points, None, hyperparam_config, 20, seed=123)

    def test_get_new_unique_point(self):
        hyperparam_config = [
            HyperParameter(0, 4, 1),
            HyperParameter(0, 5, 2)
        ]
        grid = get_hypergrid(hyperparam_config)
        previous_points = np.array([
            [0, 4],
            [1, 0],
            [3, 2],
            [4, 2]
        ])

        for idx in range(100):
            self.assertTrue(not_in_array(_get_new_unique_point(previous_points, grid, 100), previous_points))

        # Check error
        with self.assertRaises(OptimizerError):
            _get_new_unique_point(previous_points, previous_points)
