import numpy as np
import unittest

from bayesian_optimizer.gaussian_process import GaussianProcessRegression
from bayesian_optimizer.hyper_parameter import HyperParameter
from bayesian_optimizer.kernels import rbf
from bayesian_optimizer.hypergrid import not_in_array, get_hypergrid
from bayesian_optimizer.bayesian_optimizer import _get_new_unique_point, propose_points, minimize_function
from bayesian_optimizer.utilities import debugtool, OptimizerError


class BayesianOptimizerTest(unittest.TestCase):
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
                                                   max_iterations=100,
                                                   seed=123)
        np.testing.assert_allclose(best_point, np.array([0, 0]), atol=1e-5)
        np.testing.assert_allclose(best_value, np.array([0]), atol=1e-5)

    def test_optimizer(self):
        hyperparam_config = [
            HyperParameter(0, 5, 1),
            HyperParameter(0, 5, 1)
        ]
        tested_points = np.array([
            [0, 0],
            [0, 4],
            [2, 2],
            [4, 4],
            [4, 0],
            [1, 1]
        ], dtype=float)
        values = np.array([
            2,
            2,
            2,
            2,
            2,
            0
        ], dtype=float)
        gp_settings = dict(
            kernel=rbf,
            kernel_params=(0.1, 0.2),
            noise=1e-6
        )

        N = 200
        draws = np.empty((N, 2))
        for idx in range(N):
            draws[idx, :] = propose_points(tested_points, values, hyperparam_config, 1, gp_settings=gp_settings)

        empiricammean = np.mean(draws, axis=0)
        np.testing.assert_allclose(empiricammean, np.array([1, 1]), atol=2e-1)

    def test_propose_points(self):
        hyperparam_config = [
            HyperParameter(0, 3, 1),
            HyperParameter(0, 5, 2)
        ]
        tested_points = np.array([
            [0, 2],
            [2, 0],
            [1, 4],
        ], dtype=float)
        target = np.array([[2., 4.],
                           [0., 0.],
                           [3., 0.],
                           [1., 2.],
                           [3., 2.],
                           [1., 0.],
                           [0., 4.],
                           [3., 4.],
                           [2., 2.]])
        values = np.array([1, 2, 3], dtype=float)
        result = propose_points(tested_points, values, hyperparam_config, 9, seed=123)
        # print(repr(result))
        np.testing.assert_almost_equal(result, target, decimal=5)

        target = np.array([[2., 4.],
                           [0., 0.],
                           [3., 0.],
                           [1., 2.]])
        values = np.array([1, 2, 3], dtype=float)
        result = propose_points(tested_points, values, hyperparam_config, 4, seed=123)
        # print(repr(result))

        np.testing.assert_almost_equal(result, target, decimal=5)

        # Check error
        with self.assertRaises(OptimizerError):
            propose_points(tested_points, values, hyperparam_config, 20, seed=123)

    def test_get_new_unique_point(self):
        hyperparam_config = [
            HyperParameter(0, 3, 1),
            HyperParameter(0, 5, 2)
        ]
        grid = get_hypergrid(hyperparam_config)
        previous_points = np.array([
            [0, 4],
            [1, 0],
            [3, 2],
            [4, 2]
        ])

        gp_settings = dict(
            kernel=rbf,
            kernel_params=(0.02, 0.25),
            noise=1e-6
        )
        gp = GaussianProcessRegression(gp_settings["kernel"],
                                       *gp_settings["kernel_params"],
                                       noise=gp_settings["noise"])
        gp.initialize(grid)

        for idx in range(100):
            self.assertTrue(not_in_array(_get_new_unique_point(previous_points, grid, gp, 100), previous_points))

        # Check error
        gp.initialize(previous_points)
        with self.assertRaises(OptimizerError):
            _get_new_unique_point(previous_points, previous_points, gp)
