import numpy as np
import unittest

from bayesian_optimizer.gaussian_process import GaussianProcessRegression
from bayesian_optimizer.hyper_parameter import HyperParameter
from bayesian_optimizer.kernels import rbf
from bayesian_optimizer.random_optimizer import not_in_array, get_hypergrid, OptimizerError
from bayesian_optimizer.bayesian_optimizer import get_new_unique_point, propose_points
from bayesian_optimizer.utilities import debugtool


class BayesianOptimizerTest(unittest.TestCase):

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
            0,
            0,
            0,
            0,
            0,
            2
        ], dtype=float)
        gp_settings = dict(
            kernel=rbf,
            kernel_params=(0.1, 0.25),
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
        target = np.array([[3., 0.],
                           [1., 0.],
                           [3., 4.],
                           [0., 4.],
                           [2., 2.],
                           [2., 4.],
                           [3., 2.],
                           [0., 0.],
                           [1., 2.]])
        values = np.array([1, 2, 3], dtype=float)
        result = propose_points(tested_points, values, hyperparam_config, 9, 123)
        # print(repr(result))
        np.testing.assert_almost_equal(result, target, decimal=5)

        target = np.array([[3., 0.],
                           [1., 0.],
                           [3., 4.],
                           [0., 4.]])
        values = np.array([1, 2, 3], dtype=float)
        result = propose_points(tested_points, values, hyperparam_config, 4, 123)
        # print(repr(result))
        np.testing.assert_almost_equal(result, target, decimal=5)

        # Check error
        with self.assertRaises(OptimizerError):
            propose_points(tested_points, values, hyperparam_config, 20, 123)

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
            self.assertTrue(not_in_array(get_new_unique_point(previous_points, grid, gp, 100), previous_points))

        # Check error
        gp.initialize(previous_points)
        with self.assertRaises(OptimizerError):
            get_new_unique_point(previous_points, previous_points, gp)