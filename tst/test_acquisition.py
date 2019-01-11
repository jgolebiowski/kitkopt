import numpy as np
import unittest

from kitkopt.gaussian_process import GaussianProcessRegression
from kitkopt.hyper_parameter import HyperParameter
from kitkopt.kernels import rbf
from kitkopt.hypergrid import not_in_array, get_hypergrid
from kitkopt.bayesian_optimizer import propose_points, minimize_function
from kitkopt.acquisition import _get_single_point_Thompson, _get_new_unique_points_Thompson, _get_new_unique_point_UCB
from kitkopt.utilities import debugtool, OptimizerError


class AcquisitionrTest(unittest.TestCase):
    def test_get_new_unique_point_Thompson(self):
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
            self.assertTrue(not_in_array(_get_single_point_Thompson(previous_points, grid, gp, 100), previous_points))

        # Check error
        gp.initialize(previous_points)
        with self.assertRaises(OptimizerError):
            _get_single_point_Thompson(previous_points, previous_points, gp)

    def test_get_new_unique_points_Thompson(self):
        hyperparam_config = [
            HyperParameter(0, 3, 1),
            HyperParameter(0, 5, 2)
        ]
        grid = get_hypergrid(hyperparam_config)
        num_points = 5

        gp_settings = dict(
            kernel=rbf,
            kernel_params=(0.02, 0.25),
            noise=1e-6
        )
        gp = GaussianProcessRegression(gp_settings["kernel"],
                                       *gp_settings["kernel_params"],
                                       noise=gp_settings["noise"])
        gp.initialize(grid)
        new_points = _get_new_unique_points_Thompson(grid, gp, num_points)

        for i in range(len(new_points)):
            for j in range(i + 1, len(new_points)):
                self.assertFalse(np.array_equal(new_points[i], new_points[j]))

    def test_get_new_unique_points_UCB(self):
        hyperparam_config = [
            HyperParameter(0, 3, 1),
            HyperParameter(0, 5, 2)
        ]
        grid = get_hypergrid(hyperparam_config)
        known_points = np.array([
            [0, 0]])
        values = np.array([
            -2])

        gp_settings = dict(
            kernel=rbf,
            kernel_params=(0.02, 0.25),
            noise=1e-6
        )
        gp = GaussianProcessRegression(gp_settings["kernel"],
                                       *gp_settings["kernel_params"],
                                       noise=gp_settings["noise"])
        gp.fit(known_points, values, grid)
        ucb_tradeoff = 0.5
        new_points = _get_new_unique_point_UCB(grid, gp, ucb_tradeoff)
        target = np.array([[0., 0.]])
        np.testing.assert_allclose(new_points, target, atol=1e-5)


if (__name__ == "__main__"):
    unittest.main()
