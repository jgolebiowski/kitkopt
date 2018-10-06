import numpy as np
import unittest

from bayesian_optimizer.hyper_parameter import HyperParameter
from bayesian_optimizer.random_optimizer import not_in_array, get_hypergrid, get_untested_hypergrid, propose_points, \
    OptimizerError, get_new_unique_point
from bayesian_optimizer.utilities import debugtool


class RandomOptimizerTest(unittest.TestCase):

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
        result = propose_points(tested_points, None, hyperparam_config, 11, 123)
        # print(repr(result))
        np.testing.assert_almost_equal(result, target, decimal=5)

        target = np.array([[1., 2.],
                           [2., 4.],
                           [0., 2.],
                           [1., 4.],
                           [4., 4.],
                           [4., 0.]])
        result = propose_points(tested_points, None, hyperparam_config, 6, 123)
        # print(repr(result))
        np.testing.assert_almost_equal(result, target, decimal=5)

        # Check error
        with self.assertRaises(OptimizerError):
            propose_points(tested_points, None, hyperparam_config, 20, 123)


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
            self.assertTrue(not_in_array(get_new_unique_point(previous_points, grid, 100), previous_points))

        # Check error
        with self.assertRaises(OptimizerError):
            get_new_unique_point(previous_points, previous_points)




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

    def test_get_untested_hypergrid(self):
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
        grid = get_untested_hypergrid(hyperparam_config, tested_points=tested_points)
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
