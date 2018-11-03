import numpy as np
import unittest

from bayesian_optimizer.hyper_parameter import HyperParameter
from bayesian_optimizer.bayesian_optimizer import minimize_function, propose_points
from bayesian_optimizer.utilities import OptimizerError


class TestTest(unittest.TestCase):

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


if (__name__ == "__main__"):
    unittest.main()
