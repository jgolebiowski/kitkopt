import numpy as np

from bayesian_optimizer.random_optimizer import minimize_function
from bayesian_optimizer.hyper_parameter import HyperParameter
from bayesian_optimizer.kernels import rbf


def funct(x):
    return np.sum(np.square(x))


def main():
    # ------ Define hyperparameters with bounds and stepsize
    hyperparam_config = [
        HyperParameter(-5, 5, 1),
        HyperParameter(-5, 5, 1)
    ]

    # ------ Find the minimum and the value at the minumum
    best_point, best_value = minimize_function(funct, hyperparam_config,
                                               extra_function_args=(),
                                               tolerance=1e-2,
                                               max_iterations=100,
                                               seed=123)
    print("Best point {point} with the value of {value}".format(point=best_point, value=best_value))


if (__name__ == '__main__'):
    main()
