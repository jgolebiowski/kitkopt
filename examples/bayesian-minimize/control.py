import numpy as np

from kitkopt.bayesian_optimizer import minimize_function
from kitkopt.hyper_parameter import HyperParameter
from kitkopt.kernels import rbf


def funct(x):
    return np.sum(np.square(x))


def main():
    # ------ Define hyperparameters with bounds and stepsize
    hyperparam_config = [
        HyperParameter(-5, 5, 1),
        HyperParameter(-5, 5, 1)
    ]

    # ------ Define the GP parameters including the kernel and its parameters
    gp_settings = dict(
        kernel=rbf,
        kernel_params=(0.1, 1),
        noise=1e-6
    )

    # ------ Find the minimum and the value at the minumum
    best_point, best_value = minimize_function(funct, hyperparam_config,
                                               extra_function_args=(),
                                               tolerance=1e-2,
                                               max_iterations=100,
                                               seed=123,
                                               gp_settings=gp_settings)
    print("Best point {point} with the value of {value}".format(point=best_point, value=best_value))


if (__name__ == '__main__'):
    main()
