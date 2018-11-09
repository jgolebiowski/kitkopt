import numpy as np

from bayesian_optimizer.bayesian_optimizer import propose_points
from bayesian_optimizer.hyper_parameter import HyperParameter
from bayesian_optimizer.kernels import rbf

MAXIMUM_ITER = 30


def function2minimize(x):
    return np.sum(np.square(x), axis=1)


def main():
    # ------ Define hyperparameters with bounds and stepsize
    hyperparam_config = [
        HyperParameter(0, 4, 1),
        HyperParameter(-2, 2, 0.25)
    ]

    # ------ Define previously tested points and function values at those points
    tested_points = np.empty((0, len(hyperparam_config)))
    tested_values = np.empty((0, 1))

    # ------ Decide the number of points to be proposed each iteration
    num_points = 4

    # ------ Define the GP parameters including the kernel and its parameters
    gp_settings = dict(
        kernel=rbf,
        kernel_params=(0.1, 1),
        noise=1e-6
    )

    old_minimum = np.inf
    stopping_counter = 0
    tolerance = 1e-2
    for idx in range(MAXIMUM_ITER):
        # ------ Ask the optimizer for new points
        new_points = propose_points(tested_points, tested_values, hyperparam_config,
                                    num_points=num_points, gp_settings=gp_settings)
        new_values = function2minimize(new_points)
        new_values = np.expand_dims(new_values, 1)

        tested_points = np.concatenate((tested_points, new_points), axis=0)
        tested_values = np.concatenate((tested_values, new_values), axis=0)

        # ------ Compute the stopping criterion
        if np.abs(old_minimum - np.min(tested_values)) < tolerance:
            stopping_counter += 1
        else:
            stopping_counter = 0

        if stopping_counter >= 2:
            break

        old_minimum = np.min(tested_values)

    # ------ pritn the proposed points
    best_point = np.argmin(tested_values, axis=0).item()
    print(tested_points[best_point], tested_values[best_point])


if (__name__ == '__main__'):
    main()
