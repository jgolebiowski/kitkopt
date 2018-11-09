import numpy as np

from bayesian_optimizer.random_optimizer import propose_points
from bayesian_optimizer.hyper_parameter import HyperParameter
from bayesian_optimizer.kernels import rbf


def main():
    # ------ Define hyperparameters with bounds and stepsize
    hyperparam_config = [
        HyperParameter(0, 3, 1),
        HyperParameter(0, 5, 2)
    ]

    # ------ Define previously tested points and function values at those points
    # ------ Here, valus do not make a difference as the random optimizer draws from a uniform distributon
    tested_points = np.array([
        [0, 2],
        [2, 0],
        [1, 4]
    ], dtype=float)
    values = None

    # ------ Decide the number of points to be proposed each iteration
    num_points = 4

    # ------ Ask the optimizer for new points
    new_points = propose_points(tested_points, values, hyperparam_config,
                                num_points=num_points, seed=123)

    # ------ pritn the proposed points
    print(result)


if (__name__ == '__main__'):
    main()
