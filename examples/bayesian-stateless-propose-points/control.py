import numpy as np

from blackop.bayesian_optimizer import propose_points
from blackop.hyper_parameter import HyperParameter
from blackop.kernels import rbf


def main():
    # ------ Define hyperparameters with bounds and stepsize
    hyperparam_config = [
        HyperParameter(0, 3, 1),
        HyperParameter(0, 5, 2)
    ]

    # ------ Define previously tested points and function values at those points
    tested_points = np.array([
        [0, 2],
        [2, 0],
        [1, 4]
    ], dtype=float)
    values = np.array([1,
                       2,
                       3], dtype=float)

    # ------ Decide the number of points to be proposed each iteration
    num_points = 4

    # ------ Define the GP parameters including the kernel and its parameters
    gp_settings = dict(
        kernel=rbf,
        kernel_params=(0.1, 1),
        noise=1e-6
    )

    # ------ Ask the optimizer for new points
    new_points = propose_points(tested_points, values, hyperparam_config,
                                num_points=num_points, seed=123, gp_settings=gp_settings)

    # ------ pritn the proposed points
    print(result)


if (__name__ == '__main__'):
    main()
