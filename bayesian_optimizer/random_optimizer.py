import numpy as np
from typing import List, Callable, Tuple

from bayesian_optimizer.hypergrid import not_in_array, get_hypergrid, prune_hypergrid
from bayesian_optimizer.utilities import OptimizerError
from .hyper_parameter import HyperParameter


def _get_new_unique_point(new_points_so_far: np.ndarray, hypergrid: np.ndarray, max_iter: int = 100) -> np.ndarray:
    """
    Propose a new point, different from the ones proposed before

    :param new_points_so_far: points already proposed in shape (n_points, n_hyperparams)
    :param hypergrid: grid with previously untested combinations
    :param max_iter: Maxium number of tries for drawing new points
    :return: new point, different from the ones proposed before
    """

    for idx in range(max_iter):
        proposed_point = hypergrid[np.random.randint(0, len(hypergrid)), :]
        if not_in_array(proposed_point, new_points_so_far):
            return proposed_point
    raise OptimizerError("Could not find a new unique point within iteration number!")


def propose_points(tested_points: np.ndarray,
                   tested_values: np.ndarray,
                   hyperparams_config: List[HyperParameter],
                   num_points: int,
                   seed: int = None) -> np.ndarray:
    """
    Propose new points to test based on past values

    :param tested_points: previously tested points with dims (n_points, n_hyperparameters)
    :param tested_values: List of function values for previously tested points dims: (n_points, 1)
    :param hyperparams_config: List of hyperparameters
    :param num_points: number of points to propose
    :param seed: randomizer seed
    :return: New points to test with dims (num_points, n_hyperparameters)
    """

    if seed is not None:
        np.random.seed(seed)

    hypergrid = get_hypergrid(hyperparams_config)
    hypergrid = prune_hypergrid(hypergrid, tested_points)
    maxvalue = hyperparams_config[0].upper_bound + 1
    new_points = np.ones((num_points, len(hyperparams_config)), dtype=float) * maxvalue

    for idx in range(num_points):
        new_points[idx, :] = _get_new_unique_point(new_points, hypergrid)

    return new_points


def optimize_function(function: Callable,
                      hyperparams_config: List[HyperParameter],
                      extra_function_args: Tuple,
                      tolerance: float = 1e-2,
                      seed: int = None) -> Tuple[np.ndarray, float]:
    """
    Find the minimum of a function

    :param function: Function to be optimized. It will be called as ``f(x, *args)``,
        where ``x`` is the argument in the form of a 1-D array and ``args``
        is a  tuple of any additional fixed parameters needed to completely specify the function.
    :param hyperparams_config: List of hyperparameters
    :param tolerance: Convergence tolerance
    :param seed: randomizer seed

    :return:
        - Solution (best point)
        - Function value at best point
    """
    pass
    # HERE just operate on rescaed hypergrid all the time and onle back-scale it once the result is done
    # this will make the propose-points function operate on rescaled grid and only responsible for setting up
    ### the GP and proposing points from it, higher level routines take care of the scaling buisness.
