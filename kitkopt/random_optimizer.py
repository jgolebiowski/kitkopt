import numpy as np
from typing import List, Callable, Tuple

from kitkopt.hypergrid import not_in_array, get_hypergrid, prune_hypergrid
from kitkopt.utilities import OptimizerError, debugtool
from kitkopt.hyper_parameter import HyperParameter


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


def _propose_new_points(
        previous_points: np.ndarray,
        hypergrid: np.ndarray,
        hyperparams_config: List[HyperParameter],
        num_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propose new points, given old ones

    :param previous_points: New points proposed in the last iteration
    :param hypergrid: grid with previously untested combinations
    :param hyperparams_config: List of hyperparameters
    :param num_points: Number of points to return
    :return:
        - Updated hypergrid
        - New proposed hyperparam combination
    """
    maxvalue = hyperparams_config[0].upper_bound + 1
    num_hyperparams = len(hyperparams_config)

    hypergrid = prune_hypergrid(hypergrid, previous_points)
    new_points = np.ones((num_points, num_hyperparams), dtype=float) * maxvalue

    for idx in range(num_points):
        new_points[idx, :] = _get_new_unique_point(new_points, hypergrid)
    return hypergrid, new_points


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
    hypergrid, new_points = _propose_new_points(
        tested_points,
        hypergrid,
        hyperparams_config,
        num_points)

    return new_points


def minimize_function(function: Callable,
                      hyperparams_config: List[HyperParameter],
                      extra_function_args: Tuple = (),
                      tolerance: float = 1e-2,
                      max_iterations: int = 1000,
                      seed: int = None) -> Tuple[np.ndarray, float]:
    """
    Find the minimum of a function

    :param function: Function to be optimized. It will be called as ``f(x, *extra_function_args)``,
        where ``x`` is the argument in the form of a 1-D array and ``extra_function_args``
        is a  tuple of any additional fixed parameters needed to completely specify the function.
    :param hyperparams_config: List of hyperparameters
    :param extra_function_args: Function is called as ``f(x, *extra_function_args)``
    :param tolerance: Convergence tolerance, the optimization stops if (last_best_value - new_best_value) < tolerance
    :param max_iterations: Maximum allowed number of iterations
    :param seed: randomizer seed

    :return:
        - Solution (best point)
        - Function value at best point
    """
    if seed is not None:
        np.random.seed(seed)

    hypergrid = get_hypergrid(hyperparams_config)
    tested_points = np.empty((0, len(hyperparams_config)))
    tested_values = np.empty((0, 1))
    maxvalue = hyperparams_config[0].upper_bound + 1

    old_minimum = np.inf
    num_points = 1
    new_points = np.ones((num_points, len(hyperparams_config)), dtype=float) * maxvalue

    if (len(hypergrid) < max_iterations):
        max_iterations = len(hypergrid)

    for idx in range(max_iterations):
        hypergrid, new_points = _propose_new_points(
            new_points,
            hypergrid,
            hyperparams_config,
            num_points)

        new_values = np.empty((num_points, 1))
        for idx in range(num_points):
            new_values[idx, 0] = function(new_points[idx, :], *extra_function_args)

        tested_points = np.concatenate((tested_points, new_points), axis=0)
        tested_values = np.concatenate((tested_values, new_values), axis=0)

        if np.abs(old_minimum - np.min(tested_values)) < tolerance:
            break

    best_point = np.argmin(tested_values, axis=0).item()
    return tested_points[best_point], tested_values[best_point]
