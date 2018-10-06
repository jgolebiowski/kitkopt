import numpy as np
import itertools
import numba
from typing import List

from .hyper_parameter import HyperParameter


@numba.jit(
    [
        numba.boolean(numba.float64[:], numba.float64[:, :], numba.float64),
        numba.boolean(numba.float32[:], numba.float32[:, :], numba.float32)
    ], nopython=False
)
def _numba_not_in_array(vector: np.ndarray, array: np.ndarray, delta: float = 1e-4) -> bool:
    """
    Check if a given vector is NOT a row of a given array
    """
    diff = np.abs(array - vector)
    for idx in range(array.shape[0]):
        localdiff = np.max(diff[idx, :])
        if localdiff < delta:
            return False

    return True


def not_in_array(vector: np.ndarray, array: np.ndarray, delta: float = 1e-4) -> bool:
    """
    Check if a given vector is NOT a row of a given array


    :param vector: vector in shape (dim1, )
    :param array: array in shape (dim2, dim1)
    :param delta: delta used to compute float equality
    :return: True if a given vector is NOT a row of a given array
    """

    if len(array) == 0 or len(vector) == 0:
        return False

    try:
        return _numba_not_in_array(vector, array, delta)
    except TypeError:
        diff = np.min(np.max(np.abs(vector - array), axis=1))
        return (diff > delta)


def get_hypergrid(hyperparams_config: List[HyperParameter]) -> np.ndarray:
    """
    Return a grid with all potential hyperparameter combinations

    :param hyperparams_config: List of hyperparameters
    :return: grid with possible combinations
    """
    hypervalues = [
        np.arange(hyperparam.lower_bound, hyperparam.upper_bound + hyperparam.stepsize / 2, hyperparam.stepsize)
        for hyperparam in hyperparams_config
    ]
    potential_points = [item for item in itertools.product(*hypervalues)]
    potential_points = np.array(potential_points, dtype=float)
    return potential_points


def get_untested_hypergrid(hyperparams_config: List[HyperParameter], tested_points: np.ndarray) -> np.ndarray:
    """
    Get a grid of previously untested points

    :param hyperparams_config: List of hyperparameters
    :param tested_points: previously tested points with dims (n_points, n_hyperparameters)
    :return: grid with previously untested combinations
    """
    hypergrid = get_hypergrid(hyperparams_config)
    mask = [not_in_array(potential_point, tested_points) for potential_point in hypergrid]
    return hypergrid[mask]


def get_new_unique_point(new_points_so_far: np.ndarray, hypergrid: np.ndarray, max_iter: int = 100) -> np.ndarray:
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

    hypergrid = get_untested_hypergrid(hyperparams_config, tested_points)
    maxvalue = hyperparams_config[0].upper_bound + 1
    new_points = np.ones((num_points, len(hyperparams_config)), dtype=float) * maxvalue

    for idx in range(num_points):
        new_points[idx, :] = get_new_unique_point(new_points, hypergrid)

    return new_points


class OptimizerError(RuntimeError):
    pass
