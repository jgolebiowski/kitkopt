import itertools
from typing import List

import numba
import numpy as np

from bayesian_optimizer.hyper_parameter import HyperParameter


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


def prune_hypergrid(hypergrid: np.ndarray, tested_points: np.ndarray) -> np.ndarray:
    """
    Prun the grid of potential hyperparameters combinations, rmoving previously seen ones

    :param hypergrid: Grid with potential hyperparameter combinations
    :param tested_points: previously tested points with dims (n_points, n_hyperparameters)
    :return: grid with previously untested combinations
    """
    if len(tested_points) == 0:
        return hypergrid

    mask = [not_in_array(potential_point, tested_points) for potential_point in hypergrid]
    return hypergrid[mask]
