import numpy as np
from typing import List
from blackop.hyper_parameter import HyperParameter

TOL = 1e-12


def rescale_hypergrid(hypergrid: np.ndarray,
                      hyperparam_config: List[HyperParameter],
                      newlow: float = 0.0, newhigh: float = 1.0) -> np.ndarray:
    """
    Rescale the hyperparameters grid so that each dimensions is in (newlow, newhigh)
    Function opetares modyfing the original hypergrid

    :param hypergrid: Grid of possible hyperparameter combination in shape (n_combinations, n_hyperparams)
    :param hyperparam_config: List of hyperparameters specifying the problem
    :param newlow: New minimum value
    :param newhigh: New maximum value
    :return: Rescaled hypergrid in shape (n_combinations, n_hyperparams)
    """

    n_hyper = len(hyperparam_config)
    assert n_hyper == hypergrid.shape[1], "The Grid must be of dimensions (n_combinations, n_hyperparams)"

    for idx in range(n_hyper):
        hypergrid[:, idx] = rescale_vector(hypergrid[:, idx],
                                           hyperparam_config[idx].lower_bound,
                                           hyperparam_config[idx].upper_bound,
                                           newlow,
                                           newhigh)

    return hypergrid


def rescale_vector(data: np.ndarray,
                   oldmin: float, oldmax: float,
                   newmin: float, newmax: float) -> np.ndarray:
    """
    Rescale a one dimensional vector from (oldmin, oldmax) -> (newmin, nawmax)
    Function opetares modyfing the original vector
    """

    data -= oldmin
    if abs(oldmax - oldmin) > TOL:
        data /= oldmax - oldmin
    data *= (newmax - newmin)
    data += newmin
    return data
