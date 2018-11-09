import numpy as np
from typing import List, Dict, Tuple, Callable

from bayesian_optimizer.gaussian_process import GaussianProcessRegression
from bayesian_optimizer.kernels import rbf
from bayesian_optimizer.utilities import OptimizerError
from bayesian_optimizer.hypergrid import not_in_array, get_hypergrid, prune_hypergrid
from bayesian_optimizer.rescale import rescale_hypergrid, rescale_vector
from .hyper_parameter import HyperParameter

MIN_VALUE = -1
MAX_VALUE = 1


def _get_new_unique_point(new_points_so_far: np.ndarray,
                          hypergrid: np.ndarray,
                          gp: GaussianProcessRegression,
                          max_iter: int = 100) -> np.ndarray:
    """
    Propose a new point, different from the ones proposed before

    :param new_points_so_far: points already proposed in shape (n_points, n_hyperparams)
    :param hypergrid: grid with previously untested combinations
    :param gp: Gaussian Process Regressor that has been fit to previous points
    :param max_iter: Maxium number of tries for drawing new points
    :return: new point, different from the ones proposed before
    """

    for idx in range(max_iter):
        sample = gp.sample()
        minid = np.argmin(sample)
        proposed_point = hypergrid[minid, :]
        if not_in_array(proposed_point, new_points_so_far):
            return proposed_point
    raise OptimizerError("Could not find a new unique point within iteration number!")


def _propose_new_points(
        last_iteration_points: np.ndarray,
        tested_points: np.ndarray,
        tested_values: np.ndarray,
        hypergrid: np.ndarray,
        hyperparams_config: List[HyperParameter],
        num_points: int,
        gp_settings: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propose new points, given old ones

    :param last_iteration_points: New points proposed in the last iteration
    :param tested_points: previously tested points with dims (n_points, n_hyperparameters)
    :param tested_values: List of function values for previously tested points dims: (n_points, 1)
    :param hypergrid: grid with previously untested combinations
    :param hyperparams_config: List of hyperparameters
    :param num_points: Number of points to return
    :param gp_settings: settings for the GaussianProcessRegressor: kernel, kernel_params and noise
    :return:
        - Updated hypergrid
        - New proposed hyperparam combination
    """
    gp = GaussianProcessRegression(gp_settings["kernel"],
                                   *gp_settings["kernel_params"],
                                   noise=gp_settings["noise"])

    hypergrid = prune_hypergrid(hypergrid, last_iteration_points)
    if len(hypergrid) == 0:
        raise OptimizerError("All potential points were already tested!")
    rescaled_hypergrid = rescale_hypergrid(hypergrid.copy(), hyperparams_config)

    if tested_values.ndim == 1:
        tested_values = tested_values.reshape((-1, 1))

    if len(tested_points) == 0:
        gp.initialize(rescaled_hypergrid)
    else:
        rescaled_tested_points = rescale_hypergrid(tested_points.copy(), hyperparams_config)
        rescaled_tested_values = rescale_vector(tested_values.copy(),
                                                tested_values.min(), tested_values.max(),
                                                MIN_VALUE, MAX_VALUE)
        gp.fit(rescaled_tested_points, rescaled_tested_values, rescaled_hypergrid)

    maxvalue = hyperparams_config[0].upper_bound + 1
    new_points = np.ones((num_points, len(hyperparams_config)), dtype=float) * maxvalue

    for idx in range(num_points):
        new_points[idx, :] = _get_new_unique_point(new_points, hypergrid, gp)

    return hypergrid, new_points


def propose_points(tested_points: np.ndarray,
                   tested_values: np.ndarray,
                   hyperparams_config: List[HyperParameter],
                   num_points: int,
                   seed: int = None,
                   gp_settings: Dict = None) -> np.ndarray:
    """
    Propose new points to test based on past values. The proposed points should leaad function minimum values

    :param tested_points: previously tested points with dims (n_points, n_hyperparameters)
    :param tested_values: List of function values for previously tested points dims: (n_points, 1)
    :param hyperparams_config: List of hyperparameters
    :param num_points: number of points to propose
    :param seed: randomizer seed
    :param gp_settings: settings for the GaussianProcessRegressor: kernel, kernel_params and noise
    :return: New points to test with dims (num_points, n_hyperparameters)
    """

    if seed is not None:
        np.random.seed(seed)

    if gp_settings is None:
        gp_settings = dict(
            kernel=rbf,
            kernel_params=(0.1, 1),
            noise=1e-6
        )
    gp = GaussianProcessRegression(gp_settings["kernel"],
                                   *gp_settings["kernel_params"],
                                   noise=gp_settings["noise"])

    hypergrid = get_hypergrid(hyperparams_config)

    hypergrid, new_points = _propose_new_points(
        tested_points,
        tested_points,
        tested_values,
        hypergrid,
        hyperparams_config,
        num_points,
        gp_settings)

    return new_points


def minimize_function(function: Callable,
                      hyperparams_config: List[HyperParameter],
                      extra_function_args: Tuple = (),
                      tolerance: float = 1e-2,
                      max_iterations: int = 1000,
                      seed: int = None,
                      gp_settings: Dict = None) -> Tuple[np.ndarray, float]:
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
    :param gp_settings: settings for the GaussianProcessRegressor: kernel, kernel_params and noise

    :return:
        - Solution (best point)
        - Function value at best point
    """
    if seed is not None:
        np.random.seed(seed)

    if gp_settings is None:
        gp_settings = dict(
            kernel=rbf,
            kernel_params=(0.1, 1),
            noise=1e-6
        )

    # ------ Define the initial grid
    hypergrid = get_hypergrid(hyperparams_config)
    tested_points = np.empty((0, len(hyperparams_config)))
    tested_values = np.empty((0, 1))
    maxvalue = hyperparams_config[0].upper_bound + 1

    # ------ Define useful constants
    old_minimum = np.inf
    num_points = 1
    new_points = np.empty((0, len(hyperparams_config)))

    max_iterations = min(len(hypergrid), max_iterations)
    stopping_counter = 0
    for idx in range(max_iterations):
        # ------ Ask the optimizer for new points
        hypergrid, new_points = _propose_new_points(
            new_points,
            tested_points,
            tested_values,
            hypergrid,
            hyperparams_config,
            num_points,
            gp_settings)

        new_values = np.empty((num_points, 1))
        for idx in range(num_points):
            new_values[idx, 0] = function(new_points[idx, :], *extra_function_args)

        tested_points = np.concatenate((tested_points, new_points), axis=0)
        tested_values = np.concatenate((tested_values, new_values), axis=0)

        # ------ Compute the stopping criterion
        if np.abs(old_minimum - np.min(tested_values)) < tolerance:
            stopping_counter += 1
        else:
            stopping_counter = 0

        if stopping_counter >= 10:
            break

        old_minimum = np.min(tested_values)

    best_point = np.argmin(tested_values, axis=0).item()
    return tested_points[best_point], tested_values[best_point]
