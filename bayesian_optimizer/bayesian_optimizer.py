import numpy as np
from typing import List, Dict

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
        maxidx = np.argmax(sample)
        proposed_point = hypergrid[maxidx, :]
        if not_in_array(proposed_point, new_points_so_far):
            return proposed_point
    raise OptimizerError("Could not find a new unique point within iteration number!")


def propose_points(tested_points: np.ndarray,
                   tested_values: np.ndarray,
                   hyperparams_config: List[HyperParameter],
                   num_points: int,
                   seed: int = None,
                   gp_settings: Dict = None) -> np.ndarray:
    """
    Propose new points to test based on past values

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
    hypergrid = prune_hypergrid(hypergrid, tested_points)
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

    return new_points
