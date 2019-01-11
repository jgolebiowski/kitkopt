import numpy as np

from kitkopt.gaussian_process import GaussianProcessRegression
from kitkopt.hypergrid import not_in_array
from kitkopt.utilities import OptimizerError


def _get_new_unique_point_UCB(hypergrid: np.ndarray,
                              gp: GaussianProcessRegression,
                              ucb_tradeoff_parameter: float = 0.5) -> np.ndarray:
    """
    Propose a new point using Upper confidence bound acq(x) = mu(x) - b * sigma(x)
    points different from the ones proposed before.

    :param hypergrid: grid with previously untested combinations
    :param gp: Gaussian Process Regressor that has been fit to previous points
    :param num_points: number of points to be proposed
    :param ucb_tradeoff_parameter: Parameter b, determining the tradeoff brtween exploration and exploitation
    :param max_iter: Maxium number of tries for drawing new points
    :return: new points, different from the ones proposed before
    """
    acquisition = gp.mu.squeeze() - ucb_tradeoff_parameter * gp.sigmas.squeeze()
    minid = np.argmin(acquisition)
    return np.expand_dims(hypergrid[minid, :], axis=0)


def _get_new_unique_points_Thompson(hypergrid: np.ndarray,
                                    gp: GaussianProcessRegression,
                                    num_points,
                                    max_iter: int = 100) -> np.ndarray:
    """
    Propose a new point using Thompson sampling, different from the ones proposed before

    :param hypergrid: grid with previously untested combinations
    :param gp: Gaussian Process Regressor that has been fit to previous points
    :param num_points: number of points to be proposed
    :param max_iter: Maxium number of tries for drawing new points
    :return: new points, given in a (num_points, nunm_hyperparams) matrix
    """
    maxvalue = np.max(hypergrid) + 1
    new_points = np.ones((num_points, hypergrid.shape[1]), dtype=float) * maxvalue

    for idx in range(num_points):
        new_points[idx, :] = _get_single_point_Thompson(new_points, hypergrid, gp)

    return new_points


def _get_single_point_Thompson(new_points_so_far: np.ndarray,
                               hypergrid: np.ndarray,
                               gp: GaussianProcessRegression,
                               max_iter: int = 100) -> np.ndarray:
    """
    Propose a new point using Thompson sampling, different from the ones proposed before

    :param new_points_so_far: points already proposed in shape (n_points, n_hyperparams)
    :param hypergrid: grid with previously untested combinations
    :param gp: Gaussian Process Regressor that has been fit to previous points
    :param max_iter: Maxium number of tries for drawing new points
    :return: new points, different from the ones proposed before
    """

    for idx in range(max_iter):
        sample = gp.sample()
        minid = np.argmin(sample)
        proposed_point = hypergrid[minid, :]
        if not_in_array(proposed_point, new_points_so_far):
            return proposed_point
    raise OptimizerError("Could not find a new unique point within iteration number!")
