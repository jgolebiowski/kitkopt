class HyperParameter(object):
    """
    Class for holding hyperparameter configuration
    Tested values will be drawn from range(lower_bound, upper_bound, stepsize)

    :param lower_bound: Maximum value
    :param upper_bound: Minimum value
    :param stepsize: stepsize
    """
    lower_bound: float
    upper_bound: float
    stepsize: float

    def __init__(self, lower_bound: float, upper_bound: float, stepsize: float):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.stepsize = stepsize
    