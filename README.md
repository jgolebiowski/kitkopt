# Kitkopt: Gradient free functions optimizer
Simple, pure python, gradient-free bayesian optimiser for black box functions.
This package supports:
##### Proposing new points based on previous observations in a stateless fasion
```python
# ------ Ask the optimizer for new points
new_points = propose_points(tested_points,
                            tested_values,
                            hyperparameters_config,
                            num_points=num_points)
```

##### Function optimisation
```python
# ------ Find the minimum and the value at the minumum
best_point, best_value = minimize_function(function2minimize,
                                           hyperparameters_config,
                                           extra_function_args=(),
                                           tolerance=1e-2,
                                           max_iterations=100)
```
Currently supported optimizers are based on sampling from a uniform distribution (random_optimizer) and a Bayesian optimiser based on a Gaussian Process Regressor with Thompson Sampling and Upper Confidence Bound acquisition functions(bayesian_optimizer). See the **examples** directory for more use-cases.

## Simple Example
The simplest application to function mimization involves (examples/bayesian-minimize)

```python
import numpy as np
from kitkopt.bayesian_optimizer import minimize_function
from kitkopt.hyper_parameter import HyperParameter

def funct(x):
    return np.sum(np.square(x))

# ------ Define hyperparameters with bounds and stepsize
hyperparameters_config = [
    HyperParameter(-5, 5, 1),
    HyperParameter(-5, 5, 1)
]

# ------ Find the minimum and the value at the minumum
best_point, best_value = minimize_function(funct, hyperparameters_config,
                                           extra_function_args=(),
                                           tolerance=1e-2,
                                           max_iterations=100)
```

### Dependencies
* [Python] 3.6 or above
* [NumPy] - Linear algebra for Python
* [SciPy] - Scientific Python library
* [Numba] - JIT compilation for Python



License
----

MIT


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [numpy]: <http://www.numpy.org/>
   [python]: <https://www.python.org/>
   [scipy]: <https://www.scipy.org/index.html>
   [numba]: <https://numba.pydata.org/numba-doc/dev/index.html>