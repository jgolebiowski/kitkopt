# bayesian-optimizer
Simple, pure python, gradient-free bayesian optimiser for black box functions. This package supports
##### Proposing new points based on previous observations in a stateless fasion:
```python
# ------ Ask the optimizer for new points
new_points = propose_points(tested_points,
                            tested_values,
                            hyperparameters_config,
                            num_points=num_points)
```
##### Function optimisation:
```python
# ------ Find the minimum and the value at the minumum
best_point, best_value = minimize_function(function2minimize,
                                           hyperparameters_config,
                                           extra_function_args=(),
                                           tolerance=1e-2,
                                           max_iterations=100)
```
for optimizers based on uniform distribution (random_optimizer) and a multi-varied gaussian based on a Gaussian Process Regressor (bayesian_optimizer). See the examples directory for more usecases.

## Simple Example
The most simple case for function mimization involves (examples/bayesian-minimize)

```python
import numpy as np
from bayesian_optimizer.bayesian_optimizer import minimize_function
from bayesian_optimizer.hyper_parameter import HyperParameter

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
* [numpy] - Linear algebra for Python
* [scipy] - Scientific Python library



License
----

MIT


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)


   [numpy]: <http://www.numpy.org/>
   [python]: <https://www.python.org/>
   [scipy]: <https://www.scipy.org/index.html>