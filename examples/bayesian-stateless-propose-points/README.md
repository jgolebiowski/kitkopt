# Bayesian stateless propose points
Here, the bayesian optimizer is used to propose new points to test. The routine used in this approach is stateless: at each iteration a whole set of previously tested points / values is passed to the function. This is useful in situations where this functionality is used from an external system.

There are two example scripts
- control.py
    - Use the propose_points routine to get new, promising candidate points
- control-iterative.py
    - Use the propose_points routine to perform function minimisation (the example in bayesian-minimise might be more efficient)