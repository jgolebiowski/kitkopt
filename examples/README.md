# Examples

Here are some example usages of this optimizer.
- Bayesian stateless propose points
    - Here, the bayesian optimizer is used to propose new points to test. The routine used in this approach is stateless: at each iteration a whole set of previously tested points / values is passed to the function. This is useful in situations where this functionality is used from an external system
- Bayesian minimize
    - Here, the bayesian optimizer is used to find the minumum of a given function. This reduces the per-iteration initialization overhead present in the previous approach.
- Random stateless propose points
    - Here, the random optimizer is used to propose new points to test. The routine used in this approach is stateless: at each iteration a whole set of previously tested points / values is passed to the function. This is useful in situations where this functionality is used from an external system
- Bayesian minimize
    - Here, the random optimizer is used to find the minumum of a given function. This reduces the per-iteration initialization overhead present in the previous approach.