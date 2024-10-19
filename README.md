
Paramax
============

Parameterizations and constraints
-----------------------------------------------------------------------


## Related
We make use of the [Equinox](https://arxiv.org/abs/2111.00254) package, to register
the PyTrees used in the package.

## Positives

## Negatives
- Unwrapping after gradient computation would likely lead to invalid gradients,
applying the reparameterization to the gradient itself.
