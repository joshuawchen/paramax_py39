
Paramax
============
Work in progress. Documentation available [here](https://danielward27.github.io/paramax/index.html).

Parameterizations and constraints for JAX PyTrees
-----------------------------------------------------------------------


## Related
We make use of the [Equinox](https://arxiv.org/abs/2111.00254) package, to register
the PyTrees used in the package.

## Positives
TODO

## Negatives
TODO
- Unwrapping after gradient computation would likely lead to invalid gradients,
applying the reparameterization to the gradient itself.
