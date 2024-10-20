
Paramax
============
Parameterizations and constraints for JAX PyTrees
-----------------------------------------------------------------------

Documentation available [here](https://danielward27.github.io/paramax/).

## Installation
```bash
pip install paramax
```

## Example
```python
>>> from paramax.wrappers import Parameterize, unwrap
>>> import jax.numpy as jnp
>>> params = Parameterize(jnp.exp, jnp.zeros(3))
>>> unwrap(("abc", 1, params))
('abc', 1, Array([1., 1., 1.], dtype=float32))
```

## Why use Paramax?
Paramax allows applying custom constraints or behaviors to PyTree components, such as:
- Enforcing positivity (e.g., scale parameters)
- Structured matrices (triangular, symmetric, etc.)
- Applying tricks like weight normalization
- Marking components as non-trainable

Some benefits of the pattern we use:
- It allows parameterizations to be computed once for a model (e.g. at the top of the loss function).
- It is concise, flexible, and allows custom parameterizations to be used with PyTrees from external libraries.

## Alternative patterns
Using properties to access parameterized components is common but has drawbacks:
- Parameterizations are tied to class definition, limiting flexibility e.g. this
    cannot be used on PyTrees from external libraries.
- It can become verbose with many parameters.
- It often leads to repeatedly computing the parameterization.


## Related
We make use of the [Equinox](https://arxiv.org/abs/2111.00254) package, to register
the PyTrees used in the package.
