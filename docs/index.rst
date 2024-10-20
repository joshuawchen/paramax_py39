Paramax
===========

Paramax: a small package for applying parameterizations, constraints to JAX PyTrees.


Installation
------------------------
.. code-block:: bash

    pip install paramax


Simple example
------------------
The most common way to apply parameterizations is via
:py:class:`~paramax.wrappers.Parameterize`. This class takes a callable and any
positional or keyword arguments, which are stored and passed to the function when
unwrapping.

When :py:func:`~paramax.wrappers.unwrap` is called on a PyTree containing a
:py:class:`~paramax.wrappers.Parameterize` object, the stored function is applied
using the stored arguments.

.. doctest::

   >>> import paramax
   >>> import jax.numpy as jnp
   >>> scale = jnp.ones(3)  # Keep this positive
   >>> constrained_scale = paramax.Parameterize(jnp.exp, jnp.log(scale))
   >>> model = ("abc", 1, constrained_scale)  # Any PyTree
   >>> paramax.unwrap(model)  # Unwraps any AbstractUnwrappables
   ('abc', 1, Array([1., 1., 1.], dtype=float32))


Many simple parameterizations can be handled with this class. As another example,
we can parameterize a lower triangular matrix (such that it remains lower triangular
during optimization) as follows

.. doctest::

   >>> import paramax
   >>> import jax.numpy as jnp
   >>> tril = jnp.tril(jnp.ones((3,3)))
   >>> tril = paramax.Parameterize(jnp.tril, tril)


See :doc:`/api/wrappers` for more possibilities.

When to unwrap
-------------------
- Unwrap whenever necessary, typically at the top of loss functions, functions or 
  methods requiring the parameterizations to have been applied.
- Unwrapping prior to a gradient computation used for optimization is usually a mistake!

.. toctree::
   :caption: API
   :maxdepth: 1
   :glob:

   api/wrappers
   api/utils

