"""Utility functions."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


def inv_softplus(x: ArrayLike) -> Array:
    """The inverse of the softplus function, checking for positive inputs."""
    x = eqx.error_if(
        x,
        x < 0,
        "Expected positive inputs to inv_softplus. If you are trying to use a negative "
        "scale parameter, consider constructing with positive scales and modifying the "
        "scale attribute post-construction, e.g., using eqx.tree_at.",
    )
    return jnp.log(-jnp.expm1(-x)) + x
