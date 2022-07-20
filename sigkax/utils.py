import jax.numpy as jnp

def _repeat(x: jnp.ndarray, n: int):
    """
    Repeat last dimension of a tensor
    """
    
    return jnp.repeat(jnp.repeat(x,
                                 repeats=n,
                                 axis=-1),
                      repeats=n,
                      axis=-2)


def finite_diff(x: jnp.ndarray, dyadic_order: int):
    """
    Compute finite-difference with finer grid
    
    Args: 
        x: size (..., m, n)
        dyadic_order: int
    """
    x = jnp.diff(x, axis=-1)
    x = jnp.diff(x, axis=-2)
    n = 2 ** dyadic_order
    return _repeat(x, n=n) / n**2

def flip_last_two(x: jnp.ndarray):
    """
    Flip the last two dimension
    Args:
        x (jnp.ndarray): size (..., m, n)
    """
    return jnp.flip(x, axis=(-1,-2))


def localized_impulse(xs: jnp.ndarray, eps: float = 1e-4):
    """Most of the time xs has size (batch_x, len_x, dim)
    """
    dim = xs.shape[-1]
    new_xs = jnp.expand_dims(xs, axis=-1) +  eps* jnp.expand_dims(jnp.eye(dim), axis=(0, 1))
    new_xs = new_xs.transpose((0, 1, 3, 2))
    new_xs = new_xs.reshape((xs.shape[0], xs.shape[1]* xs.shape[2], xs.shape[2]))
    
    return new_xs