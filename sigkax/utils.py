import jax.numpy as jnp

def _repeat(x: jnp.ndarray, n: int):
    """
    Repeat two last dimensions of a tensor
    After repeat, normalized with n**2
    """
    
    return jnp.repeat(jnp.repeat(x,
                                 repeats=n,
                                 axis=-1),
                      repeats=n,
                      axis=-2) / n**2


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
    return _repeat(x, n=n)

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
    new_xs = jnp.expand_dims(xs, axis=-1) +  eps* jnp.expand_dims(jnp.eye(xs.shape[-1]), axis=0)
    new_xs = jnp.swapaxes(new_xs, -1, -2)
    new_xs = new_xs.reshape((*xs.shape[:-2], xs.shape[-1]* xs.shape[-2], xs.shape[-1]))
    
    return new_xs