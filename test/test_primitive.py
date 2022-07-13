import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from sigkax.primitive import solve_pde

def test_jit():
    input = jnp.ones((10,20)) * 0.1
    output = jax.jit(solve_pde)(input)

# def test_cpu_vs_gpu():
    
#     input = jnp.ones((10,20)) * 0.1
#     output_cpu = jax.jit(solve_pde, backend="cpu")(input)
#     output_gpu = jax.jit(solve_pde, backend="gpu")(input)
    
#     assert np.allclose(np.array(output_cpu), np.array(output_gpu))

# def test_batching():
    
#     input = jnp.ones((4, 10, 20)) * 0.1
#     jax.vmap(solve_pde)(input)