import os
import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from sigkax.primitive import solve_pde

def test_jit():
    input = jnp.ones((10,20)) * 0.1
    output = jax.jit(solve_pde)(input)

@pytest.mark.skipif(os.environ.get("SIGKAX_CUDA", "no").lower() == "no", reason="no cuda")
def test_cpu_vs_gpu():
    
    input = jnp.ones((10,20)) * 0.1
    output_cpu = jax.jit(solve_pde, backend="cpu")(input)
    output_gpu = jax.jit(solve_pde, backend="gpu")(input)
    
    assert np.allclose(np.array(output_cpu), np.array(output_gpu))

def test_batching():
    
    input = jnp.ones((4, 10, 20)) * 0.1
    output = jax.vmap(solve_pde)(input)
    assert output.shape[0] == input.shape[0]
    assert output.shape[1] == input.shape[1] + 1
    assert output.shape[2] == input.shape[2] + 1