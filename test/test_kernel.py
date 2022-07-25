import jax
import jax.numpy as jnp
import jax.random as jrandom
from sigkax.kernel import RBFSigKernel


jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")  # TODO: test with GPU


def test_rbf():

    dyadic_order = 3
    batch_x, batch_y = 2, 3
    len_x, len_y = 3, 4
    dim = 2
    xs = jrandom.normal(key=jrandom.PRNGKey(0), shape=(batch_x, len_x, dim))
    ys = jrandom.normal(key=jrandom.PRNGKey(1), shape=(batch_y, len_y, dim))
    xs /= jnp.max(xs)
    ys /= jnp.max(ys)

    sk = RBFSigKernel(
        log_scale=jnp.array(0.0),
        log_length_scale=jnp.array(0.0),
        dyadic_order=dyadic_order,
        use_autodiff=False,
    )

    # do not jit this function
    def func_no_pde(a):
        def _batch_y(x):
            return jax.vmap(lambda _y: sk.naive_kernel(x, _y))(ys * a)

        return jnp.sum(jax.vmap(_batch_y)(xs * a))

    @jax.jit
    def func(a):
        def _batch_y(x):
            return jax.vmap(lambda _y: sk.kernel(x, _y))(ys * a)

        return jnp.sum(jax.vmap(_batch_y)(xs * a))

    a = 1.0
    our_estimate, our_grad = jax.value_and_grad(func)(a)

    naive_estimate, naive_grad = jax.value_and_grad(func_no_pde)(a)

    jnp.allclose(our_estimate, naive_estimate)
    # allow this relative tolerance 0.01
    jnp.allclose(our_grad, naive_grad, rtol=1e-2)
