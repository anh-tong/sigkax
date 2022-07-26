import equinox as eqx
import jax
import jax.numpy as jnp
from sigkax.primitive import solve_pde
from sigkax.utils import _repeat, finite_diff, flip_last_two, localized_impulse


class BaseSigKernel(eqx.Module):

    dyadic_order: int
    use_autodiff: bool = False
    eps: float = 1e-4  # finite-difference

    def static_kernel(self, x, y):
        """Compute static kernel"""
        raise NotImplementedError

    def naive_kernel(self, x, y):
        """Pure python implementation of signature kernel
        WARNING: JAX jit suffers long compilation. DO NOT jit this function
        """
        K = self.static_kernel(x, y)
        dot_kernel = finite_diff(K, dyadic_order=self.dyadic_order)
        pde_sol = jnp.empty(
            shape=(
                dot_kernel.shape[0] + 1,
                dot_kernel.shape[1] + 1,
            )
        )
        pde_sol = pde_sol.at[0, :].set(1.0)
        pde_sol = pde_sol.at[:, 0].set(1.0)
        for i in range(dot_kernel.shape[0]):
            for j in range(dot_kernel.shape[1]):
                # print(i,j)
                incr = dot_kernel[i, j]
                k10 = pde_sol[i + 1, j].copy()
                k01 = pde_sol[i, j + 1].copy()
                k00 = pde_sol[i, j].copy()

                k11 = (k10 + k01) * (1.0 + 0.5 * incr + incr**2 / 12.0)
                k11 -= k00 * (1 - incr**2 / 12)

                pde_sol = pde_sol.at[i + 1, j + 1].set(k11)

        return pde_sol[-1, -1]

    def batch_kernel(self, xs, ys):
        """
        Args:
            xs : size (batch_x, len_x, dim)
            ys : size (batch_y, len_y, dim)
        """

        def _batch_y(x):
            return jax.vmap(lambda _y: self.kernel(x, _y))(ys)

        return jax.vmap(_batch_y)(xs)

    def kernel(self, x, y):
        """

        Tutorial reference for build custom JVP in JAX:
            https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
        Args:
            xs: size (len_x, dim). Use `jax.vmap` for batch inputs
            ys: size (len_y, dim). Use `jax.vmap` for batch inputs
        """

        def _sub_routine(x, y):
            """
            Args:
                x : size (len_x, dim)
                y : size (len_y, dim)
            """
            kernel = self.static_kernel(x, y)
            dot_kernel = finite_diff(kernel, dyadic_order=self.dyadic_order)
            pde_sol = solve_pde(dot_kernel)
            return pde_sol, kernel, dot_kernel

        @jax.custom_vjp
        def _kernel(x, y):
            pde_sol, _, _ = _sub_routine(x, y)
            # return PDE solution at T
            return pde_sol[-1, -1]

        # -------------------------------------------------------------
        #                      Forward implementation
        # -------------------------------------------------------------
        def _fwd_fn(x, y):
            """
            Forward function do the same thing with the main function
            It can also return additional RESIDUALS for further computation
            in backward pass

            In PyTorch, such residuals are passed via `ctx` of
            `torch.autograd.Function`
            """
            pde_sol, kernel, dot_kernel = _sub_routine(x, y)
            res = (x, y, kernel, dot_kernel, pde_sol)
            return pde_sol[-1, -1], res

        # -------------------------------------------------------------
        #                      Backward implementation
        # -------------------------------------------------------------
        def _bwd_fn(res, g):
            """
            Backward function should return the same PyTree structure like
            the input function
            In this case, the kernel function takes two input `xs` and `ys`,
            the backward function should return a tuple of two tensors having
            the same shape with `xs` and `ys`.
            """

            x, y, kernel, forward_vf, forward_sol = res

            # backward vector field is flipped then solve backward PDE
            backward_vf = flip_last_two(forward_vf)
            backward_sol = solve_pde(backward_vf)
            backward_sol = flip_last_two(backward_sol)

            vf = forward_sol[:-1, :-1] * backward_sol[1:, 1:]

            def _get_grad(x, y, vf, kernel=None):
                """
                This follows Theorem 4.1 of SigGPE paper
                Args:
                    x: size (len_x, dim)
                    y: size (len_y, dim)n

                    vf: size ((len_x - 1)*order, (len_y - 1)*order)
                    kernel: size (len_x, len_y)
                Return
                    size (len_x, dim)
                """
                len_x, dim = x.shape
                vf = jnp.expand_dims(vf, axis=0)

                def _f_eps(h):

                    # localized impulse: each pair of (data point, dimension)
                    # is added with a small difference
                    new_x = localized_impulse(x, h)
                    new_kernel = self.static_kernel(new_x, y)
                    new_kernel = new_kernel.reshape((len_x, dim, -1))
                    new_kernel = new_kernel.transpose((1, 0, 2))
                    if kernel is not None:
                        # finite difference method
                        diff = jnp.diff(new_kernel, axis=-1) - jnp.expand_dims(
                            jnp.diff(kernel, axis=-1), axis=0
                        )
                    diff_1 = diff[:, 1:, :]
                    diff_2 = diff[:, 1:, :] - diff[:, :-1, :]
                    diff_1 = _repeat(diff_1, n=2**self.dyadic_order)
                    diff_2 = _repeat(diff_2, n=2**self.dyadic_order)

                    eval_1 = vf * diff_1
                    eval_2 = vf * diff_2

                    return eval_1, eval_2

                if kernel is not None:
                    """The case of finite-difference"""
                    eval_1, eval_2 = _f_eps(self.eps)
                    grad_1, grad_2 = eval_1 / self.eps, eval_2 / self.eps
                else:
                    """Use JVP
                    This seems slow
                    """
                    _, (grad_1, grad_2) = jax.jvp(_f_eps, (0.0,), (1.0,))

                grad_1 = grad_1.sum(axis=-1)
                grad_1 = grad_1.reshape((dim, len_x - 1, -1)).sum(axis=-1).T
                grad_2 = grad_2.sum(axis=-1)
                grad_2 = grad_2.reshape((dim, len_x - 1, -1)).sum(axis=-1).T

                grad_incr = grad_2[1:, :] - jnp.diff(grad_1, axis=0)
                start = jnp.expand_dims(grad_2[0, :] - grad_1[0, :], axis=0)
                end = jnp.expand_dims(grad_1[-1, :], axis=0)

                grad_points = jnp.concatenate([start, grad_incr, end], axis=0)
                return grad_points

            if self.use_autodiff:
                # # use JAX JVP auto-diff
                # w.r.t x
                grad_x = _get_grad(x, y, vf)
                # w.r.t y
                grad_y = _get_grad(y, x, vf.T)

                return (grad_x, grad_y)
            else:
                # # use finite-difference method
                # w.r.t x
                grad_x = _get_grad(x, y, vf, kernel)

                # w.r.t y
                grad_y = _get_grad(y, x, vf.T, kernel.T)

            grad_x = g * grad_x
            grad_y = g * grad_y
            return (grad_x, grad_y)

        # register VJP here
        _kernel.defvjp(_fwd_fn, _bwd_fn)

        return _kernel(x, y)


def _exp_quadratic(x, y):
    return jnp.exp(-jnp.sum((x - y) ** 2))


def cov_map(cov_fn, xs, ys=None):

    if ys is None:
        ys = xs

    return jax.vmap(lambda x: jax.vmap(lambda y: cov_fn(x, y))(xs))(ys).T


class RBFSigKernel(BaseSigKernel):

    log_scale: jnp.ndarray
    log_length_scale: jnp.ndarray

    def __init__(self, log_scale, log_length_scale, **kwargs) -> None:
        super().__init__(**kwargs)
        self.log_length_scale = log_length_scale
        self.log_scale = log_scale

    def static_kernel(self, x, y):
        scale = jnp.exp(self.log_scale)
        length_scale = jnp.exp(self.log_length_scale)
        K = cov_map(_exp_quadratic, x / length_scale, y / length_scale)
        return scale * K
