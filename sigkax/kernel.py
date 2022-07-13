import equinox as eqx

import jax
import jax.numpy as jnp
from sigkax.primitive import solve_pde


def _diff(input):
    return input[1:, 1:] + input[:-1, :-1] - input[1:, :-1] - input[:-1, 1:]


def _double_tile(input, n):
    return jnp.repeat(
        jnp.repeat(input,
                   repeats=n,
                   axis=0),
        repeats=n,
        axis=1)


class BaseSigKernel(eqx.Module):

    dyadic_order: int

    def static_kernel(self, x, y):
        raise NotImplementedError

    def kernel(self, xs, ys):
        """

        Args:
            xs: size (len_x, dim)
            xs: size (len_x, dim)
        """

        
        n, dim = xs.shape
        m, _ = ys.shape
        assert dim == ys.shape[-1]

        K = self.static_kernel(xs, ys)
        K_diff = _diff(K)

        # n_aug = (2**self.dyadic_order)*(n-1)
        # m_aug = (2**self.dyadic_order)*(m-1)
        
        K_aug = _double_tile(K_diff, n=2**self.dyadic_order) / ((2**self.dyadic_order)**2)
            

        return solve_pde(K_aug)
    
def _exp_quadratic(x, y):
    return jnp.exp(-jnp.sum(x-y)**2)

def cov_map(cov_fn, xs, ys=None):
    
    if ys is None:
        ys = xs
    
    return jax.vmap(lambda x: jax.vmap(lambda y: cov_fn(x, y))(xs))(ys).T
    
class RBFSigKernel(BaseSigKernel):
    
    def static_kernel(self, x, y):
        return cov_map(_exp_quadratic, x, y)
    
    
if __name__ == "__main__":
    len_x, dim = 3,2
    len_y = 4
    xs = jnp.ones((len_x, dim))
    ys = jnp.zeros((len_y, dim))
    
    sk = RBFSigKernel(dyadic_order=2)
    sk.kernel(xs, ys)