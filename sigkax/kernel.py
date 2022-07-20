import equinox as eqx

import jax
import jax.numpy as jnp
from sigkax.primitive import solve_pde
from sigkax.utils import flip_last_two, _repeat, finite_diff, localized_impulse

class BaseSigKernel(eqx.Module):

    dyadic_order: int

    def static_kernel(self, x, y):
        """Compute static kernel"""
        raise NotImplementedError

    def kernel(self, xs, ys):
        """
        
        Args:
            xs: size (batch_x, len_x, dim)
            ys: size (batch_y, len_y, dim)
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
        def _kernel(xs, ys):
            
            def _v_sub_routine(x):
                return jax.vmap(lambda y: _sub_routine(x, y)[0])(ys)
            
            result = jax.vmap(_v_sub_routine)(xs)
            
            # return PDE solution at T
            return result[...,-1,-1]
        
        def _fwd_fn(xs, ys):
            
            def _v_sub_routine(x):
                return jax.vmap(lambda y: _sub_routine(x, y))(ys)
            
            pde_sol, kernel, dot_kernel = jax.vmap(_v_sub_routine)(xs)
            res = (xs, ys, kernel, dot_kernel, pde_sol)
            return pde_sol[...,-1,-1], res
            
        def _bwd_fn(res, g):
            
            xs, ys, kernel, forward_vf, forward_sol = res
            
            batch_x, batch_y = xs.shape[0], ys.shape[0]
            len_x, len_y = xs.shape[1], ys.shape[1]
            dim = xs.shape[-1]
            
            # backward vector field is flipped then solve backward PDE
            backward_vf = flip_last_two(forward_vf)
            backward_sol = jax.vmap(jax.vmap(solve_pde))(backward_vf)         # TODO: whether to use double vmap or implement in C++
            backward_sol = flip_last_two(backward_sol)
            
            vf = forward_sol[..., :-1, :-1] * backward_sol[...,1:,1:]
            
            # using finite-difference method. 
            # This can be not accurate. If eps is small, we need to set dtype with jnp.float64
            # TODO: implement with JAX auto-diff
            eps = 1e-4
            new_xs = localized_impulse(xs, eps)
            new_kernel = cov_map(self.static_kernel, new_xs, ys)
            new_kernel = new_kernel.reshape((batch_x, batch_y, len_x, dim, len_y))
            new_kernel = new_kernel.transpose((0, 1, 3, 2, 4))
            
            # finite-difference over Y
            diff_y = jnp.diff(new_kernel, axis=-1) - jnp.expand_dims(jnp.diff(kernel, axis=-1), axis=2)
            
            
            diff_1 = diff_y[...,1:,:]
            diff_2 = diff_y[...,1:,:] - diff_y[...,:-1,:]
            diff_1 = _repeat(diff_1, 2**self.dyadic_order) / ((2**self.dyadic_order)**2)
            diff_2 = _repeat(diff_2, 2**self.dyadic_order) / ((2**self.dyadic_order)**2)
            
            grad_1 = jnp.expand_dims(vf, axis=2) * diff_1 / eps
            grad_1 = grad_1.sum(axis=-1)
            grad_1 = grad_1.reshape((batch_x, batch_y, len_x-1, -1, dim)).sum(axis=-2)
            grad_2 = jnp.expand_dims(vf, axis=2) * diff_2 / eps
            grad_2 = grad_2.sum(axis=-1)
            grad_2 = grad_2.reshape((batch_x, batch_y, len_x-1, -1, dim)).sum(axis=-2)
            
            grad_incr = grad_2[...,1:,:] - jnp.diff(grad_1, axis=-2)
            start = jnp.expand_dims(grad_2[...,0,:] - grad_1[...,0,:], axis=2)
            end = jnp.expand_dims(grad_1[...,-1,:], axis=2)
            
            grad_points = jnp.concatenate([start, grad_incr, end], axis=2)
            
            grad_x = jnp.expand_dims(g, axis=(-1, -2)) * grad_points
            grad_x = grad_x.sum(axis=1)
            
            # TODO: implement this 
            grad_y = jnp.zeros_like(ys)
            
            return (grad_x, grad_y)
            
        # register VJP here
        _kernel.defvjp(_fwd_fn, _bwd_fn)
        
        return _kernel(xs, ys)
        
                
def _exp_quadratic(x, y):
    return jnp.exp(-jnp.sum((x-y)**2))

def cov_map(cov_fn, xs, ys=None):
    
    if ys is None:
        ys = xs
    
    return jax.vmap(lambda x: jax.vmap(lambda y: cov_fn(x, y))(xs))(ys).T
    
class RBFSigKernel(BaseSigKernel):
    
    def static_kernel(self, x, y):
        return cov_map(_exp_quadratic, x, y)
    
    
if __name__ == "__main__":
    import jax.random as jrandom
    
    # jax.config.update('jax_disable_jit', True)
    # jax.config.update('jax_platform_name', "cpu")
    
    # batch_x, batch_y = 2, 3
    # len_x, len_y = 3, 4
    # dim = 2
    # xs = jrandom.normal(key=jrandom.PRNGKey(0), shape=(batch_x, len_x, dim))
    # ys = jrandom.normal(key=jrandom.PRNGKey(1), shape=(batch_y, len_y, dim))
    
    xs = jnp.array([[[0., 1., 2.],
                     [3., 4., 5.]]]) / 5.
    ys = jnp.array([[[0., 2., 4.],
                     [6., 8., 10.]]]) /10.
    
    sk = RBFSigKernel(dyadic_order=5)    
   
    def func(a):
        return jnp.sum(sk.kernel(xs*a, ys))
    # it fails gradient check of JAX
    func = jax.jit(func, backend="cpu")
    a = 2.
    func(a)
    our_grad =  jax.grad(func)(a)
    print(our_grad)
    eps = 1e-4
    numerical_grad = (func(a + eps / 2.) - func(a- eps / 2.))/eps
    print(numerical_grad)
    
   
    