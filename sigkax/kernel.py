import equinox as eqx

import jax
import jax.numpy as jnp
from sigkax.primitive import solve_pde
from sigkax.utils import flip_last_two, _repeat, finite_diff, localized_impulse

class BaseSigKernel(eqx.Module):

    dyadic_order: int
    use_autodiff: bool = True
    eps: float = 1e-4           # finite-difference

    def static_kernel(self, x, y):
        """Compute static kernel"""
        raise NotImplementedError

    def kernel(self, xs, ys):
        """
        
        Tutorial reference for build custom JVP in JAX:
            https://jax.readthedocs.io/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html
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
            """
            Forward function do the same thing with the main function
            It can also return additional RESIDUALS for further computation in backward pass
            In PyTorch, such residuals are passed via `ctx` of `torch.autograd.Function` 
            """
            
            def _v_sub_routine(x):
                return jax.vmap(lambda y: _sub_routine(x, y))(ys)
            
            pde_sol, kernel, dot_kernel = jax.vmap(_v_sub_routine)(xs)
            res = (xs, ys, kernel, dot_kernel, pde_sol)
            return pde_sol[...,-1,-1], res
            
        def _bwd_fn(res, g):
            """
            Backward function should return the same PyTree structure like the input function
            In this case, the kernel function takes two input `xs` and `ys`, the backward function 
            should return a tuple of two tensors having the same shape with `xs` and `ys`.
            """
            
            xs, ys, kernel, forward_vf, forward_sol = res
            
            batch_x, batch_y = xs.shape[0], ys.shape[0]
            len_x, len_y = xs.shape[1], ys.shape[1]
            dim = xs.shape[-1]
            
            # backward vector field is flipped then solve backward PDE
            backward_vf = flip_last_two(forward_vf)
            backward_sol = jax.vmap(jax.vmap(solve_pde))(backward_vf)         # TODO: whether to use double vmap or implement in C++
            backward_sol = flip_last_two(backward_sol)
            
            batch_vf = forward_sol[..., :-1, :-1] * backward_sol[...,1:,1:]
            
            
            def _get_grad(x, y, vf, kernel=None):
                """
                Args:
                    x: size (len_x, dim)
                    y: size (len_y, dim)
                Return 
                    size (len_x, dim)
                """
                len_x, dim = x.shape
                vf = jnp.expand_dims(vf, axis=0)
                def _f_eps(h):
                    
                    # localized impulse: each pair of (data point, dimension) is added with a small difference
                    new_x = localized_impulse(x)
                    new_kernel = self.static_kernel(new_x, y)
                    new_kernel = new_kernel.reshape((len_x, dim, -1))
                    new_kernel = new_kernel.transpose((1, 0, 2))
                    diff = jnp.diff(new_kernel, axis=-1)
                    if kernel is not None:
                        # finite difference method
                        diff = diff - jnp.expand_dims(jnp.diff(kernel, axis=-1),
                                                    axis=0)
                    diff_1 = diff[:,1:,:]
                    diff_2 = diff[:,1:,:] - diff[:,:-1,:]
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
                    """Use JVP"""
                    _, (grad_1, grad_2) = jax.jvp(_f_eps, (0., ), (1., ))
                    
                grad_1 = grad_1.sum(axis=-1)
                grad_1 = grad_1.reshape((len_x-1, -1, dim)).sum(axis=1)
                grad_2 = grad_2.sum(axis=-1)
                grad_2 = grad_2.reshape((len_x-1, -1, dim)).sum(axis=1)
                
                grad_incr = grad_2[1:,:] - jnp.diff(grad_1, axis=0)
                start = jnp.expand_dims(grad_2[0,:] - grad_1[0,:], axis=0)
                end = jnp.expand_dims(grad_1[-1,:], axis=0)
                
                grad_points = jnp.concatenate([start, grad_incr, end], axis=0)
                return grad_points
            
            if self.use_autodiff:
                
                # w.r.t x
                def _get_grad_batch_y(x, vf):
                    return jax.vmap(lambda _y, _vf: _get_grad(x, _y, vf=_vf, kernel=None))(ys, vf)
                _get_grad_batch_xy = jax.vmap(_get_grad_batch_y)
                grad_x = _get_grad_batch_xy(xs, batch_vf)
                grad_x = jnp.expand_dims(g, axis=(-1, -2)) * grad_x
                grad_x = grad_x.sum(axis=1)
                
                # w.r.t y
                def _get_grad_batch_y(x, vf):
                    return jax.vmap(lambda _y, _vf: _get_grad(x, _y, vf=_vf, kernel=None))(xs, vf)
                _get_grad_batch_xy = jax.vmap(_get_grad_batch_y)
                grad_y = _get_grad_batch_xy(ys, batch_vf.transpose((1,0,3,2)))
                grad_y = jnp.expand_dims(g, axis=(-1, -2)) * grad_y
                grad_y = grad_y.sum(axis=1)
                return (grad_x, grad_y)
            else:
                # -------
                #  NEW
                # -------
                def _get_grad_batch_y(x, vf, kernel):
                    return jax.vmap(lambda _y, _vf, _kernel: _get_grad(x, _y, _vf, _kernel))(ys, vf, kernel)
                _get_grad_batch_xy = jax.vmap(_get_grad_batch_y)
                grad_x = _get_grad_batch_xy(xs, batch_vf, kernel)
                grad_x = jnp.expand_dims(g, axis=(-1, -2)) * grad_x
                grad_x = grad_x.sum(axis=1)
                
                # w.r.t y
                def _get_grad_batch_y(x, vf, kernel):
                    return jax.vmap(lambda _y, _vf, _kernel: _get_grad(x, _y, _vf, _kernel))(xs, vf, kernel)
                _get_grad_batch_xy = jax.vmap(_get_grad_batch_y)
                grad_y = _get_grad_batch_xy(ys, batch_vf.transpose((1,0,3,2)), kernel.transpose((1,0, 3, 2)))
                grad_y = jnp.expand_dims(g, axis=(-1, -2)) * grad_y
                grad_y = grad_y.sum(axis=1)
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
    
    jax.config.update('jax_disable_jit', True)
    jax.config.update('jax_platform_name', "cpu")
    
    batch_x, batch_y = 2, 3
    len_x, len_y = 3, 4
    dim = 2
    # xs = jrandom.normal(key=jrandom.PRNGKey(0), shape=(batch_x, len_x, dim))
    # ys = jrandom.normal(key=jrandom.PRNGKey(1), shape=(batch_y, len_y, dim))
    
    xs = jnp.array([[[0., 1., 2.],
                     [3., 4., 5.]]]) / 5.
    ys = jnp.array([[[0., 2., 4.],
                     [6., 8., 10.]]]) /10.
    
    sk = RBFSigKernel(dyadic_order=5, use_autodiff=False)    
   
    def func(a):
        return jnp.sum(sk.kernel(xs*a, ys*a))
    # it fails gradient check of JAX
    # func = jax.jit(func, backend="cpu")
    a = 0.5
    func(a)
    our_grad =  jax.grad(func)(a)
    print(our_grad)
    eps = 1e-4
    numerical_grad = (func(a + eps / 2.) - func(a- eps / 2.))/eps
    print(numerical_grad)    
   
    