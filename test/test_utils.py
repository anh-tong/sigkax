import jax.numpy as jnp

from sigkax.utils import _repeat, finite_diff, flip_last_two, localized_impulse

def test_repeat():

    x = jnp.array([[[0., 1., 2.],
                    [3., 4., 5.]]])
    expected = jnp.array([[[0., 0., 1., 1., 2., 2.],
                           [0., 0., 1., 1., 2., 2.],
                           [3., 3., 4., 4., 5., 5.],
                           [3., 3., 4., 4., 5., 5.]]])/2**2
    res = _repeat(x, n=2)
    jnp.allclose(expected, res)


def test_flip_last_two():
    x = jnp.array([[[0., 1., 2.],
                    [3., 4., 5.]]])
    expected = jnp.array([[[5., 4., 3.],
                           [2., 1., 0.]]])
    res = flip_last_two(x)
    jnp.allclose(expected, res)


def test_finite_diff():
    x = jnp.array([[[-2.7599,  0.3896, -2.8790],
                    [-0.0213, -0.7853, -0.8277]]])
    expected = jnp.array([[[-0.2446, -0.2446, -0.2446, -0.2446,  0.2016,  0.2016,  0.2016,
                            0.2016],
                           [-0.2446, -0.2446, -0.2446, -0.2446,  0.2016,  0.2016,  0.2016,
                            0.2016],
                           [-0.2446, -0.2446, -0.2446, -0.2446,  0.2016,  0.2016,  0.2016,
                            0.2016],
                           [-0.2446, -0.2446, -0.2446, -0.2446,  0.2016,  0.2016,  0.2016,
                            0.2016]]])
    dyadic_order = 2
    res = finite_diff(x, dyadic_order=dyadic_order)
    jnp.allclose(expected, res)
    
def test_localized_impulse():
    
    x = jnp.ones((2,3))
    output = localized_impulse(x)
    assert output.shape == (6,3)

# # just for generate test cases that match with PyTorch sigkernel
# if __name__ == "__main__":

#     import torch
#     import numpy as np

#     # this is copied from sigkernel to generate test cases
#     def tile(a, dim, n_tile):
#         init_dim = a.size(dim)
#         repeat_idx = [1] * a.dim()
#         repeat_idx[dim] = n_tile
#         a = a.repeat(*(repeat_idx))
#         order_index = torch.LongTensor(np.concatenate(
#             [init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(a.device)
#         return torch.index_select(a, dim, order_index)

#     def flip(x, dim):
#         xsize = x.size()
#         dim = x.dim() + dim if dim < 0 else dim
#         x = x.view(-1, *xsize[dim:])
#         x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(
#             x.size(1)-1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
#         return x.view(xsize)

#     batch_size = 1
#     m = 2
#     n = 3
#     dyadic_order=2
#     a = torch.arange(0, batch_size * m * n).reshape((batch_size, m, n))
#     print(a)
#     ## this give
#     # tensor([[[0, 1, 2],
#     #      [3, 4, 5]]])
#     b = tile(tile(a, 1, 2), 2, 2)
#     print(b)
#     # # this give
#     # tensor([[[0, 0, 1, 1, 2, 2],
#     #      [0, 0, 1, 1, 2, 2],
#     #      [3, 3, 4, 4, 5, 5],
#     #      [3, 3, 4, 4, 5, 5]]]

#     c = flip(flip(a, dim=-1), dim=-2)
#     print(c)
#     # # this give
#     # # tensor([[[5, 4, 3],
#     # #      [2, 1, 0]]])
#     a = torch.randn(batch_size, m, n)
#     print (a)
#     diff = a[..., 1:, 1:] + a[..., :-1, :-1] - \
#         a[..., 1:, :-1] - a[..., :-1, 1:]
#     diff = tile(tile(diff, dim=1, n_tile=2**dyadic_order)/2**dyadic_order, dim=2, n_tile=2**dyadic_order)/2**dyadic_order
#     print(diff)
