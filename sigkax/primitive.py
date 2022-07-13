from functools import partial
from platform import platform

import numpy as np

import jax.numpy as jnp
from jax import core, dtypes, lax
from jaxlib import xla_client
from jax.abstract_arrays import ShapedArray
from jax.interpreters import xla

from sigkax import cpu_ops

for name, value in cpu_ops.registrations().items():
    xla_client.register_cpu_custom_call_target(name, value)


def _solve_pde_abstract(inc_mat):
    input_shape = inc_mat.shape
    dtype = dtypes.canonicalize_dtype(inc_mat.dtype)
    output_shape = (input_shape[0] + 1, input_shape[1] + 1)
    return ShapedArray(output_shape, dtype)


def _solve_pde_translation(ctx,
                           inc_mat,
                           *,
                           platform="cpu"):

    input_shape = ctx.get_shape(inc_mat)

    dtype = input_shape.element_type()
    dims = input_shape.dimensions()

    output_shape = xla_client.Shape.array_shape(
        np.dtype(dtype), tuple(dim + 1 for dim in dims), tuple(range(len(dims)-1, -1, -1)))

    if dtype == np.float32:
        op_name = platform.encode() + b"_solve_pde_f32"
    elif dtype == np.float64:
        op_name = platform.encode() + b"_solve_pde_f64"
    else:
        raise NotImplementedError(f"dtype {dtype} is not supported")

    if platform == "gpu":
        raise NotImplementedError
    elif platform == "cpu":
        return xla_client.ops.CustomCallWithLayout(
            ctx,
            op_name,
            operands=(xla_client.ops.ConstantLiteral(ctx, dims[0]),
                      xla_client.ops.ConstantLiteral(ctx, dims[1]),
                      inc_mat),
            operand_shapes_with_layout=(xla_client.Shape.array_shape(np.dtype(np.int64), (), ()),
                                        xla_client.Shape.array_shape(
                                            np.dtype(np.int64), (), ()),
                                        input_shape),
            shape_with_layout=output_shape)
    else:
        raise ValueError(
            f"Platform {platform} is not supported. It should be 'cpu' or 'gpu'")


def solve_pde(inc_mat):

    return _solve_pde_prim.bind(inc_mat)


_solve_pde_prim = core.Primitive("solve_pde")
_solve_pde_prim.def_impl(partial(xla.apply_primitive, _solve_pde_prim))
_solve_pde_prim.def_abstract_eval(_solve_pde_abstract)

xla.backend_specific_translations["cpu"][_solve_pde_prim] = partial(
    _solve_pde_translation, platform="cpu")

if __name__ == "__main__":

    inc_mat = np.ones((2, 2))
    import jax
    output = jax.jit(solve_pde, backend="cpu")(inc_mat)
    print(output)
