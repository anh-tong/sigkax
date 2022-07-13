
#include "cuda_kernel.h"
#include "kernel_pybind11_helpers.h"

using namespace sigkax;

namespace
{
    pybind11::dict Registrations()
    {
        pybind11::dict dict;
        dict["gpu_solve_pde_f32"] = EncapsulateFunction(gpu_solve_pde_f32);
        dict["gpu_solve_pde_f64"] = EncapsulateFunction(gpu_solve_pde_f64);
        return dict;
    }

    PYBIND11_MODULE(gpu_ops, m)
    {
        m.def("registrations", &Registrations);
        m.def("build_sigkax_descriptor",
              [](std::int64_t n_rows, std::int64_t n_cols)
              { return PackDescriptor(Descriptor{n_rows, n_cols}); });
    }
}