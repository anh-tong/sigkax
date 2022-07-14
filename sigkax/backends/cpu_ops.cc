#include "ops.h"
#include "kernel_pybind11_helpers.h"

using namespace sigkax;

namespace
{   
    
    template <typename T>
    void cpu_solve_pde(void *out, const void **in){
        
        //parsing input. our XLA custom call should have signature as (int, int, array)
        const std::int64_t n_rows = *reinterpret_cast<const std::int64_t *>(in[0]);
        const std::int64_t n_cols = *reinterpret_cast<const std::int64_t *>(in[1]);
        const T *inc_mat = reinterpret_cast<const T *>(in[2]);

        // Only one output
        T *sol_mat = reinterpret_cast<T *>(out);

        solve_pde(inc_mat, sol_mat, n_rows, n_cols);
    }

    // C++ function is encapsulated and exposed to Python via Pybind11
    pybind11::dict Registrations()
    {
        pybind11::dict dict;
        dict["cpu_solve_pde_f32"] = EncapsulateFunction(cpu_solve_pde<float>);
        dict["cpu_solve_pde_f64"] = EncapsulateFunction(cpu_solve_pde<double>);
        return dict;
    }

    PYBIND11_MODULE(cpu_ops, m) { m.def("registrations", &Registrations); }
}