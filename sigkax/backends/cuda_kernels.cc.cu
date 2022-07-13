
#include "kernel_helpers.h"
#include "cuda_kernel.h"
// #include "ops.h"

namespace sigkax
{

    namespace
    {
        template <typename T>
        __global__ void solve_pde_kernel(const T *inc_arr, T *sol_arr, int n_rows, int n_cols)
        {
           
        }

        void ThrowIfError(cudaError_t error)
        {
            if (error != cudaSuccess)
            {
                throw std::runtime_error(cudaGetErrorString(error));
            }
        }

        template <typename T>
        inline void apply_solve_pde(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len)
        {
            const Descriptor &d = *UnpackDescriptor<Descriptor>(opaque, opaque_len);
            const int n_rows = d.n_rows;
            const int n_cols = d.n_cols;

            const T *inc_arr = reinterpret_cast<const T *>(buffers[0]);
            T *sol_arr = reinterpret_cast<T *>(buffers[1]);


            // allocate GPU resource here. How to do this automatically?
            const int block_dim = 128;
            const int grid_dim = std::min<int>(1024, 1024);

            solve_pde_kernel<T><<<grid_dim, block_dim, 0, stream>>>(inc_arr, sol_arr, n_rows, n_cols);
        }

    } // namespace

    void gpu_solve_pde_f32(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len)
    {
        apply_solve_pde<float>(stream, buffers, opaque, opaque_len);
    }

    void gpu_solve_pde_f64(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len)
    {
        apply_solve_pde<double>(stream, buffers, opaque, opaque_len);
    }
} // namespace sigkax