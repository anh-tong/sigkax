
#include "kernel_helpers.h"
#include "cuda_kernel.h"

namespace sigkax
{

    namespace
    {

        template <typename T>
        __global__ void set_boundary_condition(T *sol_arr, int n_rows, int n_cols){
            if ((threadIdx.x <= n_rows) && (threadIdx.y <= n_cols)){
                sol_arr[threadIdx.x * (n_cols + 1)] = 1.;
                sol_arr[threadIdx.y] = 1;
            }
            __syncthreads();
        }


        template <typename T>
        __global__ void solve_pde_kernel(const T *inc_arr, T *sol_arr, int n_rows, int n_cols, int n_anti_diags)
        {
            int thread_id = threadIdx.x;
            int I = thread_id;

            T k00, k01, k10, inc;
            int k00_idx, k01_idx, k10_idx, k11_idx, inc_idx;

            for (int p = 0; p < n_anti_diags; ++p)
            {

                int J = max(0, min(p - thread_id, n_cols - 1));
                int i = I + 1;
                int j = J + 1;

                if ((I + J == p) && (I < n_rows && J < n_cols))
                {
                    // find index in 1 dim array given index of two 2 arrary
                    inc_idx = (i - 1) * n_cols + (j - 1);
                    k00_idx = (i - 1) * (n_cols + 1) + (j - 1);
                    k01_idx = (i - 1) * (n_cols + 1) + j;
                    k10_idx = i * (n_cols + 1) + (j - 1);
                    k11_idx = i * (n_cols + 1) + j;

                    inc = inc_arr[inc_idx];
                    k00 = sol_arr[k00_idx];
                    k01 = sol_arr[k01_idx];
                    k10 = sol_arr[k10_idx];

                    // update solution matrix
                    sol_arr[k11_idx] = (k01 + k10) * (1. + 0.5 *inc + pow(inc, 2) / 12.) - k00 * (1. - pow(inc, 2) / 12.);

                    // printf("(%d, %d): k00=%.2f, k01=%.2f, k10=%.2f, k11=%.2f \n", i, j, k00, k01, k10, sol_arr[k11_idx]);
                }
            }
            __syncthreads();
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
            const int n_anti_diags = 2 * std::max<int>(n_rows, n_cols) - 1;

            const T *inc_arr = reinterpret_cast<const T *>(buffers[0]);
            T *sol_arr = reinterpret_cast<T *>(buffers[1]);

            // invoke kernel to set boundary condition
            dim3 thread_per_block_1(n_rows + 1, n_cols + 1);
            set_boundary_condition<T><<<1, thread_per_block_1, 0, stream>>>(sol_arr, n_rows, n_cols);

            // allocate GPU resource here. How to do this automatically?
            dim3 thread_per_block(n_anti_diags, 1); // threadIdx.x, threadIdx.y

            // kernel invocation (grid, block, thread)
            solve_pde_kernel<T><<<1, thread_per_block, 0, stream>>>(inc_arr, sol_arr, n_rows, n_cols, n_anti_diags);
            ThrowIfError(cudaGetLastError());
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