#ifndef SIGKAX_CUDA_KERNEL_H_
#define SIGKAX_CUDA_KERNEL_H_

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace sigkax
{
#ifdef __CUDACC__
#define SIGKAX_INLINE_OR_DEVICE __host__ __device__
#else
#define SIGKAX_INLINE_OR_DEVICE inline
#endif
    struct Descriptor
    {
        /* define struct of number of rows and columns
        This will be unpacked from opaque and opaque_len
         */
        std::int64_t n_rows;
        std::int64_t n_cols;
    };

    void gpu_solve_pde_f32(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);

    void gpu_solve_pde_f64(cudaStream_t stream, void **buffers, const char *opaque, std::size_t opaque_len);

}

#endif /* SIGKAX_CUDA_KERNEL_H_ */
