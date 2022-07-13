#ifndef SIGKAX_OPS_H_
#define SIGKAX_OPS_H_

#include <cmath>
#include <vector>
#include <iostream>

namespace sigkax
{

#ifdef __CUDACC__
#define SIGKAX_INLINE_OR_DEVICE __host__ __device__
#else
#define SIGKAX_INLINE_OR_DEVICE inline
#endif

    template <typename T>
    SIGKAX_INLINE_OR_DEVICE void solve_pde(const T *inc_arr, T *sol_arr, const int n_rows, const int n_cols)
    {

        const T(*inc_mat)[n_cols] = reinterpret_cast<const T(*)[n_cols]>(inc_arr);
        T(*sol_mat)[n_cols + 1] = reinterpret_cast<T(*)[n_cols + 1]>(sol_arr);

        // set boundary condition
        for (int i = 0; i < n_rows + 1; ++i)
        {
            sol_mat[i][0] = 1.;
        }
        for (int i = 0; i < n_cols + 1; ++i)
        {
            sol_mat[0][i] = 1.;
        }

        // recursive compute
        for (int i = 0; i < n_rows; ++i)
        {
            for (int j = 0; j < n_cols; ++j)
            {
                sol_mat[i + 1][j + 1] = (sol_mat[i][j + 1] + sol_mat[i + 1][j]) * (1. + 0.5 * inc_mat[i][j] + pow(inc_mat[i][j], 2) / 12.) - sol_mat[i][j] * (1. - pow(inc_mat[i][j], 2) / 12.);
            }
        }
    }
}

#endif /* SIGKAX_OPS_H_ */
