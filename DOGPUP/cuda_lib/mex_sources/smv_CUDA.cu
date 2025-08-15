#include "mex.h"
#include <math.h>  
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "smvMex.hpp"

__global__ void smv_kernel(double* const out_real,
                                double* const out_imag,
                                const double* const x_real,
                                const double* const x_imag,
                                const double* const fAxis,
                                const double* const valA_real,
                                const double* const valA_imag,
                                const int* const rPtr,
                                const int* const cPtr,
                                const int nrows,
                                const int ncols,
                                const int nfreqs)
{
    // Thread indices for row, column, and frequency
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int freq = blockIdx.z * blockDim.z + threadIdx.z;

    // Initialize to zero
    double temp_real = 0.0;
    double temp_imag = 0.0;
    
    if (row < nrows && col < ncols && freq < nfreqs) 
    {
        int outIdx = freq * ncols * nrows + col * nrows + row;
        double f = fAxis[freq]; // Read frequency axis for the current frequency
        int row_start = rPtr[row]; // Start index of non-zero elements in the row
        int row_end = rPtr[row + 1]; // End index of non-zero elements in the row

        // Loop over the non-zero elements in the row
        for (int i = row_start; i < row_end; ++i) 
        {
            int col_idx = cPtr[i];
            int xIdx = freq * ncols * nrows + col * nrows + col_idx;

            double real_val = valA_real[i];  // Real part of matrix value
            double imag_val = f * valA_imag[i];  // Imaginary part scaled by frequency

            // Accumulate real and imaginary parts
            temp_real += real_val * x_real[xIdx] - imag_val * x_imag[xIdx];
            temp_imag += real_val * x_imag[xIdx] + imag_val * x_real[xIdx];
        }

        out_real[outIdx] = temp_real;
        out_imag[outIdx] = temp_imag;
    }
}



// Host function called by MEX gateway.
void smv_call(double* const d_out_real,
                      double* const d_out_imag,
                      const double* const d_x_real,
                      const double* const d_x_imag,
                      const double* const d_fAxis,
                      const double* const d_valA_real,
                      const double* const d_valA_imag,
                      const int* const d_rPtr,
                      const int* const d_cPtr,
                      const int nrows,
                      const int ncols,
                      const int nfreqs)
{

// Configure kernel execution parameters
int t1x = 32;
int t1y = 2;
int t1z = 8;
dim3 threadsPerBlock(t1x, t1y, t1z);
dim3 numBlocks((nrows + t1x - 1) / t1x, (ncols + t1y - 1) / t1y, (nfreqs + t1z - 1) / t1z);

smv_kernel<<<numBlocks, threadsPerBlock>>>(d_out_real, d_out_imag, d_x_real, d_x_imag, d_fAxis,
                                           d_valA_real, d_valA_imag, d_rPtr, d_cPtr, nrows, ncols, nfreqs);
}