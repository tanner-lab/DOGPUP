#include "mex.h"
#include <math.h>  
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "FSAIP.hpp"

struct Complex {
    double real;
    double imag;
};

__device__ Complex complexProduct(double a, double b, double c, double d) {
    Complex result;
    result.real = a * c - b * d; // Real part
    result.imag = a * d + b * c; // Imaginary part
    return result;
}

__device__ Complex complexDivision(double a, double b, double c, double d) {
    Complex result;
    double denom = c * c + d * d;
    result.real = (a * c + b * d) / denom; // Real part
    result.imag = (b * c - a * d) / denom; // Imaginary part
    return result;
}

__device__ Complex complexSqrt(double a, double b) {
    Complex result;
    double z = sqrt(a * a + b * b);
    result.real = sqrt((z + a) / 2); // Real part
    if (b == 0) {
        if (a < 0) {  // Purely real negative number
        result.real = 0;
        result.imag = sqrt(-a);  // Correct imaginary part
    } else {
        result.imag = 0;
    }
    } else {
        result.imag = (b > 0 ? 1 : -1) * sqrt((z - a) / 2); // Imaginary part
    }
    return result;
}

__device__ void partial_pivot(double A_real[50][50 + 1], double A_imag[50][50 + 1], int n) {
    for (int i = 0; i < n; i++) {
        int pivot_row = i;
        for (int j = i+1; j < n; j++) {
            double A_absJ = sqrt(A_real[j][i] * A_real[j][i] + A_imag[j][i] * A_imag[j][i]);
            double A_absPiv = sqrt(A_real[pivot_row][i] * A_real[pivot_row][i] + A_imag[pivot_row][i] * A_imag[pivot_row][i]);
            if (A_absJ > A_absPiv) {
                pivot_row = j;
            }
        }
        if (pivot_row != i) {
            for (int j = i; j <= n; j++) {
                double temp_real = A_real[i][j];
                double temp_imag = A_imag[i][j];
                A_real[i][j] = A_real[pivot_row][j];
                A_imag[i][j] = A_imag[pivot_row][j];
                A_real[pivot_row][j] = temp_real;
                A_imag[pivot_row][j] = temp_imag; 

            }
        }
        for (int j = i+1; j < n; j++) {
            Complex factor = complexDivision(A_real[j][i], A_imag[j][i], A_real[i][i], A_imag[i][i]);
            for (int k = i; k <= n; k++) {
                Complex res = complexProduct(factor.real, factor.imag, A_real[i][k], A_imag[i][k]);
                A_real[j][k] -= res.real;
                A_imag[j][k] -= res.imag;
            }
        }
    }
}

__device__ void back_substitute(double A_real[50][50 + 1], double A_imag[50][50 + 1], int n, double x_real[50], double x_imag[50]) {
    for (int i = n-1; i >= 0; i--) {
        double sum_real = 0;
        double sum_imag = 0;
        for (int j = i+1; j < n; j++) {
            Complex res = complexProduct(A_real[i][j],A_imag[i][j],x_real[j],x_imag[j]);
            sum_real += res.real;
            sum_imag += res.imag;
        }
        Complex x = complexDivision((A_real[i][n] - sum_real),(A_imag[i][n] - sum_imag),A_real[i][i],A_imag[i][i]);
        x_real[i] = x.real;
        x_imag[i] = x.imag;
    }
}

__global__ void FSAI_kernel(double* const out_real,
                            double* const out_imag,
                            const double* const valA_real,
                            const double* const valA_imag,
                            const int* const rPtrA,
                            const int* const cPtrA,
                            const int* const rPtrG,
                            const int* const cPtrG,
                            const int nfreqs,
                            const int nrows){

    // Thread indices for row and frequency
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int freq = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < nrows && freq < nfreqs){
        double nf = (double)freq;
        int row_start = rPtrG[row];
        int row_end = rPtrG[row+1];
        int m = row_end - row_start;
        int outIdx = freq + nfreqs*(row_start);

        double A_real[50][50+1] = {};
        double A_imag[50][50+1] = {};

        for (int i = 0; i < m; ++i){
            int rIdx = cPtrG[row_start+i];

            for (int j = 0; j <= i; ++j){
                int cIdx = cPtrG[row_start+j];
                int row_startA = rPtrA[rIdx];
                int row_endA = rPtrA[rIdx+1];

                for (int a = 0; a < (row_endA - row_startA); ++a){
                    int cIdxA = cPtrA[row_startA+a];

                    if (cIdxA == cIdx){
                        A_real[i][j] = valA_real[row_startA+a];
                        A_real[j][i] = valA_real[row_startA+a];

                        A_imag[i][j] = nf*valA_imag[row_startA+a];
                        A_imag[j][i] = nf*valA_imag[row_startA+a];
                    }
                }
            }
        }

        // for (int i = 0; i < m; i++) {
        //     A_real[i][i] = 2 * A_real[i][i];
        //     A_imag[i][i] = 2 * A_imag[i][i];
        // }

        A_real[m-1][m] = 1;

        double x_real[50];
        double x_imag[50];

        partial_pivot(A_real, A_imag, m);
        back_substitute(A_real, A_imag, m, x_real, x_imag);

        Complex sqrtRes = complexSqrt(x_real[m-1],x_imag[m-1]);

        for (int i = 0; i < m; ++i){
            Complex res = complexDivision(x_real[i],x_imag[i],sqrtRes.real,sqrtRes.imag);
            out_real[outIdx + i * nfreqs] = res.real;
            out_imag[outIdx + i * nfreqs] = res.imag;
        }
    }
}

void FSAIP_call(double* const d_out_real,
                double* const d_out_imag,
                const double* const d_valA_real,
                const double* const d_valA_imag,
                const int* const d_rPtrA,
                const int* const d_cPtrA,
                const int* const d_rPtrG,
                const int* const d_cPtrG,
                const int nfreqs,
                const int nrows)

{
    dim3 threadsPerBlock(128, 4, 1);
    dim3 numBlocks((nrows + 128 - 1) / 128, (nfreqs + 4 - 1) / 4, 1);

    FSAI_kernel<<<numBlocks, threadsPerBlock>>>(d_out_real, d_out_imag, d_valA_real,
                            d_valA_imag, d_rPtrA, d_cPtrA, d_rPtrG, d_cPtrG, nfreqs,
                            nrows);
}
