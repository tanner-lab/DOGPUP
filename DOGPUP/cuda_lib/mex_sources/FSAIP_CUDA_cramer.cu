#include "mex.h"
#include <math.h>  
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "FSAIP.hpp"

__device__ struct Complex {
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
        result.imag = 0;
    } else {
        result.imag = (b > 0 ? 1 : -1) * sqrt((z - a) / 2); // Imaginary part
    }
    return result;
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
                            const int nrows,
                            const int nnz)

{
    // Thread indices for row and frequency
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int freq = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < nrows && freq < nfreqs)
    {
        double nf = (double)freq;
        int row_start = rPtrG[row];
        int row_end = rPtrG[row+1];
        int m = row_end - row_start;
        int outIdx = freq + nfreqs*(row_start);

        double A_real[3][3] = {0.0};
        double A_imag[3][3] = {0.0};

        int test1, test2;

        for (int i = 0; i < m; ++i)
        {
            int rIdx = cPtrG[row_start+i];
            test1 = i;

            for (int j = 0; j <= i; ++j)
            {
                
                int cIdx = cPtrG[row_start+j];
                int row_startA = rPtrA[rIdx];
                int row_endA = rPtrA[rIdx+1];
                test1 = j;

                for (int a = 0; a < (row_endA - row_startA); ++a)
                {
                    int cIdxA = cPtrA[row_startA+a];

                    if (cIdxA == cIdx)
                    {
                        A_real[i][j] = valA_real[row_startA+a];
                        A_real[j][i] = valA_real[row_startA+a];

                        A_imag[i][j] = nf*valA_imag[row_startA+a];
                        A_imag[j][i] = nf*valA_imag[row_startA+a];
                    }
                }
            }
        }

        A_real[0][0] = 2 * A_real[0][0];
        A_real[1][1] = 2 * A_real[1][1];
        A_real[2][2] = 2 * A_real[2][2];

        A_imag[0][0] = 2 * A_imag[0][0];
        A_imag[1][1] = 2 * A_imag[1][1];
        A_imag[2][2] = 2 * A_imag[2][2];

        // complex division
        if (m == 1)
        {
            Complex div = complexDivision(1.0, 0.0, A_real[0][0], A_imag[0][0]);
            Complex sqrt_res = complexSqrt(div.real, div.imag);
            Complex final_res = complexDivision(div.real, div.imag, sqrt_res.real, sqrt_res.imag);

            out_real[outIdx] = final_res.real;
            out_imag[outIdx] = final_res.imag;
        }

        // 2x2 inversion by Cramers rule
        if (m == 2)
        {
            Complex D = complexProduct(A_real[0][0],A_imag[0][0],A_real[1][1],A_imag[1][1]);
            Complex D1 = complexProduct(A_real[0][1],A_imag[0][1],A_real[1][0],A_imag[1][0]);
            D.real = D.real - D1.real;
            D.imag = D.imag - D1.imag;

            Complex x = complexDivision(-A_real[0][1], -A_imag[0][1], D.real, D.imag);
            Complex y = complexDivision(A_real[0][0], A_imag[0][0], D.real, D.imag);
            Complex sqrt_res = complexSqrt(y.real,y.imag);

            x = complexDivision(x.real, x.imag, sqrt_res.real, sqrt_res.imag);
            y = complexDivision(y.real, y.imag, sqrt_res.real, sqrt_res.imag);

            out_real[outIdx] = x.real;
            out_imag[outIdx] = x.imag;

            out_real[outIdx+nfreqs] = y.real;
            out_imag[outIdx+nfreqs] = y.imag;

        }

        // 3x3 inversion by Cramers rule
        if (m == 3)
        {
            Complex D1 = complexProduct(A_real[1][1],A_imag[1][1],A_real[2][2],A_imag[2][2]);
            Complex D11 = complexProduct(A_real[1][2],A_imag[1][2],A_real[2][1],A_imag[2][1]);
            D1.real = D1.real - D11.real;
            D1.imag = D1.imag - D11.imag;
            D1 = complexProduct(A_real[0][0],A_imag[0][0],D1.real,D1.imag);

            Complex D2 = complexProduct(A_real[1][0],A_imag[1][0],A_real[2][2],A_imag[2][2]);
            Complex D22 = complexProduct(A_real[1][2],A_imag[1][2],A_real[2][0],A_imag[2][0]);
            D2.real = D2.real - D22.real;
            D2.imag = D2.imag - D22.imag;
            D2 = complexProduct(A_real[0][1],A_imag[0][1],D2.real,D2.imag);

            Complex D3 = complexProduct(A_real[1][0],A_imag[1][0],A_real[2][1],A_imag[2][1]);
            Complex D33 = complexProduct(A_real[1][1],A_imag[1][1],A_real[2][0],A_imag[2][0]);
            D3.real = D3.real - D33.real;
            D3.imag = D3.imag - D33.imag;
            D3 = complexProduct(A_real[0][2],A_imag[0][2],D3.real,D3.imag);

            Complex D;
            D.real = D1.real - D2.real + D3.real;
            D.imag = D1.imag - D2.imag + D3.real;

            Complex Dx = complexProduct(A_real[0][1],A_imag[0][1],A_real[1][2],A_imag[1][2]);
            Complex D1x = complexProduct(A_real[0][2],A_imag[0][2],A_real[1][1],A_imag[1][1]);
            Dx.real = Dx.real - D1x.real;
            Dx.imag = Dx.imag - D1x.imag;

            Complex Dy = complexProduct(A_real[0][0],A_imag[0][0],A_real[1][2],A_imag[1][2]);
            Complex D1y = complexProduct(A_real[0][2],A_imag[0][2],A_real[1][0],A_imag[1][0]);
            Dy.real = D1y.real - Dy.real;
            Dy.imag = D1y.imag - Dy.imag;

            Complex Dz = complexProduct(A_real[0][0],A_imag[0][0],A_real[1][1],A_imag[1][1]);
            Complex D1z = complexProduct(A_real[1][0],A_imag[1][0],A_real[0][1],A_imag[0][1]);
            Dz.real = Dz.real - D1z.real;
            Dz.imag = Dz.imag - D1z.imag;

            Complex x = complexDivision(Dx.real, Dx.imag, D.real, D.imag);
            Complex y = complexDivision(Dy.real, Dy.imag, D.real, D.imag);
            Complex z = complexDivision(Dz.real, Dz.imag, D.real, D.imag);
            Complex sqrt_res = complexSqrt(z.real,z.imag);

            x = complexDivision(x.real, x.imag, sqrt_res.real, sqrt_res.imag);
            y = complexDivision(y.real, y.imag, sqrt_res.real, sqrt_res.imag);
            z = complexDivision(z.real, z.imag, sqrt_res.real, sqrt_res.imag);

            out_real[outIdx] = x.real;
            out_imag[outIdx] = x.imag;

            out_real[outIdx+nfreqs] = y.real;
            out_imag[outIdx+nfreqs] = y.imag;

            out_real[outIdx+2*nfreqs] = z.real;
            out_imag[outIdx+2*nfreqs] = z.imag;

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
                const int nrows,
                const int nnz)

{
    dim3 threadsPerBlock(128, 4, 1);
    dim3 numBlocks((nrows + 128 - 1) / 128, (nfreqs + 4 - 1) / 4, 1);

    FSAI_kernel<<<numBlocks, threadsPerBlock>>>(d_out_real, d_out_imag, d_valA_real,
                            d_valA_imag, d_rPtrA, d_cPtrA, d_rPtrG, d_cPtrG, nfreqs,
                            nrows, nnz);
}
