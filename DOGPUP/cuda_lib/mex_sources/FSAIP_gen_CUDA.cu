#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <math_constants.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

// Functions for kernel
__device__ double2 complexAdd(double2 a, double2 b) {
    return make_double2(a.x + b.x, a.y + b.y);
}

__device__ double2 complexProduct(double2 a, double2 b) {
    return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ double2 complexDivision(double2 a, double2 b) {
    double denom = b.x * b.x + b.y * b.y;
    return make_double2((a.x * b.x + a.y * b.y) / denom, (a.y * b.x - a.x * b.y) / denom);
}

__device__ double2 complexSqrt(double2 a) {
    double2 result;
    double z = sqrt(a.x * a.x + a.y * a.y);
    double real = sqrt((z + a.x) / 2); // Real part
    if (a.y == 0) {
        if (a.x < 0) {  // Purely real negative number
        return make_double2(0.0, sqrt(-a.x));
    } else {
        return make_double2(real, 0.0);
    }
    } else {
        return make_double2(real, (a.y > 0 ? 1 : -1) * sqrt((z - a.x) / 2));
    }
}

__device__ void partial_pivot(double2 A[30][30 + 1], int n) {
    for (int i = 0; i < n; i++) {
        int pivot_row = i;
        for (int j = i+1; j < n; j++) {
            double A_absJ = sqrt(A[j][i].x * A[j][i].x + A[j][i].y * A[j][i].y);
            double A_absPiv = sqrt(A[pivot_row][i].x * A[pivot_row][i].x + A[pivot_row][i].y * A[pivot_row][i].y);
            if (A_absJ > A_absPiv) {
                pivot_row = j;
            }
        }
        if (pivot_row != i) {
            for (int j = i; j <= n; j++) {
                double2 temp = make_double2(A[i][j].x, A[i][j].y);
                A[i][j].x = A[pivot_row][j].x;
                A[i][j].y = A[pivot_row][j].y;
                A[pivot_row][j].x = temp.x;
                A[pivot_row][j].y = temp.y; 

            }
        }
        for (int j = i+1; j < n; j++) {
            double2 factor = complexDivision(A[j][i], A[i][i]);
            for (int k = i; k <= n; k++) {
                double2 res = complexProduct(factor, A[i][k]);
                A[j][k].x -= res.x;
                A[j][k].y -= res.y;
            }
        }
    }
}

__device__ void back_substitute(double2 A[30][30 + 1], int n, double2 x[30]) {
    for (int i = n-1; i >= 0; i--) {
        double2 sum = make_double2(0.0, 0.0);
        for (int j = i+1; j < n; j++) {
            double2 res = complexProduct(A[i][j],x[j]);
            sum.x -= res.x;
            sum.y -= res.y;
        }
        x[i] = complexDivision(complexAdd(A[i][n],sum),A[i][i]);
    }
}

// Kernel

__global__ void FSAI_kernel(double2* const out,
                            const double2* const valA,
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

        double2 A[30][30+1] = {};

        for (int i = 0; i < m; ++i){
            int rIdx = cPtrG[row_start+i];

            for (int j = 0; j <= i; ++j){
                int cIdx = cPtrG[row_start+j];
                int row_startA = rPtrA[rIdx];
                int row_endA = rPtrA[rIdx+1];

                for (int a = 0; a < (row_endA - row_startA); ++a){
                    int cIdxA = cPtrA[row_startA+a];

                    if (cIdxA == cIdx){
                        double2 valA_temp = valA[row_startA+a];
                        valA_temp.y *= nf;
                        A[i][j] = valA_temp;
                        A[j][i] = valA_temp;

                    }
                }
            }
        }

        A[m-1][m].x = 1.0;

        double2 x[30];

        partial_pivot(A, m);
        back_substitute(A, m, x);

        double2 sqrtRes = complexSqrt(x[m-1]);

        for (int i = 0; i < m; ++i){
            out[outIdx + i * nfreqs] = complexDivision(x[i],sqrtRes);
        }
    }
}

// Wrapper

void FSAIP_call(double2* const d_out,
                const double2* const d_valA,
                const int* const d_rPtrA,
                const int* const d_cPtrA,
                const int* const d_rPtrG,
                const int* const d_cPtrG,
                const int nfreqs,
                const int nrows)

{
    dim3 threadsPerBlock(128, 4, 1);
    dim3 numBlocks((nrows + 128 - 1) / 128, (nfreqs + 4 - 1) / 4, 1);

    FSAI_kernel<<<numBlocks, threadsPerBlock>>>(d_out, d_valA, d_rPtrA, d_cPtrA, d_rPtrG, d_cPtrG, nfreqs,
                            nrows);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Check number of inputs and outputs
    if (nrhs != 6) {
        mexErrMsgIdAndTxt("CUDA:FSAIP_mex:nrhs", "6 inputs required.");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("CUDA:FSAIP_mex:nlhs", "1 output required.");
    }
    

    // Initialize the GPU
    mxInitGPU();

    // Declare variables
    mxGPUArray *out;
    mxGPUArray const *valA, *rPtrA, *cPtrA, *rPtrG, *cPtrG;
    const mwSize *dimsR, *dimsC;
    mwSize dims[2];
    int nrows, nfreqs;
    
    // Create GPU arrays from inputs
    valA = mxGPUCreateFromMxArray(prhs[0]);
    rPtrA = mxGPUCreateFromMxArray(prhs[1]);
    cPtrA = mxGPUCreateFromMxArray(prhs[2]);
    rPtrG = mxGPUCreateFromMxArray(prhs[3]);
    cPtrG = mxGPUCreateFromMxArray(prhs[4]);
    nfreqs = (int)mxGetScalar(prhs[5]);
    // Get dimensions from the inputs
    dimsR = mxGPUGetDimensions(rPtrG);
    dimsC = mxGPUGetDimensions(cPtrG);
    nrows = (int)dimsR[0] - 1;

    dims[0] = (size_t)nfreqs;
    dims[1] = dimsC[0];

    out = mxGPUCreateGPUArray(2, dims, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);

    // Get pointers to GPU arrays
    double2 *d_out;
    int const *d_rPtrA, *d_cPtrA, *d_rPtrG, *d_cPtrG;
    double2 const *d_valA;
    d_out = (double2*)(mxGPUGetData(out));

    d_rPtrA = (int const*)(mxGPUGetDataReadOnly(rPtrA));
    d_cPtrA = (int const*)(mxGPUGetDataReadOnly(cPtrA));
    d_rPtrG = (int const*)(mxGPUGetDataReadOnly(rPtrG));
    d_cPtrG = (int const*)(mxGPUGetDataReadOnly(cPtrG));

    d_valA = (double2 const*)(mxGPUGetDataReadOnly(valA));
    
    // call .cu
    FSAIP_call(d_out, d_valA, d_rPtrA, 
               d_cPtrA, d_rPtrG, d_cPtrG, nfreqs, nrows);

    // return to matlab gpuarray
    plhs[0] = mxGPUCreateMxArrayOnCPU(out);

    // Cleanup: free GPU arrays
    mxGPUDestroyGPUArray(out);
    mxGPUDestroyGPUArray(valA);
    mxGPUDestroyGPUArray(rPtrA);
    mxGPUDestroyGPUArray(cPtrA);
    mxGPUDestroyGPUArray(rPtrG);
    mxGPUDestroyGPUArray(cPtrG);

    // Note: The output arrays are automatically handled, so no need to free them explicitly.
}