#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <math_constants.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "FSAIP.hpp"

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
    mxGPUArray *out_complex, *out_real, *out_imag;
    mxGPUArray const *valA_complex, *valA_real, *valA_imag, *rPtrA, *cPtrA, *rPtrG, *cPtrG;
    const mwSize *dimsR, *dimsC;
    mwSize* dims;
    int nrows, nfreqs;
    
    // Create GPU arrays from inputs
    valA_complex = mxGPUCopyFromMxArray(prhs[0]);
    valA_real = mxGPUCopyReal(valA_complex);
    valA_imag = mxGPUCopyImag(valA_complex);
    rPtrA = mxGPUCopyFromMxArray(prhs[1]);
    cPtrA = mxGPUCopyFromMxArray(prhs[2]);
    rPtrG = mxGPUCopyFromMxArray(prhs[3]);
    cPtrG = mxGPUCopyFromMxArray(prhs[4]);
    nfreqs = (int)mxGetScalar(prhs[5]);
    // Get dimensions from the inputs
    dimsR = mxGPUGetDimensions(rPtrG);
    dimsC = mxGPUGetDimensions(cPtrG);
    nrows = (int)dimsR[0] - 1;

    dims[0] = (size_t)nfreqs;
    dims[1] = dimsC[0];

    out_real = mxGPUCreateGPUArray(2, dims, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    out_imag = mxGPUCreateGPUArray(2, dims, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);

    // Get pointers to GPU arrays
    double *d_out_real, *d_out_imag;
    int const *d_rPtrA, *d_cPtrA, *d_rPtrG, *d_cPtrG;
    double const *d_valA_real, *d_valA_imag;
    d_out_real = (double*)(mxGPUGetData(out_real));
    d_out_imag = (double*)(mxGPUGetData(out_imag));

    d_rPtrA = (int const*)(mxGPUGetDataReadOnly(rPtrA));
    d_cPtrA = (int const*)(mxGPUGetDataReadOnly(cPtrA));
    d_rPtrG = (int const*)(mxGPUGetDataReadOnly(rPtrG));
    d_cPtrG = (int const*)(mxGPUGetDataReadOnly(cPtrG));

    d_valA_real = (double const*)(mxGPUGetDataReadOnly(valA_real));
    d_valA_imag = (double const*)(mxGPUGetDataReadOnly(valA_imag));
    
    // call .cu
    FSAIP_call(d_out_real, d_out_imag, d_valA_real, d_valA_imag, d_rPtrA, 
               d_cPtrA, d_rPtrG, d_cPtrG, nfreqs, nrows);

    // return to matlab gpuarray
    out_complex = mxGPUCreateComplexGPUArray(out_real, out_imag);
    plhs[0] = mxGPUCreateMxArrayOnCPU(out_complex);

    // Cleanup: free GPU arrays
    mxGPUDestroyGPUArray(out_complex);
    mxGPUDestroyGPUArray(out_real);
    mxGPUDestroyGPUArray(out_imag);
    mxGPUDestroyGPUArray(valA_real);
    mxGPUDestroyGPUArray(valA_imag);
    mxGPUDestroyGPUArray(valA_complex);
    mxGPUDestroyGPUArray(rPtrA);
    mxGPUDestroyGPUArray(cPtrA);
    mxGPUDestroyGPUArray(rPtrG);
    mxGPUDestroyGPUArray(cPtrG);

    // Note: The output arrays are automatically handled, so no need to free them explicitly.
}
