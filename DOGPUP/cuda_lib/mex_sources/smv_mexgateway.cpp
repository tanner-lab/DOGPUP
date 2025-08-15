#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <math_constants.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "smvMex.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Check number of inputs and outputs
    if (nrhs != 5) {
        mexErrMsgIdAndTxt("CUDA:fwdSubCSRmex:nrhs", "5 inputs required.");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("CUDA:fwdSubCSRmex:nlhs", "1 outputs required.");
    }
    

    // Initialize the GPU
    mxInitGPU();

    // Declare variables
    mxGPUArray *out_real, *out_imag, *out_complex;
    mxGPUArray const *x_complex, *x_real, *x_imag, *fAxis, *valA_complex, *valA_real, *valA_imag, *rPtrA, *cPtrA;
    const mwSize *dims;
    int nrows, ncols, nfreqs;
    int const *d_rPtrA, *d_cPtrA;
    double *d_out_real, *d_out_imag;
    double const *d_x_real, *d_x_imag, *d_fAxis, *d_valA_real, *d_valA_imag;
    
    // Create GPU arrays from inputs
    x_complex = mxGPUCopyFromMxArray(prhs[0]);
    x_real = mxGPUCopyReal(x_complex);
    x_imag = mxGPUCopyImag(x_complex);
    fAxis = mxGPUCopyFromMxArray(prhs[1]);
    valA_complex = mxGPUCopyFromMxArray(prhs[2]);
    valA_real = mxGPUCopyReal(valA_complex);
    valA_imag = mxGPUCopyImag(valA_complex);
    rPtrA = mxGPUCopyFromMxArray(prhs[3]);
    cPtrA = mxGPUCopyFromMxArray(prhs[4]);

    // Get dimensions from the inputs
    dims = mxGPUGetDimensions(x_real);
    nrows = (int)dims[0];
    ncols = (int)dims[1];
    nfreqs = (int)dims[2];

    // Prepare output arrays (create GPU arrays)
    out_real = mxGPUCreateGPUArray(3, dims, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    out_imag = mxGPUCreateGPUArray(3, dims, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);

    // Get pointers to GPU arrays
    d_out_real = (double*)(mxGPUGetData(out_real));
    d_out_imag = (double*)(mxGPUGetData(out_imag));

    d_rPtrA = (int const*)(mxGPUGetDataReadOnly(rPtrA));
    d_cPtrA = (int const*)(mxGPUGetDataReadOnly(cPtrA));

    d_x_real = (double const*)(mxGPUGetDataReadOnly(x_real));
    d_x_imag = (double const*)(mxGPUGetDataReadOnly(x_imag));
    d_fAxis = (double const*)(mxGPUGetDataReadOnly(fAxis));
    d_valA_real = (double const*)(mxGPUGetDataReadOnly(valA_real));
    d_valA_imag = (double const*)(mxGPUGetDataReadOnly(valA_imag));
    

    // call .cu
    smv_call(d_out_real, 
              d_out_imag, 
              d_x_real, 
              d_x_imag, 
              d_fAxis, 
              d_valA_real, 
              d_valA_imag, 
              d_rPtrA, 
              d_cPtrA,
              nrows,
              ncols, 
              nfreqs);


    // return to matlab gpuarray
    out_complex = mxGPUCreateComplexGPUArray(out_real, out_imag);
    plhs[0] = mxGPUCreateMxArrayOnGPU(out_complex);

    // Cleanup: free GPU arrays
    mxGPUDestroyGPUArray(out_complex);
    mxGPUDestroyGPUArray(out_real);
    mxGPUDestroyGPUArray(out_imag);
    mxGPUDestroyGPUArray(x_real);
    mxGPUDestroyGPUArray(x_imag);
    mxGPUDestroyGPUArray(x_complex);
    mxGPUDestroyGPUArray(fAxis);
    mxGPUDestroyGPUArray(valA_real);
    mxGPUDestroyGPUArray(valA_imag);
    mxGPUDestroyGPUArray(valA_complex);
    mxGPUDestroyGPUArray(rPtrA);
    mxGPUDestroyGPUArray(cPtrA);

    // Note: The output arrays are automatically handled, so no need to free them explicitly.
}
