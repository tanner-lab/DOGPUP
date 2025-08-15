#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <math_constants.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "FSAImvMex.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Check number of inputs and outputs
    if (nrhs != 4) {
        mexErrMsgIdAndTxt("CUDA:FSAImv:nrhs", "4 inputs required.");
    }
    if (nlhs != 1) {
        mexErrMsgIdAndTxt("CUDA:FSAImv:nlhs", "1 outputs required.");
    }
    

    // Initialize the GPU
    mxInitGPU();

    // Declare variables
    mxGPUArray *out_real, *out_imag, *out_complex;
    mxGPUArray const *x_complex, *x_real, *x_imag, *fAxis, *valG_complex, *valG_real, *valG_imag, *rPtrG, *cPtrG;
    const mwSize *dims, *dimsC;
    int nrows, ncols, nfreqs, nnz;
    int const *d_rPtrG, *d_cPtrG;
    double *d_out_real, *d_out_imag;
    double const *d_x_real, *d_x_imag, *d_valG_real, *d_valG_imag;
    
    // Create GPU arrays from inputs
    x_complex = mxGPUCopyFromMxArray(prhs[0]);
    x_real = mxGPUCopyReal(x_complex);
    x_imag = mxGPUCopyImag(x_complex);
    valG_complex = mxGPUCopyFromMxArray(prhs[1]);
    valG_real = mxGPUCopyReal(valG_complex);
    valG_imag = mxGPUCopyImag(valG_complex);
    rPtrG = mxGPUCopyFromMxArray(prhs[2]);
    cPtrG = mxGPUCopyFromMxArray(prhs[3]);

    // Get dimensions from the inputs
    dims = mxGPUGetDimensions(x_real);
    dimsC = mxGPUGetDimensions(cPtrG);
    nrows = (int)dims[0];
    ncols = (int)dims[1];
    nfreqs = (int)dims[2];
    nnz = (int)dimsC[0];

    // Prepare output arrays (create GPU arrays)
    out_real = mxGPUCreateGPUArray(3, dims, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    out_imag = mxGPUCreateGPUArray(3, dims, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);

    // Get pointers to GPU arrays
    d_out_real = (double*)(mxGPUGetData(out_real));
    d_out_imag = (double*)(mxGPUGetData(out_imag));

    d_rPtrG = (int const*)(mxGPUGetDataReadOnly(rPtrG));
    d_cPtrG = (int const*)(mxGPUGetDataReadOnly(cPtrG));

    d_x_real = (double const*)(mxGPUGetDataReadOnly(x_real));
    d_x_imag = (double const*)(mxGPUGetDataReadOnly(x_imag));
    d_valG_real = (double const*)(mxGPUGetDataReadOnly(valG_real));
    d_valG_imag = (double const*)(mxGPUGetDataReadOnly(valG_imag));
    

    // call .cu
    FSAImv_call(d_out_real, 
              d_out_imag, 
              d_x_real, 
              d_x_imag, 
              d_valG_real, 
              d_valG_imag,
              d_rPtrG, 
              d_cPtrG,
              nnz,
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
    mxGPUDestroyGPUArray(valG_real);
    mxGPUDestroyGPUArray(valG_imag);
    mxGPUDestroyGPUArray(valG_complex);
    mxGPUDestroyGPUArray(rPtrG);
    mxGPUDestroyGPUArray(cPtrG);


    // Note: The output arrays are automatically handled, so no need to free them explicitly.
}