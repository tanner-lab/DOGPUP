#include "mex.h"
#include <stdio.h>
#include "cuda.h"
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Check number of inputs and outputs
    if (nrhs != 0) {
        mexErrMsgIdAndTxt("checkMem_CUDA:nrhs", "0 inputs required.");
    }
    if (nlhs > 2) {
        mexErrMsgIdAndTxt("checkMem_CUDA:nlhs", "Maximum 2 outputs.");
    }

    size_t free_t, total_t;
    cudaMemGetInfo(&free_t, &total_t);

    plhs[0] = mxCreateDoubleScalar((double)free_t);
    plhs[1] = mxCreateDoubleScalar((double)total_t);

}