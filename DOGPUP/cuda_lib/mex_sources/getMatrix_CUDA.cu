#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <math_constants.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

// Constantls
// Tetrahedral Quadrature weights and basis functions at the quadrature points
//  P Keast, Moderate degree tetrahedral quadrature formulas, CMAME 55: 339-348 (1986)
//  O. C. Zienkiewicz, The Finite Element Method,  Sixth Edition,
// From https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tet/quadrature_rules_tet.html
// 3D Tetra
__constant__ double wt_3[5] = {-0.8, 0.45, 0.45, 0.45, 0.45};
__constant__ double N_3[4][5] = {{0.25, 1.0/6.0, 0.5, 1.0/6.0, 1.0/6.0}, \
                                {0.25, 0.5, 1.0/6.0, 1.0/6.0, 1.0/6.0}, \
                                {0.25, 1.0/6.0, 1.0/6.0, 1.0/6.0, 0.5}, \
                                {0.25, 1.0/6.0, 1.0/6.0, 0.5, 1.0/6.0}};
__constant__ double NGrad_3[4][3] = {{-1,-1,-1}, \
                                        {1,0,0}, \
                                        {0,1,0}, \
                                        {0,0,1}};

// 2D Triangle
__constant__ double wt_2[3] = {1.0/3.0, 1.0/3.0, 1.0/3.0};
__constant__ double N_2[3][3] = {{1.0/6.0, 1.0/6.0, 2.0/3.0}, \
                                { 2.0/3.0, 1.0/6.0, 1.0/6.0}, \
                                {1.0/6.0, 2.0/3.0, 1.0/6.0}};

// Kernels
__global__ void fMat_Kernel(int* const row,
                                    int* const col,
                                    double* const kval,
                                    double* const mval,
                                    const int* const elem, // element list in sorted order for each element
                                    const double* const vol,
                                    const double* const gradScale,
                                    const double* const mua,
                                    const double* const kappa,
                                    const int* const face, // face element list in sorted order for each element
                                    const double* const area,
                                    const double* const r, // boundary factor
                                    const int n_elem,
                                    const int n_face)
{
    int elem_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (elem_idx < n_elem) 
    {   
        
        // Element volume
        double vol_e = vol[elem_idx];

        // interpolate optical properties
        double kappa_e[5] = {0.0};
        double mua_e[5] = {0.0};

        #pragma unroll
         for (int q = 0; q < 5; ++q) {
             for (int vtx = 0; vtx < 4; ++vtx)
            {
                int vtx_idx = elem[elem_idx + vtx * n_elem] - 1;
                kappa_e[q] += (N_3[vtx][q] * kappa[vtx_idx]);
                mua_e[q] += (N_3[vtx][q] * mua[vtx_idx]);
            }
         }
            
        double grad_dot[10];
        int g_idx = elem_idx*9;

        int idx = 0;
        
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            #pragma unroll
            for (int j = i; j < 4; ++j)
            {
                double gi0 = gradScale[g_idx + 3*0 + 0] * NGrad_3[i][0]
                    + gradScale[g_idx + 3*1 + 0] * NGrad_3[i][1]
                    + gradScale[g_idx + 3*2 + 0] * NGrad_3[i][2];

                double gj0 = gradScale[g_idx + 3*0 + 0] * NGrad_3[j][0]
                    + gradScale[g_idx + 3*1 + 0] * NGrad_3[j][1]
                    + gradScale[g_idx + 3*2 + 0] * NGrad_3[j][2];
        
                double gi1 = gradScale[g_idx + 3*0 + 1] * NGrad_3[i][0]
                    + gradScale[g_idx + 3*1 + 1] * NGrad_3[i][1]
                    + gradScale[g_idx + 3*2 + 1] * NGrad_3[i][2];

                double gj1 = gradScale[g_idx + 3*0 + 1] * NGrad_3[j][0]
                    + gradScale[g_idx + 3*1 + 1] * NGrad_3[j][1]
                    + gradScale[g_idx + 3*2 + 1] * NGrad_3[j][2];
        
                double gi2 = gradScale[g_idx + 3*0 + 2] * NGrad_3[i][0]
                    + gradScale[g_idx + 3*1 + 2] * NGrad_3[i][1]
                    + gradScale[g_idx + 3*2 + 2] * NGrad_3[i][2];
                
                double gj2 = gradScale[g_idx + 3*0 + 2] * NGrad_3[j][0]
                    + gradScale[g_idx + 3*1 + 2] * NGrad_3[j][1]
                    + gradScale[g_idx + 3*2 + 2] * NGrad_3[j][2];
        
                grad_dot[idx] = gi0 * gj0 + gi1 * gj1 + gi2 * gj2;
        
                idx++;
            }
        }
                

        int idx_out = elem_idx*10;
        idx = 0;

        #pragma unroll
        for (int i = 0; i < 4; ++i) 
        {
            #pragma unroll
            for (int j = i; j < 4; ++j)
            {
                
                double absrp = 0.0;
                double mass = 0.0;
                double stiff = 0;
                
                #pragma unroll
                for (int q = 0; q < 5; ++q)
                {
                    absrp += wt_3[q] * mua_e[q] * N_3[i][q] * N_3[j][q];
                    mass  += wt_3[q] * N_3[i][q] * N_3[j][q];
                    stiff += wt_3[q] * kappa_e[q];
                }
                
                absrp = vol_e * absrp;
                mass = vol_e * mass;
                stiff = vol_e * grad_dot[idx] * stiff;

                idx++;
                // row and column index of values
                row[idx_out] = elem[elem_idx + i * n_elem];
                col[idx_out] = elem[elem_idx + j * n_elem];
                // values 
                kval[idx_out] = absrp + stiff;
                mval[idx_out] = mass;
                idx_out += 1;
            }
        }
    }

    // Boundary Condition
    int face_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (face_idx < n_face) 
    {   
        
        // Element area
        double area_f = area[face_idx];

        // interpolate optical properties
        double r_f[3] = {0.0};

        #pragma unroll
         for (int q = 0; q < 3; ++q) {
             for (int vtx = 0; vtx < 3; ++vtx)
            {
                int vtx_idx = face[face_idx + vtx * n_face] - 1;
                r_f[q] += (N_2[vtx][q] * r[vtx_idx]);
            }
         }
            

        int idx_out = face_idx*6 + n_elem*10;
        
        #pragma unroll
        for (int i = 0; i < 3; ++i) 
        {
            #pragma unroll
            for (int j = i; j < 3; ++j)
            {
                
                double bound = 0.0;
                
                #pragma unroll
                for (int q = 0; q < 3; ++q)
                {
                    bound += wt_2[q] * r_f[q] * N_2[i][q] * N_2[j][q];
                }
                
                bound = area_f * bound;

                // row and column index of values
                row[idx_out] = face[face_idx + i * n_face];
                col[idx_out] = face[face_idx + j * n_face];
                // values 
                kval[idx_out] = bound;
                idx_out += 1;
            }
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Check number of inputs and outputs
    if (nrhs != 8) {
        mexErrMsgIdAndTxt("getMatrix_CUDA:nrhs", "8 inputs required.");
    }
    if (nlhs != 4) {
        mexErrMsgIdAndTxt("getMatrix_CUDA:nlhs", "4 outputs required.");
    }
    

    // Initialize the GPU
    mxInitGPU();

    // Declare variables
    mxGPUArray *row, *col, *kval, *mval;
    mxGPUArray const *elem, *vol, *gradScale, *mua, *kappa, *face, *area, *r;
    const mwSize *dimsInt, *dimsBound;
    mwSize dims_m[1], dims_out[1];
    int n_elem, n_face;
    int const *d_elem, *d_face;
    int *d_row, *d_col;
    double const *d_vol, *d_gradScale, *d_mua, *d_kappa, *d_area, *d_r;
    double *d_kval, *d_mval;
    
    // Create GPU arrays from inputs
    elem = mxGPUCreateFromMxArray(prhs[0]);
    vol = mxGPUCreateFromMxArray(prhs[1]);
    face = mxGPUCreateFromMxArray(prhs[2]);
    area = mxGPUCreateFromMxArray(prhs[3]);
    gradScale = mxGPUCreateFromMxArray(prhs[4]);
    mua = mxGPUCreateFromMxArray(prhs[5]);
    kappa = mxGPUCreateFromMxArray(prhs[6]);
    r = mxGPUCreateFromMxArray(prhs[7]);

    // Get dimensions from the inputs
    dimsInt = mxGPUGetDimensions(elem);
    n_elem = (int)dimsInt[0];
    dimsBound = mxGPUGetDimensions(face);
    n_face = (int)dimsBound[0];
    dims_m[0] = (size_t)(n_elem*10);
    dims_out[0] = (size_t)(n_elem*10 + n_face*6);

    // Prepare output arrays (create GPU arrays)
    row = mxGPUCreateGPUArray(1, dims_out, mxINT32_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    col = mxGPUCreateGPUArray(1, dims_out, mxINT32_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    kval = mxGPUCreateGPUArray(1, dims_out, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);
    mval = mxGPUCreateGPUArray(1, dims_m, mxDOUBLE_CLASS, mxREAL, MX_GPU_INITIALIZE_VALUES);

    // Get pointers to GPU arrays
    d_row = (int*)(mxGPUGetData(row));
    d_col = (int*)(mxGPUGetData(col));
    d_kval = (double*)(mxGPUGetData(kval));
    d_mval = (double*)(mxGPUGetData(mval));

    d_elem = (int const*)(mxGPUGetDataReadOnly(elem));
    d_vol = (double const*)(mxGPUGetDataReadOnly(vol));
    d_face = (int const*)(mxGPUGetDataReadOnly(face));
    d_area = (double const*)(mxGPUGetDataReadOnly(area));
    d_gradScale = (double const*)(mxGPUGetDataReadOnly(gradScale));
    d_mua = (double const*)(mxGPUGetDataReadOnly(mua));
    d_kappa = (double const*)(mxGPUGetDataReadOnly(kappa));
    d_r = (double const*)(mxGPUGetDataReadOnly(r));
    
    // compute forward matrix
    int numBlocks =  (n_elem > n_face) ? (n_elem + 1023) / 1024 : (n_face + 1023) / 1024;
    fMat_Kernel<<<numBlocks, 1024>>>(d_row,d_col,d_kval,d_mval,d_elem,d_vol,d_gradScale,
                                        d_mua,d_kappa,d_face,d_area,d_r,n_elem,n_face);

    // return to matlab array
    plhs[0] = mxGPUCreateMxArrayOnCPU(row);
    plhs[1] = mxGPUCreateMxArrayOnCPU(col);
    plhs[2] = mxGPUCreateMxArrayOnCPU(kval);
    plhs[3] = mxGPUCreateMxArrayOnCPU(mval);

    // Cleanup: free GPU arrays
    mxGPUDestroyGPUArray(row);
    mxGPUDestroyGPUArray(col);
    mxGPUDestroyGPUArray(kval);
    mxGPUDestroyGPUArray(mval);
    mxGPUDestroyGPUArray(elem);
    mxGPUDestroyGPUArray(vol);
    mxGPUDestroyGPUArray(face);
    mxGPUDestroyGPUArray(area);
    mxGPUDestroyGPUArray(gradScale);
    mxGPUDestroyGPUArray(kappa);
    mxGPUDestroyGPUArray(mua);
    mxGPUDestroyGPUArray(r);

}