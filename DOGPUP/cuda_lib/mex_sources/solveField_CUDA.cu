#include "mex.h"
#include "gpu/mxGPUArray.h"
#include <math_constants.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

// Device functions
__device__ double2 complexProduct(double2 a, double2 b) {
    return make_double2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ double2 complexAdd(double2 a, double2 b) {
    return make_double2(a.x + b.x, a.y + b.y);
}


// Cuda Kernels
// FSAI preconditioner * vector
__global__ void FSAImv_kernel(double2* const out,
                                double2* const x,
                                const double2* const valG, 
                                const int* const rPtr,
                                const int* const cPtr,
                                const int nnz,
                                const int nrows,
                                const int ncols,
                                const int nfreqs)
{
    // Thread indices for row, column, and frequency
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int freq = blockIdx.z * blockDim.z + threadIdx.z;

    // Initialize to zero
    double2 temp = make_double2(0.0, 0.0);
    
    if (row < nrows && col < ncols && freq < nfreqs) 
    {
        int outIdx = freq * ncols * nrows + col * nrows + row;
        int row_start = rPtr[row]; // Start index of non-zero elements in the row
        int row_end = rPtr[row + 1]; // End index of non-zero elements in the row

        // Loop over the non-zero elements in the row
        for (int i = row_start; i < row_end; ++i) 
        {
            int col_idx = cPtr[i];
            int xIdx = freq * ncols * nrows + col * nrows + col_idx;
            int gIdx = nnz * freq + i;

            temp = complexAdd(temp,complexProduct(valG[gIdx],x[xIdx]));
        }

        out[outIdx] = temp;
    }
}

// Matrix * vector
__global__ void smv_kernel(double2* const out,
                                double2* const x,
                                const double* const fAxis,
                                const double2* const valA,
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
    double2 temp = make_double2(0.0, 0.0);
    
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
            double2 A = valA[i];
            A.y *= f; // scale imaginary component

            temp = complexAdd(temp,complexProduct(A,x[xIdx]));

        }

        out[outIdx] = temp;
    }
}

__global__ void cuda_sqrt_kernel(double* const a,
                                const int ncols,
                                const int nfreqs)
{
    // Thread indices for row, column, and frequency
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int freq = blockIdx.y * blockDim.y + threadIdx.y;

    
    if (col < ncols && freq < nfreqs) 
    {
        int out_idx = freq * ncols + col;
        a[out_idx] = sqrt(a[out_idx]);
    }
}

__global__ void cuda_sum_reduce_kernel2(double2* const out,
                                            double2* const a,
                                            const int nrows,
                                            const int ncols,
                                            const int nfreqs)
{
    __shared__ double s_val_re[32];
    __shared__ double s_val_im[32];
    for (int i = threadIdx.x; i < 32; i += blockDim.x){
        s_val_re[i] = 0.0;
        s_val_im[i] = 0.0;
    }
    __syncthreads();

    // Thread indices for row, column, and frequency
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y;
    int freq = blockIdx.z;

    int lane = threadIdx.x % 32;
    int warpID = threadIdx.x / 32;

    double val_re = 0.0;
    double val_im = 0.0;

    if (row < nrows && col < ncols && freq < nfreqs) {
        int val_idx = freq * ncols * nrows + col  * nrows + row;

        val_re = a[val_idx].x;
        val_im = a[val_idx].y;
    }

    unsigned mask = 0xffffffff;

    // warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        val_re += __shfl_down_sync(mask, val_re, offset);
        val_im += __shfl_down_sync(mask, val_im, offset);
    }

    // write to shared mem
    if(lane == 0){
        s_val_re[warpID] = val_re;
        s_val_im[warpID] = val_im;
    }
    __syncthreads();

    // block reduction
    if (warpID == 0){
        val_re = s_val_re[lane];
        val_im = s_val_im[lane];
        for (int offset = 32>>1; offset > 0; offset >>= 1) {
            val_re += __shfl_down_sync(mask, val_re, offset);
            val_im += __shfl_down_sync(mask, val_im, offset);
        }
        if (threadIdx.x == 0){
            int out_idx = freq * ncols * gridDim.x + col * gridDim.x + blockIdx.x;
            out[out_idx].x = val_re;
            out[out_idx].y = val_im;
        }
    }
}

__global__ void cuda_sum_reduce_kernel(double* const out,
                                    double* const a,
                                    const int nrows,
                                    const int ncols,
                                    const int nfreqs)
{
    __shared__ double s_val[32];
    for (int i = threadIdx.x; i < 32; i += blockDim.x){
        s_val[i] = 0.0;
    }
    __syncthreads();

    // Thread indices for row, column, and frequency
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y;
    int freq = blockIdx.z;

    int lane = threadIdx.x % 32;
    int warpID = threadIdx.x / 32;

    double val = 0.0;

    if (row < nrows && col < ncols && freq < nfreqs) {
        int val_idx = freq * ncols * nrows + col  * nrows + row;

        val = a[val_idx];
    }

    unsigned mask = 0xffffffff;

    // warp reduction
    for (int offset = 32>>1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }

    // write to shared mem
    if(lane == 0){
        s_val[warpID] = val;
    }
    __syncthreads();

    // block reduction
    if (warpID == 0){
        val = s_val[lane];
        for (int offset = 32>>1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (threadIdx.x == 0){
            int out_idx = freq * ncols * gridDim.x + col * gridDim.x + blockIdx.x;
            out[out_idx] = val;
        }
    }
    
}


__global__ void cuda_norm_reduce_kernel(double* const out,
                                            double2* const a,
                                            const int nrows,
                                            const int ncols,
                                            const int nfreqs)
{
    
    __shared__ double s_val[32];
    for (int i = threadIdx.x; i < 32; i += blockDim.x){
        s_val[i] = 0.0;
    }
    __syncthreads();

    // Thread indices for row, column, and frequency
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y;
    int freq = blockIdx.z;

    int lane = threadIdx.x % 32;
    int warpID = threadIdx.x / 32;

    double val = 0.0;

    if (row < nrows && col < ncols && freq < nfreqs) {
        int val_idx = freq * ncols * nrows + col  * nrows + row;

        val = a[val_idx].x * a[val_idx].x + a[val_idx].y * a[val_idx].y;
    }

    unsigned mask = 0xffffffff;

    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(mask, val, offset);
    }

    if (lane == 0) {
        s_val[warpID] = val;
    }

    __syncthreads();

    // block reduction
    if (warpID == 0){
        val = s_val[lane];
        for (int offset = 32>>1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (threadIdx.x == 0){
            int out_idx = freq * ncols * gridDim.x + col * gridDim.x + blockIdx.x;
            out[out_idx] = val;
        }
    } 
}

__global__ void cuda_max_reduce_kernel(double* const out,
                                double* const a,
                                const int len)
{

    __shared__ double s_val[32];
    for (int i = threadIdx.x; i < 32; i += blockDim.x){
        s_val[i] = -1e30;
    }
    __syncthreads();

    int lane = threadIdx.x % 32;
    int warpID = threadIdx.x / 32;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double val = -1e30;

    if (idx < len) {
        val = a[idx];
    }

    unsigned mask = 0xffffffff;

    for (int offset = 16; offset > 0; offset >>= 1) {
        val = max(val,__shfl_down_sync(mask, val, offset));
    }

    if (lane == 0) {
        s_val[warpID] = val;
    }

    __syncthreads();

    if (warpID == 0){
        val = s_val[lane];
        for (int offset = 32>>1; offset > 0; offset >>= 1) {
            val = max(val,__shfl_down_sync(mask, val, offset));
        }
        if (threadIdx.x == 0){
            out[blockIdx.x] = val;
        }
    }
}

__global__ void cuda_prod_reduce_kernel(double2* const out,
                                            double2* const a,
                                            double2* const b,
                                            const int nrows,
                                            const int ncols,
                                            const int nfreqs)
{
    __shared__ double s_val_re[32];
    __shared__ double s_val_im[32];
    for (int i = threadIdx.x; i < 32; i += blockDim.x){
        s_val_re[i] = 0.0;
        s_val_im[i] = 0.0;
    }
    __syncthreads();
    // Thread indices for row, column, and frequency
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y;
    int freq = blockIdx.z;

    int lane = threadIdx.x % 32;
    int warpID = threadIdx.x / 32;

    double val_re = 0.0;
    double val_im = 0.0;

    if (row < nrows && col < ncols && freq < nfreqs) {
        int val_idx = freq * ncols * nrows + col  * nrows + row;
        int c_idx = freq * ncols + col;
        double2 aa = a[val_idx];
        double2 bb = b[val_idx];
        double2 temp = complexProduct(aa,bb);
        val_re = temp.x;
        val_im = temp.y;
    }

    unsigned mask = 0xffffffff;

    // warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        val_re += __shfl_down_sync(mask, val_re, offset);
        val_im += __shfl_down_sync(mask, val_im, offset);
    }

    // write to shared mem
    if(lane == 0){ // first thread in warp
        s_val_re[warpID] = val_re;
        s_val_im[warpID] = val_im;
    }
    __syncthreads();

    // block reduction
    if (warpID == 0){ // first warp of block
        // read in warp reduction to first warp
        val_re = s_val_re[lane];
        val_im = s_val_im[lane];
        // reduce again, now reducing the entire block (32 * 32 = 1024)
        for (int offset = 32>>1; offset > 0; offset >>= 1) {
            val_re += __shfl_down_sync(mask, val_re, offset);
            val_im += __shfl_down_sync(mask, val_im, offset);
        }
        // write out using first thread of block
        if (threadIdx.x == 0){
            int out_idx = freq * ncols * gridDim.x + col * gridDim.x + blockIdx.x;
            out[out_idx].x = val_re;
            out[out_idx].y = val_im;
        }
    }
}

__global__ void cuda_alph_reduce_kernel(double2* const out,
                                            double2* const a,
                                            double2* const b,
                                            double2* const c,
                                            const int nrows,
                                            const int ncols,
                                            const int nfreqs)
{
    __shared__ double s_val_re[32];
    __shared__ double s_val_im[32];
    for (int i = threadIdx.x; i < 32; i += blockDim.x){
        s_val_re[i] = 0.0;
        s_val_im[i] = 0.0;
    }
    __syncthreads();
    // Thread indices for row, column, and frequency
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y;
    int freq = blockIdx.z;

    int lane = threadIdx.x % 32;
    int warpID = threadIdx.x / 32;

    double val_re = 0.0;
    double val_im = 0.0;

    // calcualte alpha
    if (row < nrows && col < ncols && freq < nfreqs) {
        int val_idx = freq * ncols * nrows + col  * nrows + row;
        int c_idx = freq * ncols + col;
        double2 aa = a[val_idx];
        double2 bb = b[val_idx];
        double2 cc = c[c_idx];
        double2 temp = complexProduct(aa,bb);
        double denom = cc.x * cc.x + cc.y * cc.y;
        val_re = (temp.x * cc.x + temp.y * cc.y)/denom;
        val_im = (temp.y * cc.x - temp.x * cc.y)/denom;
    }

    unsigned mask = 0xffffffff;

    // warp reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        val_re += __shfl_down_sync(mask, val_re, offset);
        val_im += __shfl_down_sync(mask, val_im, offset);
    }

    // write to shared mem
    if(lane == 0){
        s_val_re[warpID] = val_re;
        s_val_im[warpID] = val_im;
    }
    __syncthreads();

    // block reduction
    if (warpID == 0){
        val_re = s_val_re[lane];
        val_im = s_val_im[lane];
        for (int offset = 32>>1; offset > 0; offset >>= 1) {
            val_re += __shfl_down_sync(mask, val_re, offset);
            val_im += __shfl_down_sync(mask, val_im, offset);
        }
        if (threadIdx.x == 0){
            int out_idx = freq * ncols * gridDim.x + col * gridDim.x + blockIdx.x;
            out[out_idx].x = val_re;
            out[out_idx].y = val_im;
        }
    }
}

__global__ void cuda_beta_kernel(double2* const d_beta,
                                    double2* const d_rho,
                                    double2* const d_rho0,
                                    double2* const d_alph,
                                    double2* const d_om,
                                    const int ncols,
                                    const int nfreqs)
{
    // Thread indices for row, column, and frequency
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int freq = blockIdx.y * blockDim.y + threadIdx.y;

    
    if (col < ncols && freq < nfreqs) 
    {
        int out_idx = freq * ncols + col;
        double2 temp = complexProduct(d_rho0[out_idx],d_alph[out_idx]);
        temp = complexProduct(temp,d_om[out_idx]);
        double denom = temp.x * temp.x + temp.y * temp.y;
        double2 rho = d_rho[out_idx];
        d_beta[out_idx].x = (rho.x * temp.x + rho.y * temp.y) / denom;
        d_beta[out_idx].y = (rho.y * temp.x - rho.x * temp.y) / denom;
    }
}

__global__ void cuda_sh_kernel(double2* const h,
                                double2* const s,
                                double2* const x,
                                double2* const h0,
                                double2* const r,
                                double2* const v,
                                double2* const alph,
                                const int nrows,
                                const int ncols,
                                const int nfreqs)
{
    // Thread indices for row, column, and frequency
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int freq = blockIdx.z * blockDim.z + threadIdx.z;

    if (row < nrows && col < ncols && freq < nfreqs) 
    {
        int outIdx = freq * ncols * nrows + col * nrows + row;
        int alphIdx = freq * ncols + col;
        
        double2 alphalph = alph[alphIdx];
        double2 xx = x[outIdx];
        double2 hh = h0[outIdx];
        double2 vv = v[outIdx];
        double2 rr = r[outIdx];
        double denom = alphalph.x * alphalph.x + alphalph.y * alphalph.y;
        hh = make_double2((hh.x * alphalph.x + hh.y * alphalph.y) / denom, (hh.y * alphalph.x - hh.x * alphalph.y) / denom);
        vv = make_double2((vv.x * alphalph.x + vv.y * alphalph.y) / denom, (vv.y * alphalph.x - vv.x * alphalph.y) / denom);
        h[outIdx] = make_double2(xx.x + hh.x, xx.y + hh.y);
        s[outIdx] = make_double2(rr.x - vv.x, rr.y - vv.y);
    }
}

__global__ void cuda_xrp_kernel(double2* const x,
                                double2* const r,
                                double2* const p,
                                double2* const h,
                                double2* const z,
                                double2* const s,
                                double2* const t,
                                double2* const v,
                                double2* const om,
                                const int nrows,
                                const int ncols,
                                const int nfreqs)
{
    // Thread indices for row, column, and frequency
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int freq = blockIdx.z * blockDim.z + threadIdx.z;

    if (row < nrows && col < ncols && freq < nfreqs) 
    {
        int outIdx = freq * ncols * nrows + col * nrows + row;
        int omIdx = freq * ncols + col;
        
        double2 omom = om[omIdx];
        double2 hh = h[outIdx];
        double2 zz = z[outIdx];
        double2 ss = s[outIdx];
        double2 tt = t[outIdx];
        double2 pp = p[outIdx];
        double2 vv = v[outIdx];
 
        zz = complexProduct(zz,omom);
        tt = complexProduct(tt,omom);
        vv = complexProduct(vv,omom);
        x[outIdx] = make_double2(hh.x + zz.x, hh.y + zz.y);
        r[outIdx] = make_double2(ss.x - tt.x, ss.y - tt.y);
        p[outIdx] = make_double2(pp.x - vv.x, pp.y - vv.y);
    }
}

__global__ void cuda_div_kernel2(double2* const out,
                                double2* const a,
                                double2* const b, 
                                const int nrows,
                                const int ncols,
                                const int nfreqs)
{
    // Thread indices for row, column, and frequency
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int freq = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (row < nrows && col < ncols && freq < nfreqs) 
    {
        int outIdx = freq * ncols * nrows + col * nrows + row;
        double2 aa = a[outIdx];
        double2 bb = b[outIdx];
        double denom = bb.x * bb.x + bb.y * bb.y;
        out[outIdx] = make_double2((aa.x * bb.x + aa.y * bb.y) / denom, (aa.y * bb.x - aa.x * bb.y) / denom);
    }
}

__global__ void cuda_div_kernel(double* const out,
                                double* const a,
                                double* const b, 
                                const int nrows,
                                const int ncols,
                                const int nfreqs)
{
    // Thread indices for row, column, and frequency
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int freq = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (row < nrows && col < ncols && freq < nfreqs) 
    {
        int outIdx = freq * ncols * nrows + col * nrows + row;
        out[outIdx] = a[outIdx] / b[outIdx];
    }
}


__global__ void cuda_add_prod_kernel(double2* const out,
                                double2* const a,
                                double2* const b,
                                double2* const c,  
                                const int nrows,
                                const int ncols,
                                const int nfreqs)
{
    // Thread indices for row, column, and frequency
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int freq = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (row < nrows && col < ncols && freq < nfreqs) 
    {
        int outIdx = freq * ncols * nrows + col * nrows + row;
        int cIdx = freq * ncols + col;
        double2 aa = a[outIdx];
        double2 bb = b[outIdx];
        double2 cc = c[cIdx];
        double2 times = make_double2(bb.x * cc.x - bb.y * cc.y, bb.x * cc.y + bb.y * cc.x);
        out[outIdx] = make_double2(aa.x + times.x, aa.y + times.y);
    }
}

// Function wrappers for kernels
void smv_call(double2* const d_out,
                      double2* const d_x,
                      const double* const d_fAxis,
                      const double2* const d_valA,
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

smv_kernel<<<numBlocks, threadsPerBlock>>>(d_out, d_x, d_fAxis,
                                           d_valA, d_rPtr, d_cPtr, nrows, ncols, nfreqs);  

}

void FSAImv_call(double2* const d_out,
                      double2* const d_x,
                      const double2* const d_valG,
                      const int* const d_rPtr,
                      const int* const d_cPtr,
                      const int nnz,
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

FSAImv_kernel<<<numBlocks, threadsPerBlock>>>(d_out, d_x,
d_valG, d_rPtr, d_cPtr, nnz, nrows, ncols, nfreqs);

}

void cuda_sum(double* d_out,
            double* d_a,
            const int nrows,
            const int ncols,
            const int nfreqs)
{
    // Sum by warp reduction
    double *d_tmp1, *d_tmp2;
    size_t tmp_size = (size_t)(((nrows + 1023) / 1024) * ncols * nfreqs) * sizeof(double);
    size_t a_size = (size_t)(nrows * ncols * nfreqs) * sizeof(double);

    cudaMalloc(&d_tmp1, a_size);
    cudaMemcpy(d_tmp1,d_a,a_size,cudaMemcpyDeviceToDevice);
    cudaMalloc(&d_tmp2, tmp_size);

    int curr_rows = nrows;

    while (curr_rows > 1)
    {
        dim3 threadsPerBlock(1024, 1, 1);
        dim3 numBlocks((curr_rows + 1023) / 1024, ncols, nfreqs);
        cuda_sum_reduce_kernel<<<numBlocks, threadsPerBlock>>>(d_tmp2,d_tmp1,curr_rows,ncols,nfreqs); 
        double* tmp = d_tmp1;
        d_tmp1 = d_tmp2;
        d_tmp2 = tmp;
        curr_rows = (curr_rows + 1023) / 1024;
    }

    size_t out_size = (size_t)(ncols * nfreqs) * sizeof(double);
    cudaMemcpy(d_out,d_tmp1,out_size,cudaMemcpyDeviceToDevice);
    cudaFree(d_tmp1);
    cudaFree(d_tmp2);
    
}

void cuda_sum2(double2* d_out,
            double2* d_a,
            const int nrows,
            const int ncols,
            const int nfreqs)
{
    // Sum by warp reduction
    double2 *d_tmp1, *d_tmp2;;
    size_t tmp_size = (size_t)(((nrows + 1023) / 1024) * ncols * nfreqs) * sizeof(double2);
    size_t a_size = (size_t)(nrows * ncols * nfreqs) * sizeof(double2);

    cudaMalloc(&d_tmp1, a_size);
    cudaMemcpy(d_tmp1,d_a,a_size,cudaMemcpyDeviceToDevice);
    cudaMalloc(&d_tmp2, tmp_size);

    int curr_rows = nrows;

    while (curr_rows > 1)
    {
        dim3 threadsPerBlock(1024, 1, 1);
        dim3 numBlocks((curr_rows + 1023) / 1024, ncols, nfreqs);
        cuda_sum_reduce_kernel2<<<numBlocks, threadsPerBlock>>>(d_tmp2,d_tmp1,curr_rows,ncols,nfreqs); 
        double2* tmp = d_tmp1;
        d_tmp1 = d_tmp2;
        d_tmp2 = tmp;
        curr_rows = (curr_rows + 1023) / 1024;
    }

    size_t out_size = (size_t)(ncols * nfreqs) * sizeof(double2);
    cudaMemcpy(d_out,d_tmp1,out_size,cudaMemcpyDeviceToDevice);
    cudaFree(d_tmp1);
    cudaFree(d_tmp2);
  
}

void cuda_max(double* d_out,
            double* d_a,
            const int ncols,
            const int nfreqs)
{
    // Max by warp reduction
    int len = ncols * nfreqs;
    int curr_len = len;
    double *d_tmp1, *d_tmp2;
    size_t tmp_size = (size_t)(((len + 31) /32)*sizeof(double));
    size_t a_size = (size_t)(len * sizeof(double));

    cudaMalloc(&d_tmp1, a_size);
    cudaMemcpy(d_tmp1,d_a,a_size,cudaMemcpyDeviceToDevice);
    cudaMalloc(&d_tmp2, tmp_size);


    while (curr_len > 1)
    {
        int numBlocks = (curr_len + 1023) / 1024;
        cuda_max_reduce_kernel<<<numBlocks, 1024>>>(d_tmp2,d_tmp1,curr_len);               
        double* tmp = d_tmp1;
        d_tmp1 = d_tmp2;
        d_tmp2 = tmp;
        curr_len = numBlocks;
    }

    cudaMemcpy(d_out,d_tmp1,sizeof(double),cudaMemcpyDeviceToDevice);
    cudaFree(d_tmp1);
    cudaFree(d_tmp2);
    
}

void cuda_prod_sum(double2* d_out,
            double2* const d_a,
            double2* const d_b,
            const int nrows,
            const int ncols,
            const int nfreqs)
{

    double2 *d_tmp;
    size_t tmp_size = (size_t)(((nrows + 1023) / 1024) * ncols * nfreqs) * sizeof(double2);
    size_t in_size = (size_t)(nrows * ncols * nfreqs) * sizeof(double2);
    cudaMalloc(&d_tmp, tmp_size);

    dim3 threadsPerBlock(1024, 1, 1);
    dim3 numBlocks((nrows + 1023) / 1024, ncols, nfreqs);
    cuda_prod_reduce_kernel<<<numBlocks, threadsPerBlock>>>(d_tmp,d_a,d_b,nrows,ncols,nfreqs);                

    int curr_rows = (nrows + 1023) / 1024;

    cuda_sum2(d_out,d_tmp,curr_rows,ncols,nfreqs);

    cudaFree(d_tmp);
}

void cuda_alph(double2* d_alph,
            double2* d_v,
            double2* d_rhat0,
            double2* d_rho0,
            const int nrows,
            const int ncols,
            const int nfreqs)
{

    double2 *d_tmp;
    size_t tmp_size = (size_t)(((nrows + 1023) / 1024) * ncols * nfreqs) * sizeof(double2);
    size_t in_size = (size_t)(nrows * ncols * nfreqs) * sizeof(double2);
    cudaMalloc(&d_tmp, tmp_size);
    
    
    dim3 threadsPerBlock_0(1024, 1, 1);
    dim3 numBlocks_0((nrows + 1023) / 1024, ncols, nfreqs);
    cuda_alph_reduce_kernel<<<numBlocks_0, threadsPerBlock_0>>>(d_tmp,d_v,d_rhat0,d_rho0,nrows,ncols,nfreqs);

    int curr_rows = (nrows + 1023) / 1024;

    cuda_sum2(d_alph,d_tmp,curr_rows,ncols,nfreqs);

    cudaFree(d_tmp);
    
}

void cuda_err(double* d_err,
            double2* const d_a,
            const int nrows,
            const int ncols,
            const int nfreqs)
{

    double *d_tmp;
    size_t tmp_size = (size_t)(((nrows + 1023) / 1024) * ncols * nfreqs) * sizeof(double);
    size_t in_size = (size_t)(nrows * ncols * nfreqs) * sizeof(double);
    cudaMalloc(&d_tmp, tmp_size);

    dim3 threadsPerBlock(1024, 1, 1);
    dim3 numBlocks((nrows + 1023) / 1024, ncols, nfreqs);
    cuda_norm_reduce_kernel<<<numBlocks, threadsPerBlock>>>(d_tmp,d_a,nrows,ncols,nfreqs);

    int curr_rows = (nrows + 1023) / 1024;

    cuda_sum(d_err,d_tmp,curr_rows,ncols,nfreqs);
    cudaFree(d_tmp);

    dim3 threadsPerBlock_1(32, 32, 1);
    dim3 numBlocks_1((ncols + 31) / 32, (nfreqs + 31) / 32, 1);
    cuda_sqrt_kernel<<<numBlocks_1, threadsPerBlock_1>>>(d_err,ncols,nfreqs);   

}

void cuda_err_rel(double* d_err,
            double2* const d_a,
            double* const d_b,
            const int nrows,
            const int ncols,
            const int nfreqs)
{
    
    cuda_err(d_err,d_a,nrows,ncols,nfreqs);
    
    int t1x = 1;
    int t1y = 32;
    int t1z = 32;
    dim3 threadsPerBlock(t1x, t1y, t1z);
    dim3 numBlocks((1 + t1x - 1) / t1x, (ncols + t1y - 1) / t1y, (nfreqs + t1z - 1) / t1z);

    cuda_div_kernel<<<numBlocks, threadsPerBlock>>>(d_err,d_err,d_b,1,ncols,nfreqs); 

}

void cuda_sh(double2* const d_h,
                double2* const d_s,
                double2* const d_x,
                double2* const d_h0,
                double2* const d_r,
                double2* const d_v,
                double2* const d_alph,
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

cuda_sh_kernel<<<numBlocks, threadsPerBlock>>>(d_h, d_s, d_x, d_h0, d_r, d_v, d_alph, nrows, ncols, nfreqs);

}

void cuda_xrp(double2* const d_x,
                double2* const d_r,
                double2* const d_p,
                double2* const d_h,
                double2* const d_z,
                double2* const d_s,
                double2* const d_t,
                double2* const d_v,
                double2* const d_om,
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

cuda_xrp_kernel<<<numBlocks, threadsPerBlock>>>(d_x, d_r, d_p, d_h, d_z, d_s, d_t, d_v, d_om, nrows, ncols, nfreqs);

}

void cuda_add_prod(double2* const d_out,
                      double2* const d_a,
                      double2* const d_b,
                      double2* const d_c,
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

cuda_add_prod_kernel<<<numBlocks, threadsPerBlock>>>(d_out, d_a, d_b, d_c,nrows, ncols, nfreqs);  

}

void cuda_om_calc(double2* const d_out,
            double2* const d_a,
            double2* const d_b,
            const int nrows,
            const int ncols,
            const int nfreqs)
{
    double2* d_prod_sum0 = nullptr;
    double2* d_prod_sum1 = nullptr;
    size_t prod_sum_size = (size_t)(ncols * nfreqs) * sizeof(double2);
    cudaMalloc(&d_prod_sum0, prod_sum_size);
    cudaMalloc(&d_prod_sum1, prod_sum_size);

    cuda_prod_sum(d_prod_sum0,d_a,d_b,nrows,ncols,nfreqs);
    cuda_prod_sum(d_prod_sum1,d_b,d_b,nrows,ncols,nfreqs);
    
    int t1x = 1;
    int t1y = 16;
    int t1z = 16;
    dim3 threadsPerBlock(t1x, t1y, t1z);
    dim3 numBlocks(1, (ncols + t1y - 1) / t1y, (nfreqs + t1z - 1) / t1z);

    cuda_div_kernel2<<<numBlocks, threadsPerBlock>>>(d_out,d_prod_sum0,d_prod_sum1,1,ncols,nfreqs);

    cudaFree(d_prod_sum0);
    cudaFree(d_prod_sum1);
}

void cuda_beta_calc(double2* const d_out,
            double2* const d_a,
            double2* const d_b,
            double2* const d_c,
            double2* const d_d,
            const int ncols,
            const int nfreqs)
{

    dim3 threadsPerBlock(32, 32, 1);
    dim3 numBlocks((ncols + 31) / 32, (nfreqs + 31) / 32, 1);

    cuda_beta_kernel<<<numBlocks, threadsPerBlock>>>(d_out,d_a,d_b,d_c,d_d,ncols,nfreqs);

}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // Check number of inputs and outputs
    if (nrhs != 13) {
        mexErrMsgIdAndTxt("solveField_CUDA:nrhs", "13 inputs required.");
    }
    if (nlhs > 5) {
        mexErrMsgIdAndTxt("solveField_CUDA:nlhs", "5 or less outputs required.");
    }

    // Initialize the GPU
    mxInitGPU();

    // Create GPU arrays from inputs
    mxGPUArray *Q = mxGPUCopyFromMxArray(prhs[0]);
    mxGPUArray const *fAxis = mxGPUCreateFromMxArray(prhs[1]);
    mxGPUArray const *rPtrA = mxGPUCreateFromMxArray(prhs[2]);
    mxGPUArray const *cPtrA = mxGPUCreateFromMxArray(prhs[3]);
    mxGPUArray const *valA = mxGPUCreateFromMxArray(prhs[4]);
    mxGPUArray const *rPtrG = mxGPUCreateFromMxArray(prhs[5]);
    mxGPUArray const *cPtrG = mxGPUCreateFromMxArray(prhs[6]);
    mxGPUArray const *valG = mxGPUCreateFromMxArray(prhs[7]);
    mxGPUArray const *rPtrGT = mxGPUCreateFromMxArray(prhs[8]);
    mxGPUArray const *cPtrGT = mxGPUCreateFromMxArray(prhs[9]);
    mxGPUArray const *valGT = mxGPUCreateFromMxArray(prhs[10]);
    double tol = mxGetScalar(prhs[11]);

    // Extract relevant values
    // number of iterations
    int iter = mxGetScalar(prhs[12]);
    // Get dimensions from the inputs
    mwSize n_dims = mxGPUGetNumberOfDimensions(Q);
    const mwSize *dims_in = mxGPUGetDimensions(Q);

    int nrows = (int)dims_in[0];
    int ncols = (int)dims_in[1];
    int nfreqs = (n_dims < 3) ? 1 : (int)dims_in[2]; // handles case where input is 2D

    // assigning dimensions like this generalises 2D matrix to 3D ensuring kernels index correctly
    mwSize dims[3];
    dims[0] = nrows;
    dims[1] = ncols;
    dims[2] = nfreqs;
    mwSize dims2[2];
    dims2[0] = ncols;
    dims2[1] = nfreqs;
    
    const mwSize *lenCG = mxGPUGetDimensions(cPtrG);
    int nnz_g = (int)lenCG[0];
    const mwSize *lenCGT = mxGPUGetDimensions(cPtrGT);
    int nnz_gt = (int)lenCGT[0];

    // Prepare output and intermediate vectors
    // 3D arrays
    mxGPUArray *x = mxGPUCreateGPUArray(3, dims, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_INITIALIZE_VALUES);
    mxGPUArray *r = mxGPUCreateGPUArray(3, dims, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *rhat0 = mxGPUCreateGPUArray(3, dims, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *p = mxGPUCreateGPUArray(3, dims, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *h0 = mxGPUCreateGPUArray(3, dims, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *h1 = mxGPUCreateGPUArray(3, dims, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *v = mxGPUCreateGPUArray(3, dims, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *s = mxGPUCreateGPUArray(3, dims, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *Gs = mxGPUCreateGPUArray(3, dims, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *z = mxGPUCreateGPUArray(3, dims, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *t = mxGPUCreateGPUArray(3, dims, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *Gt = mxGPUCreateGPUArray(3, dims, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    // 2D arrays
    mxGPUArray *err0 = mxGPUCreateGPUArray(2, dims2, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *err = mxGPUCreateGPUArray(2, dims2, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *rho0 = mxGPUCreateGPUArray(2, dims2, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *rho = mxGPUCreateGPUArray(2, dims2, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *alph = mxGPUCreateGPUArray(2, dims2, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *beta = mxGPUCreateGPUArray(2, dims2, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    mxGPUArray *om = mxGPUCreateGPUArray(2, dims2, mxDOUBLE_CLASS, mxCOMPLEX, MX_GPU_DO_NOT_INITIALIZE);
    // Scalar
    mwSize err_size[1];
    err_size[0] = 1;
    mxGPUArray *err_max = mxGPUCreateGPUArray(1, err_size, mxDOUBLE_CLASS, mxREAL, MX_GPU_DO_NOT_INITIALIZE);

    // Sizes
    size_t size_out = (size_t)(nrows * ncols * nfreqs) * sizeof(double2);
    size_t size_reduce = (size_t)(ncols * nfreqs) * sizeof(double2);

    // Get pointers to GPU arrays
    // 3D Arrays
    // In
    double2 *d_Q = (double2*)(mxGPUGetData(Q));
    double const *d_fAxis = (double const*)(mxGPUGetDataReadOnly(fAxis));
    int const *d_rPtrA = (int const*)(mxGPUGetDataReadOnly(rPtrA));
    int const *d_cPtrA = (int const*)(mxGPUGetDataReadOnly(cPtrA));
    double2 const *d_valA = (double2 const*)(mxGPUGetDataReadOnly(valA));
    int const *d_rPtrG = (int const*)(mxGPUGetDataReadOnly(rPtrG));
    int const *d_cPtrG = (int const*)(mxGPUGetDataReadOnly(cPtrG));
    double2 const *d_valG = (double2 const*)(mxGPUGetDataReadOnly(valG));
    int const *d_rPtrGT = (int const*)(mxGPUGetDataReadOnly(rPtrGT));
    int const *d_cPtrGT = (int const*)(mxGPUGetDataReadOnly(cPtrGT));
    double2 const *d_valGT = (double2 const*)(mxGPUGetDataReadOnly(valGT));
    // Out
    double2 *d_x = (double2*)(mxGPUGetData(x));
    double2 *d_r = (double2*)(mxGPUGetData(r));
    double2 *d_rhat0 = (double2*)(mxGPUGetData(rhat0));
    double2 *d_p = (double2*)(mxGPUGetData(p));
    double2 *d_h0 = (double2*)(mxGPUGetData(h0));
    double2 *d_h1 = (double2*)(mxGPUGetData(h1));
    double2 *d_v = (double2*)(mxGPUGetData(v));
    double2 *d_s = (double2*)(mxGPUGetData(s));
    double2 *d_Gs = (double2*)(mxGPUGetData(Gs));
    double2 *d_z = (double2*)(mxGPUGetData(z));
    double2 *d_t = (double2*)(mxGPUGetData(t));
    double2 *d_Gt = (double2*)(mxGPUGetData(Gt));
    // 2D Arrays
    // Out
    double *d_err0 = (double*)(mxGPUGetData(err0));
    double *d_err = (double*)(mxGPUGetData(err));
    double2 *d_rho0 = (double2*)(mxGPUGetData(rho0));
    double2 *d_rho = (double2*)(mxGPUGetData(rho));
    double2 *d_alph = (double2*)(mxGPUGetData(alph));
    double2 *d_beta = (double2*)(mxGPUGetData(beta));
    double2 *d_om = (double2*)(mxGPUGetData(om));
    // Scalar
    // Out
    double *d_err_max = (double*)(mxGPUGetData(err_max));
    double err_max_h;
    

    // r = A * x
    // r = Q - A * x = Q - r
    // r = Q as x = 0
    cudaMemcpy(d_r,d_Q,size_out,cudaMemcpyDeviceToDevice);
    // err0 = sqrt(sum(abs(r .^ 2)))
    cuda_err(d_err0,d_r,nrows,ncols,nfreqs);
    // rhat0 = r
    cudaMemcpy(d_rhat0,d_r,size_out,cudaMemcpyDeviceToDevice);
    // rho0 = sum(rhat0 .* r))
    cuda_prod_sum(d_rho0,d_rhat0,d_r,nrows,ncols,nfreqs);
    // p = r
    cudaMemcpy(d_p,d_r,size_out,cudaMemcpyDeviceToDevice);

    int i = 0;
    int bp;
    
    while (i < iter)
    {

        // hG = G * p
        FSAImv_call(d_h0,d_p,d_valG,d_rPtrG,d_cPtrG,nnz_g,nrows,ncols,nfreqs);

        // h = G^T * G * h
        FSAImv_call(d_h1,d_h0,d_valGT,d_rPtrGT,d_cPtrGT,nnz_gt,nrows,ncols,nfreqs);

        // v = A * h
        smv_call(d_v,d_h1,d_fAxis,d_valA,d_rPtrA,d_cPtrA,nrows,ncols,nfreqs);

        // alph = sum(v .* rhat0) ./ rho0
        cuda_alph(d_alph,d_v,d_rhat0,d_rho0,nrows,ncols,nfreqs);

        // h = x + h ./ alph
        // s = r - v ./ alph
        cuda_sh(d_h0,d_s,d_x,d_h1,d_r,d_v,d_alph,nrows,ncols,nfreqs);

        // err = sqrt(sum(abs(s .^ 2))) ./ err0
        cuda_err_rel(d_err,d_s,d_err0,nrows,ncols,nfreqs);
        // err_max
        cuda_max(d_err_max,d_err,ncols,nfreqs);
        cudaMemcpy(&err_max_h, d_err_max, sizeof(double), cudaMemcpyDeviceToHost);

        if (err_max_h < tol)
        {
            bp = 1;
            cudaMemcpy(d_x,d_h0,size_out,cudaMemcpyDeviceToDevice);
            break;
        }

        // Gs = G * s
        FSAImv_call(d_Gs,d_s,d_valG,d_rPtrG,d_cPtrG,nnz_g,nrows,ncols,nfreqs);

        // z = G^T * Gs = G^T * G * s
        FSAImv_call(d_z,d_Gs,d_valGT,d_rPtrGT,d_cPtrGT,nnz_gt,nrows,ncols,nfreqs);

        // t = A * z
        smv_call(d_t,d_z,d_fAxis,d_valA,d_rPtrA,d_cPtrA,nrows,ncols,nfreqs);

        // Gt = G * t
        FSAImv_call(d_Gt,d_t,d_valG,d_rPtrG,d_cPtrG,nnz_g,nrows,ncols,nfreqs);

        // om = sum(Gs .* Gt) ./ sum(Gt .* Gt)
        cuda_om_calc(d_om,d_Gs,d_Gt,nrows,ncols,nfreqs);

        // x = h + z .* om
        // r = s - t .* om
        // p = p - v .* om
        cuda_xrp(d_x,d_r,d_p,d_h0,d_z,d_s,d_t,d_v,d_om,nrows,ncols,nfreqs);

        // err = sqrt(sum(abs(r .^ 2))) ./ err0
        cuda_err_rel(d_err,d_r,d_err0,nrows,ncols,nfreqs);
        // err_max
        cuda_max(d_err_max,d_err,ncols,nfreqs);
        cudaMemcpy(&err_max_h, d_err_max, sizeof(double), cudaMemcpyDeviceToHost);
        
        if (err_max_h < tol)
        {
            bp = 2;
            break;
        }

        i += 1;

        if (i == iter)
        {
            mexWarnMsgIdAndTxt("solveField_CUDA:convergence","Failed to converge");
            break;
        }

        // rho = sum(r . * rhat0)
        cuda_prod_sum(d_rho,d_rhat0,d_r,nrows,ncols,nfreqs);
        // beta = rho ./ rho0 ./ (alph .* om)
        cuda_beta_calc(d_beta,d_rho,d_rho0,d_alph,d_om,ncols,nfreqs);
        // rho0 = rho;
        cudaMemcpy(d_rho0,d_rho,size_reduce,cudaMemcpyDeviceToDevice);
        // p = r + p .* beta;
        cuda_add_prod(d_p,d_r,d_p,d_beta,nrows,ncols,nfreqs);
    }


    // return to matlab array
    plhs[0] = mxGPUCreateMxArrayOnCPU(x);
    plhs[1] = mxCreateDoubleScalar(i);
    plhs[2] = mxGPUCreateMxArrayOnCPU(err);
    plhs[3] = mxCreateDoubleScalar(err_max_h);
    plhs[4] = mxCreateDoubleScalar(bp);


    // Cleanup: free GPU arrays
    mxGPUDestroyGPUArray(Q);
    mxGPUDestroyGPUArray(fAxis);
    mxGPUDestroyGPUArray(rPtrA);
    mxGPUDestroyGPUArray(cPtrA);
    mxGPUDestroyGPUArray(valA);
    mxGPUDestroyGPUArray(rPtrG);
    mxGPUDestroyGPUArray(cPtrG);
    mxGPUDestroyGPUArray(valG);
    mxGPUDestroyGPUArray(rPtrGT);
    mxGPUDestroyGPUArray(cPtrGT);
    mxGPUDestroyGPUArray(valGT);

    mxGPUDestroyGPUArray(x);
    mxGPUDestroyGPUArray(r);
    mxGPUDestroyGPUArray(rhat0);
    mxGPUDestroyGPUArray(p);
    mxGPUDestroyGPUArray(h0);
    mxGPUDestroyGPUArray(h1);
    mxGPUDestroyGPUArray(v);
    mxGPUDestroyGPUArray(s);
    mxGPUDestroyGPUArray(Gs);
    mxGPUDestroyGPUArray(z);
    mxGPUDestroyGPUArray(t);
    mxGPUDestroyGPUArray(Gt);

    mxGPUDestroyGPUArray(err0);
    mxGPUDestroyGPUArray(err);
    mxGPUDestroyGPUArray(rho0);
    mxGPUDestroyGPUArray(alph);
    mxGPUDestroyGPUArray(beta);
    mxGPUDestroyGPUArray(om);

    mxGPUDestroyGPUArray(err_max);

}
