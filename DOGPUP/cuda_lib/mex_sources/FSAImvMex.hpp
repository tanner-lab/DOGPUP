#ifndef FSAImvMex_hpp
#define FSAImvMex_hpp

void FSAImv_call(double* const d_out_real,
                      double* const d_out_imag,
                      const double* const d_x_real,
                      const double* const d_x_imag,
                      const double* const d_valG_real,
                      const double* const d_valG_imag,
                      const int* const d_rPtr,
                      const int* const d_cPtr,
                      const int nnz,
                      const int nrows,
                      const int ncols,
                      const int nfreqs);

#endif