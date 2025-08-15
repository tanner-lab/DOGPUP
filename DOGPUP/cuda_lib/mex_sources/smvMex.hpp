#ifndef smvMex_hpp
#define smvMex_hpp

void smv_call(double* const d_out_real,
                      double* const d_out_imag,
                      const double* const d_x_real,
                      const double* const d_x_imag,
                      const double* const d_fAxis,
                      const double* const d_valA_real,
                      const double* const d_valA_imag,
                      const int* const d_rPtr,
                      const int* const d_cPtr,
                      const int nrows,
                      const int ncols,
                      const int nfreqs);

#endif
    