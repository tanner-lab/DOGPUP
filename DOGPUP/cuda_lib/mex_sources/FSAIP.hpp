#ifndef FSAIP_hpp
#define FSAIP_hpp

void FSAIP_call(double* const d_out_real,
                double* const d_out_imag,
                const double* const d_valA_real,
                const double* const d_valA_imag,
                const int* const d_rPtrA,
                const int* const d_cPtrA,
                const int* const d_rPtrG,
                const int* const d_cPtrG,
                const int nfreqs,
                const int nrows);

#endif
    