#pragma once
#include <complex>
#include <cuComplex.h>

// ==============================================
// Original Implementation:
// Only keeping to support non - precomp version
// Should either be updated to use Givens or removed
// ==============================================

__device__ double atomicAdd_double(double* address, double val);

__global__ void apply_individual_nbody1_accumulate_kernel_atomic(
    const cuDoubleComplex coeff, 
    const cuDoubleComplex* d_Cin, 
    cuDoubleComplex* d_Cout, 
    const int* d_sourcea,
    const int* d_targeta,
    const cuDoubleComplex* d_paritya,
    const int* d_sourceb,
    const int* d_targetb,
    const cuDoubleComplex* d_parityb,
    int nbeta_strs_,
    int targeta_size,
    int targetb_size,
    int tensor_size);

extern "C" void apply_individual_nbody1_accumulate_wrapper(
    const cuDoubleComplex coeff, 
    const cuDoubleComplex* d_Cin, 
    cuDoubleComplex* d_Cout, 
    const int* d_sourcea,
    const int* d_targeta,
    const cuDoubleComplex* d_paritya,
    const int* d_sourceb,
    const int* d_targetb,
    const cuDoubleComplex* d_parityb,
    int nbeta_strs_,
    int targeta_size,
    int targetb_size,
    int tensor_size);

// ==============================================
// Scale elements kernel and wrapper (Complex)
// ==============================================

__global__ void scale_elements_kernel(
    cuDoubleComplex* d_Cout,
    const int* d_first, 
    int first_size,
    const int* d_second, 
    int second_size,
    int nbeta_strs_,
    cuDoubleComplex factor);

extern "C" void scale_elements_wrapper_complex(
    cuDoubleComplex* d_Cout,
    const int* d_first, 
    int first_size,
    const int* d_second, 
    int second_size,
    int nbeta_strs_,
    cuDoubleComplex factor);

// ==============================================
// Scale elements kernel and wrapper (Real)
// ==============================================

__global__ void scale_elements_kernel_real(
    double* __restrict__ d_Cout,
    const int* __restrict__ d_first,
    int first_size,
    const int* __restrict__ d_second,
    int second_size,
    int nbeta_strs_,
    double factor);

extern "C" void scale_elements_wrapper_real(
    double* d_Cout,
    const int* d_first,
    int first_size,
    const int* d_second,
    int second_size,
    int nbeta_strs_,
    double factor);

// ==============================================
// In-place Givens update kernels and wrappers (Complex)
// ==============================================

__global__ void inplace_givens_update_rows_kernel(
    cuDoubleComplex* __restrict__ d_Cout,
    const int* __restrict__ sourcea1,
    const int* __restrict__ targeta1,
    const cuDoubleComplex* __restrict__ paritya1,
    const cuDoubleComplex* __restrict__ paritya2,
    int na,
    int nbeta_strs_,
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2);

extern "C" void inplace_givens_update_complex_rows_wrapper(
    cuDoubleComplex* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const cuDoubleComplex* paritya1,
    const cuDoubleComplex* paritya2,
    int na,
    int nbeta_strs_,
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2);

__global__ void inplace_givens_update_cols_kernel(
    cuDoubleComplex* __restrict__ d_Cout,
    const int* __restrict__ sourcea1,
    const int* __restrict__ targeta1,
    const cuDoubleComplex* __restrict__ paritya1,
    const cuDoubleComplex* __restrict__ paritya2,
    const int* __restrict__ sourceb1,
    const int* __restrict__ targetb1,
    const cuDoubleComplex* __restrict__ parityb1,
    const cuDoubleComplex* __restrict__ parityb2,
    int nalpha, 
    int nb,
    int nbeta_strs_,
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2);

__global__ void inplace_givens_update_complex_tiled(
    cuDoubleComplex* __restrict__ d_Cout,
    const int* __restrict__ sourcea1,
    const int* __restrict__ targeta1,
    const cuDoubleComplex* __restrict__ paritya1,
    const cuDoubleComplex* __restrict__ paritya2,
    const int* __restrict__ sourceb1,
    const int* __restrict__ targetb1,
    const cuDoubleComplex* __restrict__ parityb1,
    const cuDoubleComplex* __restrict__ parityb2,
    int nalpha,          // rows
    int nb,              // number of column-pairs
    int nbeta_strs_,
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2);

extern "C" void inplace_givens_update_complex_tiled_wrapper(
    int BX_runtime,                      // pick 32 or 64 (must divide warp multiples)
    cuDoubleComplex* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const cuDoubleComplex* paritya1,
    const cuDoubleComplex* paritya2,
    const int* sourceb1,
    const int* targetb1,
    const cuDoubleComplex* parityb1,
    const cuDoubleComplex* parityb2,
    int nalpha,          // rows
    int nb,              // number of column-pairs
    int nbeta_strs_,     // leading dimension (num columns)
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2);

// ==============================================
// In-place Givens update kernels and wrappers (Real)
// ==============================================

__global__ void inplace_givens_update_rows_kernel_real(
    double* __restrict__ d_Cout,
    const int* __restrict__ sourcea1,      // [na]
    const int* __restrict__ targeta1,      // [na]
    const double* __restrict__ paritya1,   // [na]  (g† leg, row)
    const double* __restrict__ paritya2,   // [na]  (g  leg, row)
    int na,
    int nbeta_strs_,                        // number of columns
    double factor,
    double acc_coeff1,
    double acc_coeff2);

extern "C" void inplace_givens_update_real_rows_wrapper(
    double* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const double* paritya1,
    const double* paritya2,
    int na,
    int nbeta_strs_,
    double factor,
    double acc_coeff1,
    double acc_coeff2);

__global__ void inplace_givens_update_real_tiled(
    double* __restrict__ d_Cout,
    const int* __restrict__ sourcea1,
    const int* __restrict__ targeta1,
    const double* __restrict__ paritya1,
    const double* __restrict__ paritya2,
    const int* __restrict__ sourceb1,
    const int* __restrict__ targetb1,
    const double* __restrict__ parityb1,
    const double* __restrict__ parityb2,
    int nalpha,          // rows
    int nb,              // number of column-pairs
    int nbeta_strs_,     // leading dimension (num columns)
    double factor,
    double acc_coeff1,
    double acc_coeff2);

static void launch_inplace_givens_update_real_tiled(
    double* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const double* paritya1,
    const double* paritya2,
    const int* sourceb1,
    const int* targetb1,
    const double* parityb1,
    const double* parityb2,
    int nalpha,
    int nb,
    int nbeta_strs_,
    double factor,
    double acc_coeff1,
    double acc_coeff2);

extern "C" void inplace_givens_update_real_tiled_wrapper(
    int BX_runtime,                      // pick 32 or 64 (must divide warp multiples)
    double* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const double* paritya1,
    const double* paritya2,
    const int* sourceb1,
    const int* targetb1,
    const double* parityb1,
    const double* parityb2,
    int nalpha,          // rows
    int nb,              // number of column-pairs
    int nbeta_strs_,     // leading dimension (num columns)
    double factor,
    double acc_coeff1,
    double acc_coeff2);

// ==============================================
// LM Apply Array12 Same Spin kernels and wrappers
// ==============================================

__global__ void lm_apply_array12_same_spin_opt_kernel(
    cuDoubleComplex* __restrict__ d_out,
    const cuDoubleComplex* __restrict__ d_C,
    const int* __restrict__ d_dexc,
    const cuDoubleComplex* __restrict__ d_h1e,
    const cuDoubleComplex* __restrict__ d_h2e,
    int states1,
    int states2,
    int ndexc,
    int norbs,
    int inc1,
    int inc2,
    cuDoubleComplex* __restrict__ temp_global); // size states1*states1

extern "C" void lm_apply_array12_same_spin_opt_wrapper(
    cuDoubleComplex* d_out,
    const cuDoubleComplex* d_C,
    const int* d_dexc,
    const cuDoubleComplex* d_h1e,
    const cuDoubleComplex* d_h2e,
    int states1,
    int states2,
    int ndexc,
    int norbs,
    int inc1,
    int inc2,
    cuDoubleComplex* __restrict__ temp_global); // size states1*states1

// ==============================================
// LM Apply Array12 Diff Spin kernels and wrappers
// ==============================================

__global__ void lm_apply_array12_diff_spin_kernel(
    cuDoubleComplex* __restrict__ d_out,
    const cuDoubleComplex* __restrict__ d_C,
    const int* __restrict__ d_adexc,
    const int* __restrict__ d_bdexc,
    const cuDoubleComplex* __restrict__ d_h2e,
    const int* __restrict__ d_signs,
    const int* __restrict__ d_coff,
    const int* __restrict__ d_boff,
    int alpha_states,
    int beta_states,
    int nadexc,
    int nbdexc,
    int norbs,
    int nsig,
    int orbid);

extern "C" void lm_apply_array12_diff_spin_wrapper(
    cuDoubleComplex* d_out,
    const cuDoubleComplex* d_C,
    const int* d_adexc,
    const int* d_bdexc,
    const cuDoubleComplex* d_h2e,
    int alpha_states,
    int beta_states,
    int nadexc,
    int nbdexc,
    int norbs);

// ==============================================
// LM Apply Array12 Diff Spin v2 (New Implementation)
// ==============================================

__global__ void lm_apply_array12_diff_spin_kernel(
    cuDoubleComplex* __restrict__ d_out,
    const cuDoubleComplex* __restrict__ d_C,
    const int* __restrict__ d_ad_offsets, // size norbs2+1
    const int* __restrict__ d_ad_coff,
    const int* __restrict__ d_ad_boff,
    const int* __restrict__ d_ad_sign,
    const int* __restrict__ d_bd_idx2,
    const int* __restrict__ d_bd_orbkl,
    const int* __restrict__ d_bd_parity,
    const cuDoubleComplex* __restrict__ d_h2e,
    int alpha_states,
    int beta_states,
    int nbdexc,
    int norbs);

extern "C" void lm_apply_array12_diff_spin_wrapper_v2(
    cuDoubleComplex* d_out,
    const cuDoubleComplex* d_C,
    const int* d_adexc,
    const int* d_bdexc,
    const cuDoubleComplex* d_h2e,
    int alpha_states,
    int beta_states,
    int nadexc,
    int nbdexc,
    int norbs);

// ==============================================
// LM Apply Array12 Same Spin split & tiled
// ==============================================

__global__ void lm_same_spin_build_temp_tiled_kernel(
    cuDoubleComplex* __restrict__ d_temp,   // [states1 * states1]
    const int* __restrict__ d_dexc,         // [states1 * ndexc * 3]
    const cuDoubleComplex* __restrict__ d_h1e,
    const cuDoubleComplex* __restrict__ d_h2e,
    int states1,
    int ndexc,
    int norbs);

extern "C" void lm_same_spin_build_temp_tiled_wrapper(
    cuDoubleComplex* d_temp,
    const int* d_dexc,
    const cuDoubleComplex* d_h1e,
    const cuDoubleComplex* d_h2e,
    int states1,
    int ndexc,
    int norbs);

__global__ void lm_same_spin_gemv_tiled_kernel(
    cuDoubleComplex* __restrict__ d_out,
    const cuDoubleComplex* __restrict__ d_C,
    const cuDoubleComplex* __restrict__ d_temp,
    int states1,
    int states2,
    int inc1,
    int inc2);


extern "C" void lm_apply_array12_same_spin_opt_gemv_tiled_wrapper(
    cuDoubleComplex* d_out,
    const cuDoubleComplex* d_C,
    const cuDoubleComplex* d_temp,
    int states1,
    int states2,
    int inc1,
    int inc2);

// ==============================================
// New Same Spin Implementation
// ==============================================

extern "C" void lm_apply_array12_same_spin_spmm_wrapper(
    cuDoubleComplex* d_out,                 // [states1 x states2]
    const cuDoubleComplex* d_C,             // [states1 x states2]
    const int* d_dexc,                      // [states1 * ndexc * 3]
    const cuDoubleComplex* d_h1e,
    const cuDoubleComplex* d_h2e,
    int states1,
    int states2,
    int ndexc,
    int norbs,
    int inc1,   // from your existing logic
    int inc2);   // from your existing logic

// ==============================================
// even newer Same Spin Implementation with cuSPARSE
// ==============================================

extern "C" void lm_apply_array12_same_spin_spmm_csr_coalesced_wrapper(
    cuDoubleComplex* d_out,
    const cuDoubleComplex* d_C,
    const int* d_dexc,
    const cuDoubleComplex* d_h1e,
    const cuDoubleComplex* d_h2e,
    int states1,
    int states2,
    int ndexc,
    int norbs,
    int inc1,
    int inc2);