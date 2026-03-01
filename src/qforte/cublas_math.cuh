#ifndef _cublas_math_h_
#define _cublas_math_h_

#include "qforte-def.h"
#include <cublas_v2.h>

// cuBLAS wrapper functions for GPU-accelerated BLAS operations
// All functions expect GPU memory pointers (device pointers)

// Initialize cuBLAS handle
void math_gpu_init();

// Finalize cuBLAS handle
void math_gpu_finalize();

// Function declaration for cuBLAS DAXPY function
// daxpy: y = alpha * x + y
void math_daxpy_gpu(
    const int n,
    const double alpha,
    const double* x_dev,
    const int incx,
    double* y_dev,
    const int incy);

// Function declaration for cuBLAS ZAXPY function (complex version of DAXPY)
// zaxpy: y = alpha * x + y
void math_zaxpy_gpu(
    const int n,
    const cuDoubleComplex alpha,
    const cuDoubleComplex* x_dev,
    const int incx,
    cuDoubleComplex* y_dev,
    const int incy);

#endif // _cublas_math_h_
