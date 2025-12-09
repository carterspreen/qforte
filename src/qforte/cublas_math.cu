// blas_math_gpu.cu
#include "cublas_math.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <stdexcept>
#include <iostream>

// Simple RAII wrapper around a global cuBLAS handle
namespace {
    cublasHandle_t g_cublas_handle = nullptr;

    void check_cublas(cublasStatus_t status, const char* msg) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cuBLAS error in " << msg << ": " << status << "\n";
            throw std::runtime_error(msg);
        }
    }
}

void math_gpu_init()
{
    if (!g_cublas_handle) {
        check_cublas(cublasCreate(&g_cublas_handle), "cublasCreate");
    }
}

void math_gpu_finalize()
{
    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }
}

void math_daxpy_gpu(
    const int n,
    const double alpha,
    const double* x_dev,
    const int incx,
    double* y_dev,
    const int incy)
{
    check_cublas(
        cublasDaxpy(
            g_cublas_handle,
            n,
            &alpha,
            x_dev, incx,
            y_dev, incy),
        "cublasDaxpy");
}

void math_zaxpy_gpu(
    const int n,
    const cuDoubleComplex alpha,
    const cuDoubleComplex* x_dev,
    const int incx,
    cuDoubleComplex* y_dev,
    const int incy)
{
    check_cublas(
        cublasZaxpy(
            g_cublas_handle,
            n,
            &alpha,
            x_dev, incx,
            y_dev, incy),
        "cublasZaxpy");
}