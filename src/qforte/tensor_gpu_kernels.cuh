#ifndef _tensor_gpu_kernels_cuh_
#define _tensor_gpu_kernels_cuh_

#include <cuda_runtime.h>
#include <cuComplex.h>

// Tile dimensions for transpose kernel
#define TILE_DIM 32
#define BLOCK_ROWS 8

// Kernel declarations
__global__ void transposeCoalescedDouble(double *odata, const double *idata, int width, int height);
__global__ void transposeCoalescedComplex(cuDoubleComplex *odata, const cuDoubleComplex *idata, int width, int height);

// Wrapper functions
void launchTransposeDouble(double *d_out, const double *d_in, int width, int height);
void launchTransposeComplex(cuDoubleComplex *d_out, const cuDoubleComplex *d_in, int width, int height);

#endif // _tensor_gpu_kernels_cuh_
