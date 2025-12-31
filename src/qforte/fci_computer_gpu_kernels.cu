#include "fci_computer_gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

#include <cusparse.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <limits>
#include <stdexcept>

// ==============================================
// Error checking macros
// =============================================

#define CHECK_CUDA(call) do {                                  \
  cudaError_t err_ = (call);                                   \
  if (err_ != cudaSuccess) {                                   \
    std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__  \
              << " : " << cudaGetErrorString(err_) << "\n";    \
    throw std::runtime_error("CUDA failure");                  \
  }                                                            \
} while(0)

#define CHECK_CUSPARSE(call) do {                              \
  cusparseStatus_t st_ = (call);                               \
  if (st_ != CUSPARSE_STATUS_SUCCESS) {                        \
    std::cerr << "cuSPARSE error " << __FILE__ << ":"          \
              << __LINE__ << " : " << (int)st_ << "\n";        \
    throw std::runtime_error("cuSPARSE failure");              \
  }                                                            \
} while(0)

// ==============================================
// Functor for cuDoubleComplex addition - used with thrust
// ==============================================

struct cuCadd_op {
  __host__ __device__
  cuDoubleComplex operator()(const cuDoubleComplex& a,
                             const cuDoubleComplex& b) const {
    return make_cuDoubleComplex(a.x + b.x, a.y + b.y);
  }
};

// ==============================================
// Original Implementation:
// Only keeping to support non - precomp version
// Should either be updated to use Givens or removed
// ==============================================

// Helper function for atomic add with double precision
__device__ double atomicAdd_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// V2_atomic - thread-safe version using atomicAdd to prevent race conditions
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
    int tensor_size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < targeta_size) {
        int ta_idx = d_targeta[idx] * nbeta_strs_;
        int sa_idx = d_sourcea[idx] * nbeta_strs_;
        
        cuDoubleComplex pref = cuCmul(coeff, d_paritya[idx]);

        if (idy < targetb_size) {
            cuDoubleComplex term = cuCmul(pref, d_parityb[idy]);
            term = cuCmul(term, d_Cin[sa_idx + d_sourceb[idy]]);

            // Thread-safe atomic accumulation
            int output_idx = ta_idx + d_targetb[idy];
            atomicAdd_double(&d_Cout[output_idx].x, term.x);
            atomicAdd_double(&d_Cout[output_idx].y, term.y);
        }
    }
}

void apply_individual_nbody1_accumulate_wrapper(
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
    int tensor_size) 
{
    // 2D grid configuration for the atomic kernel
    dim3 blockSize(16, 16);  // 16x16 = 256 threads per block
    dim3 gridSize((targeta_size + blockSize.x - 1) / blockSize.x,
                  (targetb_size + blockSize.y - 1) / blockSize.y);
    
    apply_individual_nbody1_accumulate_kernel_atomic<<<gridSize, blockSize>>>(
        coeff, d_Cin, d_Cout, d_sourcea, d_targeta, d_paritya, 
        d_sourceb, d_targetb, d_parityb, nbeta_strs_, 
        targeta_size, targetb_size, tensor_size);
   

    // Check for any errors launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch apply_individual_nbody1_accumulate_kernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        throw std::runtime_error("Kernel launch failed");
    }

    // Wait for the kernel to complete and check for errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution failed (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        throw std::runtime_error("Kernel execution failed");
    }
}

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
    cuDoubleComplex factor) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < first_size && j < second_size) {
        int idx = d_first[i] * nbeta_strs_ + d_second[j];
        d_Cout[idx] = cuCmul(d_Cout[idx], factor);
    }
}

extern "C" void scale_elements_wrapper_complex(
    cuDoubleComplex* d_Cout,
    const int* d_first, 
    int first_size,
    const int* d_second, 
    int second_size,
    int nbeta_strs_,
    cuDoubleComplex factor) 
{
    if (first_size <= 0 || second_size <= 0 || nbeta_strs_ <= 0) return;
    // Fast path for identity scaling
    if (cuCreal(factor) == 1.0 && cuCimag(factor) == 0.0) return;

    dim3 blockSize(16, 16);
    dim3 gridSize((first_size + blockSize.x - 1) / blockSize.x, 
                  (second_size + blockSize.y - 1) / blockSize.y);

    scale_elements_kernel<<<gridSize, blockSize>>>(d_Cout, d_first, first_size, d_second, second_size, nbeta_strs_, factor);

    // Check for any errors launching the kernel
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch scale_elements_kernel (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        throw std::runtime_error("Kernel launch failed");
    }

    // Wait for the kernel to complete and check for errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Kernel execution failed (error code " << cudaGetErrorString(err) << ")!" << std::endl;
        throw std::runtime_error("Kernel execution failed");
    }
}

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
    double factor)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < first_size && j < second_size) {
        const int idx = d_first[i] * nbeta_strs_ + d_second[j];
        d_Cout[idx] *= factor;
    }
}

extern "C" void scale_elements_wrapper_real(
    double* d_Cout,
    const int* d_first,
    int first_size,
    const int* d_second,
    int second_size,
    int nbeta_strs_,
    double factor)
{
    if (first_size <= 0 || second_size <= 0 || nbeta_strs_ <= 0) return;
    if (factor == 1.0) return; // noop fast-path

    dim3 blockSize(16, 16);
    dim3 gridSize((first_size + blockSize.x - 1) / blockSize.x,
                  (second_size + blockSize.y - 1) / blockSize.y);

    scale_elements_kernel_real<<<gridSize, blockSize>>>(
        d_Cout, d_first, first_size, d_second, second_size, nbeta_strs_, factor);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch scale_elements_kernel_real ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("scale_elements_kernel_real launch failed");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "scale_elements_kernel_real execution failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("scale_elements_kernel_real execution failed");
    }
}

// ==============================================
// In-place Givens update kernels and wrappers (Complex)
// ==============================================

// Rows-only, coalesced across columns.
// One block processes one (sa1, ta1) pair; threads iterate j across nbeta_strs_.
__global__ void inplace_givens_update_rows_kernel(
    cuDoubleComplex* __restrict__ d_Cout,
    const int* __restrict__ sourcea1,      // [na]
    const int* __restrict__ targeta1,      // [na]
    const cuDoubleComplex* __restrict__ paritya1, // [na]  (g† leg, row)
    const cuDoubleComplex* __restrict__ paritya2, // [na]  (g  leg, row)
    int na,
    int nbeta_strs_,                        // number of columns
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2)
{
    int ia = blockIdx.x;                          // one block per (sa1, ta1) pair
    if (ia >= na) return;

    // Broadcast row-scoped values once per block
    __shared__ int s_sa1, s_ta1;
    __shared__ cuDoubleComplex s_pa1, s_pa2;
    if (threadIdx.x == 0) {
        s_sa1 = sourcea1[ia];
        s_ta1 = targeta1[ia];
        s_pa1 = paritya1[ia];
        s_pa2 = paritya2[ia];
    }
    __syncthreads();

    const int sa1 = s_sa1, ta1 = s_ta1;
    const cuDoubleComplex pa1 = s_pa1, pa2 = s_pa2;
    const int base_u = sa1 * nbeta_strs_;
    const int base_v = ta1 * nbeta_strs_;

    for (int col = threadIdx.x; col < nbeta_strs_; col += blockDim.x) {
        const int idx_u = base_u + col;   // (sa1, col)
        const int idx_v = base_v + col;   // (ta1, col)

        const cuDoubleComplex u0 = d_Cout[idx_u];
        const cuDoubleComplex v0 = d_Cout[idx_v];

        const cuDoubleComplex u_new = cuCadd(cuCmul(factor, u0), cuCmul(acc_coeff2, cuCmul(pa2, v0)));
        const cuDoubleComplex v_new = cuCadd(cuCmul(factor, v0), cuCmul(acc_coeff1, cuCmul(pa1, u0)));

        d_Cout[idx_u] = u_new;
        d_Cout[idx_v] = v_new;
    }
}


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
    cuDoubleComplex acc_coeff2)
{
    if (na == 0 || nbeta_strs_ == 0) return;

    // Choose threads per block: cover columns with good occupancy.
    // Clamp to device limits if you prefer; 256 is a good default.
    int threads = std::min(256, nbeta_strs_);
    // Keep at least one warp
    if (threads < 32) threads = 32;

    dim3 block(threads);
    dim3 grid(na);  // one block per (sa1, ta1) pair

    inplace_givens_update_rows_kernel<<<grid, block>>>(
        d_Cout,
        sourcea1, targeta1, paritya1, paritya2,
        na, nbeta_strs_,
        factor, acc_coeff1, acc_coeff2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch inplace_givens_update_rows_kernel ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_rows_kernel launch failed");
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "inplace_givens_update_rows_kernel execution failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_rows_kernel execution failed");
    }
}

template<int BX>  // number of column-pairs handled per block (e.g., 32)
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
    cuDoubleComplex acc_coeff2)
{
    // Block covers BX consecutive column-pairs starting at ib0
    const int ib0 = blockIdx.x * BX;
    if (ib0 >= nb) return;

    // Thread layout: x = column within the tile, y = row lane inside a small row strip
    const int tx = threadIdx.x;             // [0, BX)
    const int ty = threadIdx.y;             // [0, AY)
    constexpr int AY = 8;                   // small row strip per block
    static_assert(BX % 32 == 0, "Pick BX multiple of warp width for coalescing");

    // Shared: BX col-pair metadata + AY row metadata
    __shared__ int s_sb1[BX], s_tb1[BX];
    __shared__ cuDoubleComplex s_pb1[BX], s_pb2[BX];

    __shared__ int s_sa1[AY], s_ta1[AY];
    __shared__ cuDoubleComplex s_pa1[AY], s_pa2[AY];

    // Load the BX column-pairs (one per tx lane; replicate across ty)
    if (tx + ib0 < nb && ty == 0) {
        const int ib = ib0 + tx;
        s_sb1[tx] = sourceb1[ib];
        s_tb1[tx] = targetb1[ib];
        s_pb1[tx] = parityb1[ib];
        s_pb2[tx] = parityb2[ib];
    }
    __syncthreads();

    // Sweep rows in strips of AY
    for (int ia0 = blockIdx.y * AY; ia0 < nalpha; ia0 += gridDim.y * AY)
    {
        // Cache AY row metadata once
        if (ty < AY && tx == 0) {
            const int ia = ia0 + ty;
            if (ia < nalpha) {
                s_sa1[ty] = sourcea1[ia];
                s_ta1[ty] = targeta1[ia];
                s_pa1[ty] = paritya1[ia];
                s_pa2[ty] = paritya2[ia];
            }
        }
        __syncthreads();

        const int ia = ia0 + ty;
        if (ia < nalpha && tx + ib0 < nb) {
            // Registers for the row
            const int sa1 = s_sa1[ty];
            const int ta1 = s_ta1[ty];
            const cuDoubleComplex pa1 = s_pa1[ty];
            const cuDoubleComplex pa2 = s_pa2[ty];

            // Registers for this column-pair
            // const int ib   = ib0 + tx;
            const int sb1  = s_sb1[tx];
            const int tb1  = s_tb1[tx];
            const cuDoubleComplex pb1 = s_pb1[tx];
            const cuDoubleComplex pb2 = s_pb2[tx];

            const int base_u = sa1 * nbeta_strs_;
            const int base_v = ta1 * nbeta_strs_;

            const int idx_u  = base_u + sb1;  // (sa1, sb1)
            const int idx_v  = base_v + tb1;  // (ta1, tb1)

            // Within a warp, tx varies ⇒ idx_* vary by +1 (contiguous) if sb1/tb1 are consecutive.
            // To ensure that, store column-pairs for a tile as consecutive sb1/tb1 (typical).
            const cuDoubleComplex u0 = d_Cout[idx_u];
            const cuDoubleComplex v0 = d_Cout[idx_v];

            const cuDoubleComplex p1 = cuCmul(pa1, pb1);
            const cuDoubleComplex p2 = cuCmul(pa2, pb2);

            const cuDoubleComplex u_new = cuCadd(cuCmul(factor, u0), cuCmul(acc_coeff2, cuCmul(p2, v0)));
            const cuDoubleComplex v_new = cuCadd(cuCmul(factor, v0), cuCmul(acc_coeff1, cuCmul(p1, u0)));

            d_Cout[idx_u] = u_new;
            d_Cout[idx_v] = v_new;
        }
        __syncthreads();
    }
}

// Internal helper to launch a particular BX specialization
template<int BX>
static void launch_inplace_givens_update_complex_tiled(
    cuDoubleComplex* d_Cout,
    const int* sourcea1,
    const int* targeta1,
    const cuDoubleComplex* paritya1,
    const cuDoubleComplex* paritya2,
    const int* sourceb1,
    const int* targetb1,
    const cuDoubleComplex* parityb1,
    const cuDoubleComplex* parityb2,
    int nalpha,
    int nb,
    int nbeta_strs_,
    cuDoubleComplex factor,
    cuDoubleComplex acc_coeff1,
    cuDoubleComplex acc_coeff2)
{
    if (nalpha == 0 || nb == 0 || nbeta_strs_ == 0) return;

    // Must match the kernel's constexpr AY
    constexpr int AY = 8;

    // Each block covers BX consecutive column-pairs and AY rows (as a strip).
    const int grid_x = (nb + BX - 1) / BX;
    const int grid_y = std::max(1, (nalpha + AY - 1) / AY);

    // Block has BX threads along x (columns in the tile) and AY along y (rows in the strip).
    dim3 block(BX, AY);
    dim3 grid(grid_x, grid_y);

    // Sanity: make sure block size is legal (BX*AY <= 1024 on most GPUs)
    if (block.x * block.y > 1024) {
        throw std::invalid_argument("Block size BX*AY exceeds device limit");
    }

    inplace_givens_update_complex_tiled<BX><<<grid, block>>>(
        d_Cout,
        sourcea1, targeta1, paritya1, paritya2,
        sourceb1, targetb1, parityb1, parityb2,
        nalpha, nb, nbeta_strs_,
        factor, acc_coeff1, acc_coeff2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch inplace_givens_update_complex_tiled<"
                  << BX << "> (" << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_complex_tiled launch failed");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "inplace_givens_update_complex_tiled<" << BX
                  << "> execution failed (" << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_complex_tiled execution failed");
    }
}

// Extern "C" wrapper with runtime BX selection.
// Supported BX values are 32 and 64 by default (add more cases as you like).
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
    cuDoubleComplex acc_coeff2)
{
    if (nalpha == 0 || nb == 0 || nbeta_strs_ == 0) return;

    switch (BX_runtime) {
        case 64:
            launch_inplace_givens_update_complex_tiled<64>(
                d_Cout, sourcea1, targeta1, paritya1, paritya2,
                sourceb1, targetb1, parityb1, parityb2,
                nalpha, nb, nbeta_strs_,
                factor, acc_coeff1, acc_coeff2);
            break;
        case 32:
            launch_inplace_givens_update_complex_tiled<32>(
                d_Cout, sourcea1, targeta1, paritya1, paritya2,
                sourceb1, targetb1, parityb1, parityb2,
                nalpha, nb, nbeta_strs_,
                factor, acc_coeff1, acc_coeff2);
            break;
        default:
            // Fallback or throw—here we fallback to 32 for convenience.
            std::cerr << "Warning: unsupported BX=" << BX_runtime
                      << " — defaulting to BX=32.\n";
            launch_inplace_givens_update_complex_tiled<32>(
                d_Cout, sourcea1, targeta1, paritya1, paritya2,
                sourceb1, targetb1, parityb1, parityb2,
                nalpha, nb, nbeta_strs_,
                factor, acc_coeff1, acc_coeff2);
            break;
    }
}

// ==============================================
// In-place Givens update kernels and wrappers (Real)
// ==============================================

/// One block processes one (sa1, ta1) pair; threads iterate j across nbeta_strs_.
/// pa1, pa2 are row-scoped real parities/scalings (often ±1).
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
    double acc_coeff2)
{
    const int ia = blockIdx.x;  // one block per (sa1, ta1) pair
    if (ia >= na) return;

    // Broadcast row-scoped values once per block
    __shared__ int s_sa1, s_ta1;
    __shared__ double s_pa1, s_pa2;
    if (threadIdx.x == 0) {
        s_sa1 = sourcea1[ia];
        s_ta1 = targeta1[ia];
        s_pa1 = paritya1[ia];
        s_pa2 = paritya2[ia];
    }
    __syncthreads();

    const int sa1 = s_sa1, ta1 = s_ta1;
    const double pa1 = s_pa1, pa2 = s_pa2;

    const int base_u = sa1 * nbeta_strs_;
    const int base_v = ta1 * nbeta_strs_;

    // Precompute per-row scalings to save a couple MULs in the loop
    const double a_row = acc_coeff2 * pa2;
    const double b_row = acc_coeff1 * pa1;

    // Each thread walks columns with stride blockDim.x (coalesced)
    for (int col = threadIdx.x; col < nbeta_strs_; col += blockDim.x) {
        const int idx_u = base_u + col;   // (sa1, col)
        const int idx_v = base_v + col;   // (ta1, col)

        const double u0 = d_Cout[idx_u];
        const double v0 = d_Cout[idx_v];

        const double u_new = factor * u0 + a_row * v0;
        const double v_new = factor * v0 + b_row * u0;

        d_Cout[idx_u] = u_new;
        d_Cout[idx_v] = v_new;
    }
}

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
    double acc_coeff2)
{
    if (na == 0 || nbeta_strs_ == 0) return;

    // Choose threads per block: cover columns with good occupancy.
    int threads = std::min(256, nbeta_strs_);
    if (threads < 32) threads = 32;  // keep at least one warp

    dim3 block(threads);
    dim3 grid(na);  // one block per (sa1, ta1) pair

    inplace_givens_update_rows_kernel_real<<<grid, block>>>(
        d_Cout,
        sourcea1, targeta1, paritya1, paritya2,
        na, nbeta_strs_,
        factor, acc_coeff1, acc_coeff2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch inplace_givens_update_rows_kernel_real ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_rows_kernel_real launch failed");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "inplace_givens_update_rows_kernel_real execution failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_rows_kernel_real execution failed");
    }
}

template<int BX>  // number of column-pairs handled per block (e.g., 32 or 64)
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
    double acc_coeff2)
{
    // Block covers BX consecutive column-pairs starting at ib0
    const int ib0 = blockIdx.x * BX;
    if (ib0 >= nb) return;

    // Thread layout: x = column within the tile, y = row lane inside a small row strip
    const int tx = threadIdx.x;             // [0, BX)
    const int ty = threadIdx.y;             // [0, AY)
    constexpr int AY = 8;                   // small row strip per block
    static_assert(BX % 32 == 0, "Pick BX multiple of warp width for coalescing");

    // Shared: BX col-pair metadata + AY row metadata
    __shared__ int s_sb1[BX], s_tb1[BX];
    __shared__ double s_pb1[BX], s_pb2[BX];

    __shared__ int s_sa1[AY], s_ta1[AY];
    __shared__ double s_pa1[AY], s_pa2[AY];

    // Load the BX column-pairs (one per tx lane; replicate across ty)
    if (tx + ib0 < nb && ty == 0) {
        const int ib = ib0 + tx;
        s_sb1[tx] = sourceb1[ib];
        s_tb1[tx] = targetb1[ib];
        s_pb1[tx] = parityb1[ib];
        s_pb2[tx] = parityb2[ib];
    }
    __syncthreads();

    // Sweep rows in strips of AY
    for (int ia0 = blockIdx.y * AY; ia0 < nalpha; ia0 += gridDim.y * AY)
    {
        // Cache AY row metadata once
        if (ty < AY && tx == 0) {
            const int ia = ia0 + ty;
            if (ia < nalpha) {
                s_sa1[ty] = sourcea1[ia];
                s_ta1[ty] = targeta1[ia];
                s_pa1[ty] = paritya1[ia];
                s_pa2[ty] = paritya2[ia];
            }
        }
        __syncthreads();

        const int ia = ia0 + ty;
        if (ia < nalpha && tx + ib0 < nb) {
            // Registers for the row
            const int sa1 = s_sa1[ty];
            const int ta1 = s_ta1[ty];
            const double pa1 = s_pa1[ty];
            const double pa2 = s_pa2[ty];

            // Registers for this column-pair
            const int sb1  = s_sb1[tx];
            const int tb1  = s_tb1[tx];
            const double pb1 = s_pb1[tx];
            const double pb2 = s_pb2[tx];

            const int base_u = sa1 * nbeta_strs_;
            const int base_v = ta1 * nbeta_strs_;

            const int idx_u  = base_u + sb1;  // (sa1, sb1)
            const int idx_v  = base_v + tb1;  // (ta1, tb1)

            const double u0 = d_Cout[idx_u];
            const double v0 = d_Cout[idx_v];

            // Real "parity" products
            const double p1 = pa1 * pb1;
            const double p2 = pa2 * pb2;

            // Givens-like coupled update (real)
            const double u_new = factor * u0 + acc_coeff2 * (p2 * v0);
            const double v_new = factor * v0 + acc_coeff1 * (p1 * u0);

            d_Cout[idx_u] = u_new;
            d_Cout[idx_v] = v_new;
        }
        __syncthreads();
    }
}

// Internal helper to launch a particular BX specialization
template<int BX>
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
    double acc_coeff2)
{
    if (nalpha == 0 || nb == 0 || nbeta_strs_ == 0) return;

    // Must match the kernel's constexpr AY
    constexpr int AY = 8;

    // Each block covers BX consecutive column-pairs and AY rows (as a strip).
    const int grid_x = (nb + BX - 1) / BX;
    const int grid_y = std::max(1, (nalpha + AY - 1) / AY);

    // Block has BX threads along x (columns in the tile) and AY along y (rows in the strip).
    dim3 block(BX, AY);
    dim3 grid(grid_x, grid_y);

    // Sanity: make sure block size is legal (BX*AY <= 1024 on most GPUs)
    if (block.x * block.y > 1024) {
        throw std::invalid_argument("Block size BX*AY exceeds device limit");
    }

    inplace_givens_update_real_tiled<BX><<<grid, block>>>(
        d_Cout,
        sourcea1, targeta1, paritya1, paritya2,
        sourceb1, targetb1, parityb1, parityb2,
        nalpha, nb, nbeta_strs_,
        factor, acc_coeff1, acc_coeff2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch inplace_givens_update_real_tiled<"
                  << BX << "> (" << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_real_tiled launch failed");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "inplace_givens_update_real_tiled<" << BX
                  << "> execution failed (" << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("inplace_givens_update_real_tiled execution failed");
    }
}

// Extern "C" wrapper with runtime BX selection.
// Supported BX values are 32 and 64 by default (add more cases as you like).
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
    double acc_coeff2)
{
    if (nalpha == 0 || nb == 0 || nbeta_strs_ == 0) return;

    switch (BX_runtime) {
        case 64:
            launch_inplace_givens_update_real_tiled<64>(
                d_Cout, sourcea1, targeta1, paritya1, paritya2,
                sourceb1, targetb1, parityb1, parityb2,
                nalpha, nb, nbeta_strs_,
                factor, acc_coeff1, acc_coeff2);
            break;
        case 32:
            launch_inplace_givens_update_real_tiled<32>(
                d_Cout, sourcea1, targeta1, paritya1, paritya2,
                sourceb1, targetb1, parityb1, parityb2,
                nalpha, nb, nbeta_strs_,
                factor, acc_coeff1, acc_coeff2);
            break;
        default:
            std::cerr << "Warning: unsupported BX=" << BX_runtime
                      << " — defaulting to BX=32.\n";
            launch_inplace_givens_update_real_tiled<32>(
                d_Cout, sourcea1, targeta1, paritya1, paritya2,
                sourceb1, targetb1, parityb1, parityb2,
                nalpha, nb, nbeta_strs_,
                factor, acc_coeff1, acc_coeff2);
            break;
    }
}

// ==============================================
// LM Apply Array12 Same Spin kernel and wrapper
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
    cuDoubleComplex* __restrict__ temp_global)
{
    int s1 = blockIdx.x;         // one s1 array per block
    if (s1 >= states1) return;

    cuDoubleComplex* temp = temp_global + s1 * states1;

    // Initialize temp to zero - parallelize across threads
    for (int i = threadIdx.x; i < states1; i += blockDim.x) {
        temp[i] = make_cuDoubleComplex(0.0, 0.0);
    }
    
    // Ensure all threads have initialized temp before continuing
    __syncthreads();

    const int* cdexc = d_dexc + 3 * s1 * ndexc;

    // let each thread handle an index of s1 and build associated
    // h1e contribution & all h2e contributions
    for (int i = threadIdx.x; i < ndexc; i += blockDim.x) {
        const int* cdex_base = cdexc + 3 * i;
        
        const int s2 = cdex_base[0];
        const int ijshift = cdex_base[1];
        const int parity1 = cdex_base[2];

        cuDoubleComplex h1e_contrib = cuCmul(
            make_cuDoubleComplex(static_cast<double>(parity1), 0.0),
            d_h1e[ijshift]
        );

        // atomics are slow but necessary here due to potential write conflicts
        atomicAdd(&(temp[s2].x), h1e_contrib.x);
        atomicAdd(&(temp[s2].y), h1e_contrib.y);

        // h2e contributions
        const int* cdexc2 = d_dexc + 3 * s2 * ndexc;
        const int h2e_id = ijshift * norbs * norbs;
        const cuDoubleComplex* h2etmp = d_h2e + h2e_id;

        // Iterate through all ndexc entries for s2
        for (int j = 0; j < ndexc; ++j) {
            const int* cdex2_ptr = cdexc2 + 3 * j;
            int target  = cdex2_ptr[0];
            int klshift = cdex2_ptr[1];
            int parity2 = cdex2_ptr[2];
        
            cuDoubleComplex pref = cuCmul(
                make_cuDoubleComplex(double(parity1 * parity2), 0.0),
                h2etmp[klshift]
            );
            atomicAdd(&temp[target].x, pref.x);
            atomicAdd(&temp[target].y, pref.y);
        }
    }

    // need to get everything in temp ready to use
    __syncthreads();

    cuDoubleComplex* cout = d_out + s1 * inc1;

    // let threads handle individual temp contributions to cout
    // This matches the CPU version: for each ii, do zaxpy operation
    // CPU: xptr starts at C_.data() and advances by inc1 each iteration
    for (int ii = threadIdx.x; ii < states1; ii += blockDim.x) {
        cuDoubleComplex ttt = temp[ii];
        const cuDoubleComplex* xptr_ii = d_C + ii * inc1;
        
        // Perform axpy: cout[j*inc2] += ttt * xptr_ii[j*inc2] for j in range(states2)
        // This is a strided vector operation matching math_zaxpy(states2, ttt, xptr, inc2, cout, inc2)
        for (int j = 0; j < states2; ++j) {
            int idx = j * inc2;
            cuDoubleComplex contrib = cuCmul(ttt, xptr_ii[idx]);

            atomicAdd(&cout[idx].x, contrib.x);
            atomicAdd(&cout[idx].y, contrib.y);
        }
    }
}

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
    cuDoubleComplex* __restrict__ temp_global) // size states1*states1
{
    if (states1 == 0 || states2 == 0) return;

    int gridSize  = states1;         // one block per s1
    int blockSize = 256;

    lm_apply_array12_same_spin_opt_kernel<<<gridSize, blockSize>>>(
        d_out, d_C, d_dexc, d_h1e, d_h2e,
        states1, states2, ndexc, norbs, inc1, inc2, temp_global);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Failed to launch lm_apply_array12_same_spin_opt_kernel ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("lm_apply_array12_same_spin_opt_kernel launch failed");
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "lm_apply_array12_same_spin_opt_kernel execution failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("lm_apply_array12_same_spin_opt_kernel execution failed");
    }
}

// ==============================================
// LM Apply Array12 Diff Spin kernel and wrapper
// ==============================================

// 1) compute nsig and alpha channel lists
__global__ void lm_diff_spin_compute_nsig_kernel(
    const int* __restrict__ d_adexc,
    int* __restrict__ d_signs,
    int* __restrict__ d_coff,
    int* __restrict__ d_boff,
    int* __restrict__ d_nsig,
    int alpha_states,
    int nadexc,
    int orbid)
{
    int s1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (s1 >= alpha_states) return;

    // Each thread scans nadexc excitations for this alpha state
    for (int i = 0; i < nadexc; ++i) {
        int idx   = 3 * (s1 * nadexc + i);
        int orbij = d_adexc[idx + 1];
        if (orbij == orbid) {
            int pos = atomicAdd(d_nsig, 1);
            d_signs[pos] = d_adexc[idx + 2]; // ±1
            d_coff[pos]  = d_adexc[idx];     // row in C (alpha_to)
            d_boff[pos]  = s1;               // alpha_from
        }
    }
}

// d_ctemp is column-major: ctemp[beta][isig] with leading dimension nsig_max.
// We pass nsig_current via d_nsig.
__global__ void lm_diff_spin_ctemp_kernel(
    cuDoubleComplex* __restrict__ d_ctemp,
    const cuDoubleComplex* __restrict__ d_C,
    const int* __restrict__ d_signs,
    const int* __restrict__ d_coff,
    const int* __restrict__ d_nsig,
    int beta_states,
    int nsig_max)
{
    int isig = blockIdx.y * blockDim.y + threadIdx.y;
    int beta = blockIdx.x * blockDim.x + threadIdx.x;

    int nsig = *d_nsig;           // current nsig for this orbid

    if (isig >= nsig || beta >= beta_states) return;

    int alpha_to = d_coff[isig];
    const cuDoubleComplex* c_row = d_C + alpha_to * beta_states;

    cuDoubleComplex zsign = make_cuDoubleComplex(double(d_signs[isig]), 0.0);
    cuDoubleComplex contrib = cuCmul(zsign, c_row[beta]);

    // column-major: ctemp[beta][isig] = contrib
    d_ctemp[beta * nsig_max + isig] = contrib;
}

// 3) beta-loop + GEMV + scatter
__global__ void lm_diff_spin_vtemp_kernel(
    cuDoubleComplex* __restrict__ d_out,
    const cuDoubleComplex* __restrict__ d_ctemp,
    const int* __restrict__ d_bdexc,
    const cuDoubleComplex* __restrict__ d_h2e,
    const int* __restrict__ d_boff,
    const int* __restrict__ d_nsig,
    int alpha_states,
    int beta_states,
    int nbdexc,
    int norbs,
    int nsig_max,
    int orbid)
{
    int s2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (s2 >= beta_states) return;

    int nsig = *d_nsig;
    if (nsig == 0) return;

    const int norbs2 = norbs * norbs;
    const cuDoubleComplex* tmperi = d_h2e + orbid * norbs2;

    cuDoubleComplex* tmpout = d_out + s2; // column s2

    // Loop over beta excitations bdexc[s2][j]
    for (int j = 0; j < nbdexc; ++j) {
        int base   = 3 * (s2 * nbdexc + j);
        int idx2   = d_bdexc[base + 0]; // column index in ctemp
        int orbkl  = d_bdexc[base + 1]; // index in tmperi
        int parity = d_bdexc[base + 2]; // ±1

        cuDoubleComplex ttt = tmperi[orbkl];
        if (parity == -1) {
            ttt.x = -ttt.x;
            ttt.y = -ttt.y;
        }

        const cuDoubleComplex* cctmp = d_ctemp + idx2 * nsig_max;

        // Accumulate contributions for all isig into out[boff[isig], s2]
        for (int isig = 0; isig < nsig; ++isig) {
            cuDoubleComplex contrib = cuCmul(ttt, cctmp[isig]);
            int alpha_from = d_boff[isig];
            int out_idx = beta_states * alpha_from;
            tmpout[out_idx].x += contrib.x;
            tmpout[out_idx].y += contrib.y;
        }
    }
}

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
    int norbs)
{
    const int norbs2     = norbs * norbs;
    const int nadexc_tot = alpha_states * nadexc;
    const int max_nsig   = nadexc_tot; // safe upper bound

    int* d_signs;
    int* d_coff;
    int* d_boff;
    int* d_nsig;
    cuDoubleComplex* d_ctemp;

    cudaMalloc(&d_signs, nadexc_tot * sizeof(int));
    cudaMalloc(&d_coff,  nadexc_tot * sizeof(int));
    cudaMalloc(&d_boff,  nadexc_tot * sizeof(int));
    cudaMalloc(&d_nsig,  sizeof(int));

    cudaMalloc(&d_ctemp, max_nsig * beta_states * sizeof(cuDoubleComplex));

    int threadsPerBlock = 256;

    for (int orbid = 0; orbid < norbs2; ++orbid) {
        cudaMemset(d_nsig, 0, sizeof(int));

        // 1) compute nsig + lists
        int blocks_alpha = (alpha_states + threadsPerBlock - 1) / threadsPerBlock;
        lm_diff_spin_compute_nsig_kernel<<<blocks_alpha, threadsPerBlock>>>(
            d_adexc, d_signs, d_coff, d_boff, d_nsig,
            alpha_states, nadexc, orbid);

        // no need to sync here; next kernels see updated d_nsig

        // 2) build ctemp with 2D grid
        dim3 block2d(16, 16);
        dim3 grid2d(
            (beta_states + block2d.x - 1) / block2d.x,
            (max_nsig    + block2d.y - 1) / block2d.y);
        lm_diff_spin_ctemp_kernel<<<grid2d, block2d>>>(
            d_ctemp, d_C, d_signs, d_coff, d_nsig,
            beta_states, max_nsig);

        // 3) beta loop + accumulate into d_out
        int blocks_beta = (beta_states + threadsPerBlock - 1) / threadsPerBlock;
        lm_diff_spin_vtemp_kernel<<<blocks_beta, threadsPerBlock>>>(
            d_out, d_ctemp, d_bdexc, d_h2e, d_boff, d_nsig,
            alpha_states, beta_states, nbdexc, norbs, max_nsig, orbid);
    }

    cudaDeviceSynchronize();

    cudaFree(d_signs);
    cudaFree(d_coff);
    cudaFree(d_boff);
    cudaFree(d_nsig);
    cudaFree(d_ctemp);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "lm_apply_array12_diff_spin_wrapper failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("lm_apply_array12_diff_spin_wrapper execution failed");
    }
}

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
    int norbs)
{
    int s1 = blockIdx.x;  // one block per s1
    if (s1 >= states1) return;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    extern __shared__ cuDoubleComplex sh_temp[];  // size = blockDim.x

    const int norbs2 = norbs * norbs;

    // pointer to dexc for this s1
    const int* cdexc_s1 = d_dexc + 3 * s1 * ndexc;

    // loop over temp tiles [tileStart, tileStart + tileLen)
    for (int tileStart = 0; tileStart < states1; tileStart += blockSize) {

        int tileLen = min(blockSize, states1 - tileStart);

        // 1) zero out the shared tile
        if (tid < tileLen) {
            sh_temp[tid] = make_cuDoubleComplex(0.0, 0.0);
        }
        __syncthreads();

        // 2) walk excitations and add contributions for targets in this tile
        //    each thread handles a subset of i in 0..ndexc-1
        for (int i = tid; i < ndexc; i += blockSize) {

            const int* cd1 = cdexc_s1 + 3 * i;
            int s2      = cd1[0];
            int ijshift = cd1[1];
            int parity1 = cd1[2];

            // h1e contribution: temp[s2] += parity1 * h1e[ijshift]
            if (s2 >= tileStart && s2 < tileStart + tileLen) {
                int local = s2 - tileStart;
                cuDoubleComplex contrib = cuCmul(
                    make_cuDoubleComplex((double)parity1, 0.0),
                    d_h1e[ijshift]
                );
                atomicAdd(&(sh_temp[local].x), contrib.x);
                atomicAdd(&(sh_temp[local].y), contrib.y);
            }

            // h2e contributions: loop over cdexc2 for this s2
            const int* cdexc2 = d_dexc + 3 * s2 * ndexc;
            const cuDoubleComplex* h2etmp = d_h2e + ijshift * norbs2;

            for (int j = 0; j < ndexc; ++j) {
                const int* cd2 = cdexc2 + 3 * j;
                int target  = cd2[0];
                int klshift = cd2[1];
                int parity2 = cd2[2];

                if (target >= tileStart && target < tileStart + tileLen) {
                    int local = target - tileStart;

                    int parity = parity1 * parity2;
                    cuDoubleComplex val = h2etmp[klshift];
                    if (parity == -1) {
                        val.x = -val.x;
                        val.y = -val.y;
                    }

                    atomicAdd(&(sh_temp[local].x), val.x);
                    atomicAdd(&(sh_temp[local].y), val.y);
                }
            }
        }

        __syncthreads();

        // 3) flush this tile to global memory (no atomics needed)
        //    Each (s1, target) is only ever written by this block
        for (int local = tid; local < tileLen; local += blockSize) {
            int globalTarget = tileStart + local;
            int gidx = s1 * states1 + globalTarget;

            cuDoubleComplex val = sh_temp[local];

            // If d_temp was cudaMemset to 0 before the kernel,
            // you can just assign:
            // d_temp[gidx] = val;
            // To be safe for reuse, you can also do +=:
            cuDoubleComplex old = d_temp[gidx];
            old.x += val.x;
            old.y += val.y;
            d_temp[gidx] = old;
        }

        __syncthreads(); // ensure tile is done before reusing sh_temp
    }
}

extern "C" void lm_same_spin_build_temp_tiled_wrapper(
    cuDoubleComplex* d_temp,
    const int* d_dexc,
    const cuDoubleComplex* d_h1e,
    const cuDoubleComplex* d_h2e,
    int states1,
    int ndexc,
    int norbs)
{
    if (states1 == 0) return;

    // zero temp once; kernel adds tile-wise
    cudaError_t err = cudaMemset(d_temp, 0, states1 * states1 * sizeof(cuDoubleComplex));
    if (err != cudaSuccess) {
        std::cerr << "cudaMemset(d_temp) failed: " << cudaGetErrorString(err) << "\n";
        throw std::runtime_error("cudaMemset(d_temp) failed");
    }

    int blockSize = 256;                // tunable; also tile size
    int gridSize  = states1;            // one block per s1
    size_t shmem  = blockSize * sizeof(cuDoubleComplex);

    lm_same_spin_build_temp_tiled_kernel<<<gridSize, blockSize, shmem>>>(
        d_temp, d_dexc, d_h1e, d_h2e,
        states1, ndexc, norbs);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "lm_same_spin_build_temp_tiled_kernel launch failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("lm_same_spin_build_temp_tiled_kernel launch failed");
    }
}

__global__ void lm_same_spin_gemv_tiled_kernel(
    cuDoubleComplex* __restrict__ d_out,        // [states1 x states2]
    const cuDoubleComplex* __restrict__ d_C,    // [states1 x states2]
    const cuDoubleComplex* __restrict__ d_temp, // [states1 x states1]
    int states1,
    int states2,
    int inc1,
    int inc2)
{
    int s1 = blockIdx.x;   // one block per s1 (row of target -> Cout)
    if (s1 >= states1) return;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    extern __shared__ cuDoubleComplex sh_temp[];  // tile buffer (size = blockDim.x)

    // Solving for: out(s1, j) += sum_{ii=0}^{states1-1} temp(s1, ii) * C(ii, j)
    // Tile over ii dimension in chunks of size blockSize
    for (int iiBase = 0; iiBase < states1; iiBase += blockSize) {

        int tileCount = min(blockSize, states1 - iiBase);

        // load this tile of temp into shared memory
        if (tid < tileCount) {
            sh_temp[tid] = d_temp[s1 * states1 + (iiBase + tid)];
        }
        __syncthreads();

        // each thread handles j = tid, tid+blockSize, ... (subset of columns)
        for (int j = tid; j < states2; j += blockSize) {

            cuDoubleComplex partial = make_cuDoubleComplex(0.0, 0.0);

            // accumulate the dot product part of shared temp * C[ii_base : ii_base + tileCount, j]
            for (int local = 0; local < tileCount; ++local) {
                int ii = iiBase + local;

                cuDoubleComplex tval = sh_temp[local];
                
                int cidx = ii * inc1 + j * inc2;
                cuDoubleComplex cval = d_C[cidx];

                cuDoubleComplex prod = cuCmul(tval, cval);
                partial.x += prod.x;
                partial.y += prod.y;
            }

            // accumulate into out(s1, j)
            int out_idx = s1 * inc1 + j * inc2;
            cuDoubleComplex prev = d_out[out_idx];
            prev.x += partial.x;
            prev.y += partial.y;
            d_out[out_idx] = prev;
        }

        __syncthreads();
    }
}

extern "C" void lm_apply_array12_same_spin_opt_gemv_tiled_wrapper(
    cuDoubleComplex* d_out,
    const cuDoubleComplex* d_C,
    const cuDoubleComplex* d_temp,
    int states1,
    int states2,
    int inc1,
    int inc2)
{
    if (states1 == 0 || states2 == 0) return;

    int blockSize = 256;                     // tune
    int gridSize  = states1;                 // one block per s1
    size_t shmem  = blockSize * sizeof(cuDoubleComplex);

    lm_same_spin_gemv_tiled_kernel<<<gridSize, blockSize, shmem>>>(
        d_out, d_C, d_temp,
        states1, states2, inc1, inc2);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "lm_same_spin_gemv_tiled_kernel launch failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("lm_same_spin_gemv_tiled_kernel launch failed");
    }
}


// ==============================================
// Diff spin new implementation
// ==============================================

// ---------------------------------------------------------
// Utility: atomic add for cuDoubleComplex
// ---------------------------------------------------------
__device__ inline void atomicAdd_cuDoubleComplex(cuDoubleComplex* addr,
                                                 const cuDoubleComplex val)
{
    atomicAdd(&addr->x, val.x);
    atomicAdd(&addr->y, val.y);
}

// ---------------------------------------------------------
// 1) Count alpha excitations per orbid: d_counts[orbid]
//    adexc layout: for each (s1, i) with idx = s1*nadexc + i
//      [ coff, orbij, sign ]
// ---------------------------------------------------------
__global__ void count_alpha_excitations_per_orbid_kernel(
    const int* __restrict__ d_adexc,
    int alpha_states,
    int nadexc,
    int norbs2,
    int* __restrict__ d_counts)
{
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int total = alpha_states * nadexc;
    if (flat >= total) return;

    int base  = 3 * flat;
    int orbij = d_adexc[base + 1];
    if (orbij >= 0 && orbij < norbs2) {
        atomicAdd(&d_counts[orbij], 1);
    }
}

// ---------------------------------------------------------
// 2) Fill CSR arrays for alpha excitations:
//
// d_ad_offsets: size norbs2+1 (CSR offsets)
// d_cursors:    size norbs2, initialized to d_ad_offsets[0..norbs2-1]
// d_ad_coff:    size total_ex (coff)
// d_ad_boff:    size total_ex (boff = alpha state index)
// d_ad_sign:    size total_ex (±1)
// ---------------------------------------------------------
__global__ void fill_alpha_csr_from_adexc_kernel(
    const int* __restrict__ d_adexc,
    int alpha_states,
    int nadexc,
    int norbs2,
    const int* __restrict__ d_ad_offsets,
    int* __restrict__ d_cursors,
    int* __restrict__ d_ad_coff,
    int* __restrict__ d_ad_boff,
    int* __restrict__ d_ad_sign)
{
    int flat  = blockIdx.x * blockDim.x + threadIdx.x;
    int total = alpha_states * nadexc;
    if (flat >= total) return;

    int base   = 3 * flat;
    int coff   = d_adexc[base + 0];
    int orbij  = d_adexc[base + 1];
    int sign   = d_adexc[base + 2];

    if (orbij < 0 || orbij >= norbs2) return;

    int pos = atomicAdd(&d_cursors[orbij], 1);

    d_ad_coff[pos] = coff;
    d_ad_boff[pos] = flat / nadexc; // s1 = row index in adexc
    d_ad_sign[pos] = sign;
}

// ---------------------------------------------------------
// 3) Split beta excitations into SoA:
//    bdexc layout: for each (s2, j) with flat = s2*nbdexc + j
//      [ idx2, orbkl, parity ]
// ---------------------------------------------------------
__global__ void split_bdexc_kernel(
    const int* __restrict__ d_bdexc,
    int beta_states,
    int nbdexc,
    int* __restrict__ d_bd_idx2,
    int* __restrict__ d_bd_orbkl,
    int* __restrict__ d_bd_parity)
{
    int flat = blockIdx.x * blockDim.x + threadIdx.x;
    int total = beta_states * nbdexc;
    if (flat >= total) return;

    int base = 3 * flat;
    d_bd_idx2[flat]   = d_bdexc[base + 0];
    d_bd_orbkl[flat]  = d_bdexc[base + 1];
    d_bd_parity[flat] = d_bdexc[base + 2];
}

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
    int norbs)
{
    // Tunable tile sizes (keep small to be safe when everything is large)
    constexpr int SIG_TILE  = 32; // # alpha excitations per tile
    constexpr int BETA_TILE = 8;  // # beta determinants per block (y)
    constexpr int J_TILE    = 32; // # beta excitations per tile

    const int norbs2 = norbs * norbs;
    const int orbid  = blockIdx.x;
    if (orbid >= norbs2) return;

    const int beta_tile_base = blockIdx.y * BETA_TILE;
    const int local_beta     = threadIdx.y;
    const int s2             = beta_tile_base + local_beta;

    // CSR range for this orbid
    const int ad_begin = d_ad_offsets[orbid];
    const int ad_end   = d_ad_offsets[orbid + 1];
    const int nsig     = ad_end - ad_begin;
    if (nsig == 0) return;

    const cuDoubleComplex* __restrict__ h2e_block =
        d_h2e + static_cast<size_t>(orbid) * norbs2;

    // Shared memory layout (ints only)
    extern __shared__ int sh_int[];
    int* sh_coff  = sh_int;                             // [SIG_TILE]
    int* sh_boff  = sh_coff  + SIG_TILE;                // [SIG_TILE]
    int* sh_sign  = sh_boff  + SIG_TILE;                // [SIG_TILE]
    int* sh_idx2  = sh_sign  + SIG_TILE;                // [BETA_TILE * J_TILE]
    int* sh_orbkl = sh_idx2  + BETA_TILE * J_TILE;      // [BETA_TILE * J_TILE]
    int* sh_par   = sh_orbkl + BETA_TILE * J_TILE;      // [BETA_TILE * J_TILE]

    const int local_sig  = threadIdx.x; // 0..SIG_TILE-1
    const int block_size = blockDim.x * blockDim.y;
    const int flat_tid   = threadIdx.y * blockDim.x + threadIdx.x;

    // Loop over tiles of alpha excitations
    for (int sig_base = 0; sig_base < nsig; sig_base += SIG_TILE) {

        // 1) Load tile of alpha excitations into shared: coff, boff, sign
        for (int t = flat_tid; t < SIG_TILE; t += block_size) {
            int g = sig_base + t;
            if (g < nsig) {
                int idx = ad_begin + g;
                sh_coff[t] = d_ad_coff[idx];
                sh_boff[t] = d_ad_boff[idx];
                sh_sign[t] = d_ad_sign[idx];
            }
        }
        __syncthreads();

        const int global_sig = sig_base + local_sig;

        int row_in   = 0;
        int row_out  = 0;
        int sign     = 0;
        const cuDoubleComplex* C_row = nullptr;

        if (local_sig < SIG_TILE && global_sig < nsig) {
            row_in  = sh_coff[local_sig];
            row_out = sh_boff[local_sig];
            sign    = sh_sign[local_sig];

            if (row_in >= 0 && row_in < alpha_states) {
                C_row = d_C + static_cast<size_t>(row_in) * beta_states;
            }
        }

        const bool active = (s2 < beta_states) &&
                            (global_sig < nsig) &&
                            (local_sig < SIG_TILE) &&
                            (C_row != nullptr);

        cuDoubleComplex accum;
        accum.x = 0.0;
        accum.y = 0.0;

        // 2) Loop over tiles of beta excitations j
        for (int j_base = 0; j_base < nbdexc; j_base += J_TILE) {

            // Load bdexc tile into shared for this beta tile
            for (int jj = threadIdx.x; jj < J_TILE; jj += blockDim.x) {
                int j_global = j_base + jj;
                if (j_global >= nbdexc) break;

                int tile_index = local_beta * J_TILE + jj;

                if (s2 < beta_states) {
                    int flat = s2 * nbdexc + j_global;
                    sh_idx2[tile_index]  = d_bd_idx2[flat];
                    sh_orbkl[tile_index] = d_bd_orbkl[flat];
                    sh_par[tile_index]   = d_bd_parity[flat];
                } else {
                    // dummy
                    sh_idx2[tile_index]  = 0;
                    sh_orbkl[tile_index] = 0;
                    sh_par[tile_index]   = 0;
                }
            }

            __syncthreads();

            if (active) {
                int max_j = min(J_TILE, nbdexc - j_base);
                for (int jj = 0; jj < max_j; ++jj) {
                    int tile_index = local_beta * J_TILE + jj;

                    int idx2   = sh_idx2[tile_index];
                    int orbkl  = sh_orbkl[tile_index];
                    int parity = sh_par[tile_index];

                    if (idx2 < 0 || idx2 >= beta_states) continue;

                    cuDoubleComplex ttt = h2e_block[orbkl];
                    if (parity == -1) {
                        ttt.x = -ttt.x;
                        ttt.y = -ttt.y;
                    }

                    cuDoubleComplex cval = C_row[idx2];
                    if (sign == -1) {
                        cval.x = -cval.x;
                        cval.y = -cval.y;
                    }

                    cuDoubleComplex prod = cuCmul(ttt, cval);
                    accum = cuCadd(accum, prod);
                }
            }

            __syncthreads();
        }

        // 3) Scatter to out: out[boff(isig), s2] += accum
        if (active) {
            int row_out_clamped = (row_out >= 0 && row_out < alpha_states)
                                ? row_out : 0;
            size_t out_idx =
                static_cast<size_t>(row_out_clamped) * beta_states + s2;
            atomicAdd_cuDoubleComplex(&d_out[out_idx], accum);
        }

        __syncthreads();
    }
}

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
    int norbs)
{
    const int norbs2      = norbs * norbs;
    const int nadexc_tot  = alpha_states * nadexc;
    const int betaexc_tot = beta_states * nbdexc;

    // -----------------------------------
    // Build CSR for alpha excitations on GPU
    // -----------------------------------
    thrust::device_vector<int> d_counts(norbs2, 0);

    // here we compute the counts per orbid
    {
        int threads = 256;
        int blocks  = (nadexc_tot + threads - 1) / threads;

        count_alpha_excitations_per_orbid_kernel<<<blocks, threads>>>(
            d_adexc,
            alpha_states,
            nadexc,
            norbs2,
            thrust::raw_pointer_cast(d_counts.data()));
        cudaDeviceSynchronize();
    }

    // Compute offsets and total number of excitations
    thrust::device_vector<int> d_ad_offsets(norbs2 + 1);
    thrust::exclusive_scan(d_counts.begin(), d_counts.end(), d_ad_offsets.begin());

    int total_ex = thrust::reduce(d_counts.begin(), d_counts.end(), 0, thrust::plus<int>());

    // Set last offset = total_ex
    cudaMemcpy(thrust::raw_pointer_cast(d_ad_offsets.data()) + norbs2, &total_ex, sizeof(int), cudaMemcpyHostToDevice);

    // ------------------------------------
    // at this point, d_ad_offsets contains 
    // the complete CSR "row pointer" array
    // ------------------------------------

    // Allocate CSR data arrays
    thrust::device_vector<int> d_ad_coff(total_ex);
    thrust::device_vector<int> d_ad_boff(total_ex);
    thrust::device_vector<int> d_ad_sign(total_ex);

    // cursors initialized to offsets[0..norbs2-1]
    thrust::device_vector<int> d_cursors(norbs2);
    thrust::copy(d_ad_offsets.begin(), d_ad_offsets.begin() + norbs2, d_cursors.begin());

    {
        int threads = 256;
        int blocks  = (nadexc_tot + threads - 1) / threads;

        fill_alpha_csr_from_adexc_kernel<<<blocks, threads>>>(
            d_adexc,
            alpha_states,
            nadexc,
            norbs2,
            thrust::raw_pointer_cast(d_ad_offsets.data()),
            thrust::raw_pointer_cast(d_cursors.data()),
            thrust::raw_pointer_cast(d_ad_coff.data()),
            thrust::raw_pointer_cast(d_ad_boff.data()),
            thrust::raw_pointer_cast(d_ad_sign.data()));
        cudaDeviceSynchronize();
    }

    // -----------------------------------
    // Build SoA for beta excitations on GPU
    // -----------------------------------
    thrust::device_vector<int> d_bd_idx2(betaexc_tot);
    thrust::device_vector<int> d_bd_orbkl(betaexc_tot);
    thrust::device_vector<int> d_bd_parity(betaexc_tot);

    {
        int threads = 256;
        int blocks  = (betaexc_tot + threads - 1) / threads;

        split_bdexc_kernel<<<blocks, threads>>>(
            d_bdexc,
            beta_states,
            nbdexc,
            thrust::raw_pointer_cast(d_bd_idx2.data()),
            thrust::raw_pointer_cast(d_bd_orbkl.data()),
            thrust::raw_pointer_cast(d_bd_parity.data()));
        cudaDeviceSynchronize();
    }

    // -----------------------------------
    // Launch tiled, shared-memory main kernel
    // -----------------------------------
    constexpr int SIG_TILE  = 32;
    constexpr int BETA_TILE = 8;
    constexpr int J_TILE    = 32;

    dim3 block(SIG_TILE, BETA_TILE);
    dim3 grid(norbs2, (beta_states + BETA_TILE - 1) / BETA_TILE);

    size_t shmem_bytes =
        static_cast<size_t>(
            (3 * SIG_TILE + 3 * BETA_TILE * J_TILE) * sizeof(int));

    lm_apply_array12_diff_spin_kernel<<<grid, block, shmem_bytes>>>(
        d_out,
        d_C,
        thrust::raw_pointer_cast(d_ad_offsets.data()),
        thrust::raw_pointer_cast(d_ad_coff.data()),
        thrust::raw_pointer_cast(d_ad_boff.data()),
        thrust::raw_pointer_cast(d_ad_sign.data()),
        thrust::raw_pointer_cast(d_bd_idx2.data()),
        thrust::raw_pointer_cast(d_bd_orbkl.data()),
        thrust::raw_pointer_cast(d_bd_parity.data()),
        d_h2e,
        alpha_states,
        beta_states,
        nbdexc,
        norbs);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "lm_apply_array12_diff_spin_wrapper failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("lm_apply_array12_diff_spin_wrapper execution failed");
    }
}

// ==============================================
// New Same Spin kernel implementation 
// ==============================================

__global__ void build_same_spin_csr_cols_vals_kernel(
    int* __restrict__ d_col_ind,                 // [nnz_total]
    cuDoubleComplex* __restrict__ d_vals,         // [nnz_total]
    const int* __restrict__ d_dexc,               // [states1 * ndexc * 3]
    const cuDoubleComplex* __restrict__ d_h1e,    // [norbs2]
    const cuDoubleComplex* __restrict__ d_h2e,    // [norbs2 * norbs2]
    int states1,
    int ndexc,
    int norbs)
{
    const int norbs2 = norbs * norbs;

    // nnz per row = ndexc (h1) + ndexc*ndexc (h2)
    const int nnz_per_row = ndexc * (ndexc + 1);
    const long long nnz_total = (long long)states1 * (long long)nnz_per_row;

    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nnz_total) return;

    // Decode row/position inside row
    int s1 = (int)(p / nnz_per_row);
    int t  = (int)(p - (long long)s1 * nnz_per_row);

    // Base pointer for row s1 in d_dexc
    // (each excitation is 3 ints: [s2, ijshift, parity1])
    const int base_s1 = 3 * (s1 * ndexc);

    if (t < ndexc) {
        // -----------------------
        // h1e contribution entry:
        // A(s1, s2) += parity1 * h1e[ijshift]
        // -----------------------
        const int i = t;

        const int s2      = d_dexc[base_s1 + 3*i + 0];
        const int ijshift = d_dexc[base_s1 + 3*i + 1];
        const int parity1 = d_dexc[base_s1 + 3*i + 2]; // ±1

        d_col_ind[p] = s2;

        cuDoubleComplex v = d_h1e[ijshift];
        if (parity1 == -1) { v.x = -v.x; v.y = -v.y; }
        d_vals[p] = v;

    } else {
        // -----------------------
        // h2e contribution entry:
        // for each i in row s1:
        //   s2 = dexc(s1,i).s2, ijshift = dexc(s1,i).ijshift, parity1
        //   for each j in row s2:
        //     target = dexc(s2,j).s2, klshift = dexc(s2,j).ijshift, parity2
        //     A(s1, target) += parity1*parity2 * h2e[ijshift, klshift]
        // -----------------------
        const int t2 = t - ndexc;         // 0 .. ndexc*ndexc-1
        const int i  = t2 / ndexc;        // 0 .. ndexc-1
        const int j  = t2 - i*ndexc;      // 0 .. ndexc-1

        const int s2      = d_dexc[base_s1 + 3*i + 0];
        const int ijshift = d_dexc[base_s1 + 3*i + 1];
        const int parity1 = d_dexc[base_s1 + 3*i + 2];

        // Row s2 in d_dexc
        const int base_s2 = 3 * (s2 * ndexc);

        const int target  = d_dexc[base_s2 + 3*j + 0];
        const int klshift = d_dexc[base_s2 + 3*j + 1];
        const int parity2 = d_dexc[base_s2 + 3*j + 2];

        d_col_ind[p] = target;

        cuDoubleComplex v = d_h2e[(long long)ijshift * norbs2 + klshift];
        const int parity = parity1 * parity2;
        if (parity == -1) { v.x = -v.x; v.y = -v.y; }
        d_vals[p] = v;
    }
}

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
    int inc2)   // from your existing logic
{
    if (states1 == 0 || states2 == 0 || ndexc == 0) return;

    // nnz_per_row = ndexc*(ndexc+1)
    const long long nnz_per_row_ll = (long long)ndexc * (long long)(ndexc + 1);
    const long long nnz_total_ll   = (long long)states1 * nnz_per_row_ll;

    // If you want 32-bit CSR indices, row_ptr values must fit in int.
    if (nnz_total_ll > (long long)std::numeric_limits<int>::max()) {
        throw std::runtime_error("CSR nnz too large for 32-bit row_ptr; switch to 64-bit indices.");
    }

    const int nnz_per_row = (int)nnz_per_row_ll;
    const int nnz_total   = (int)nnz_total_ll;

    // ---- Build CSR row_ptr: row_ptr[r] = r * nnz_per_row ----
    thrust::device_vector<int> d_row_ptr(states1 + 1);
    thrust::sequence(d_row_ptr.begin(), d_row_ptr.end(), 0, nnz_per_row);

    // ---- Allocate CSR col/value arrays ----
    thrust::device_vector<int>            d_col_ind(nnz_total);
    thrust::device_vector<cuDoubleComplex> d_vals(nnz_total);

    // ---- Fill col_ind and values in parallel ----
    {
        int threads = 256;
        int blocks  = (nnz_total + threads - 1) / threads;

        build_same_spin_csr_cols_vals_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_col_ind.data()),
            thrust::raw_pointer_cast(d_vals.data()),
            d_dexc,
            d_h1e,
            d_h2e,
            states1,
            ndexc,
            norbs);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // ---- cuSPARSE SpMM: out = alpha*A*C + beta*out ----
    static cusparseHandle_t handle = nullptr;
    static bool handle_init = false;
    if (!handle_init) {
        CHECK_CUSPARSE(cusparseCreate(&handle));
        handle_init = true;
    }

    // A is CSR: (states1 x states1)
    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA,
        (int64_t)states1, (int64_t)states1, (int64_t)nnz_total,
        (void*)thrust::raw_pointer_cast(d_row_ptr.data()),
        (void*)thrust::raw_pointer_cast(d_col_ind.data()),
        (void*)thrust::raw_pointer_cast(d_vals.data()),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_C_64F));

    // Dense descriptors for B (= d_C) and C (= d_out)
    // cuSPARSE supports ROW or COL order, but not arbitrary 2D strides.
    // Your two common layouts are:
    //   Row-major: inc2==1  (ld = states2)
    //   Col-major: inc1==1  (ld = states1)
    cusparseOrder_t order;
    int ld;

    if (inc2 == 1) {
        // row-major
        order = CUSPARSE_ORDER_ROW;
        ld    = inc1; // should equal states2
    } else if (inc1 == 1) {
        // column-major
        order = CUSPARSE_ORDER_COL;
        ld    = inc2; // should equal states1
    } else {
        CHECK_CUSPARSE(cusparseDestroySpMat(matA));
        throw std::runtime_error("Unsupported dense layout for cuSPARSE DnMat (need row-major or col-major).");
    }

    cusparseDnMatDescr_t matB, matC;
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matB,
        (int64_t)states1, (int64_t)states2, (int64_t)ld,
        (void*)d_C,
        CUDA_C_64F,
        order));

    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matC,
        (int64_t)states1, (int64_t)states2, (int64_t)ld,
        (void*)d_out,
        CUDA_C_64F,
        order));

    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta  = make_cuDoubleComplex(1.0, 0.0); // accumulate into existing out

    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA, matB,
        &beta,
        matC,
        CUDA_C_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize));

    void* dBuffer = nullptr;
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    CHECK_CUSPARSE(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA, matB,
        &beta,
        matC,
        CUDA_C_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        dBuffer));

    CHECK_CUDA(cudaFree(dBuffer));

    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

__global__ void same_spin_build_raw_keys_vals_kernel(
    unsigned long long* __restrict__ d_keys,   // [nnz_raw]
    cuDoubleComplex* __restrict__ d_vals,      // [nnz_raw]
    const int* __restrict__ d_dexc,            // [states1 * ndexc * 3]
    const cuDoubleComplex* __restrict__ d_h1e, // [norbs2]
    const cuDoubleComplex* __restrict__ d_h2e, // [norbs2 * norbs2]
    int states1,
    int ndexc,
    int norbs)
{
    const int norbs2 = norbs * norbs;

    const long long nnz_per_row = (long long)ndexc * (long long)(ndexc + 1);
    const long long nnz_raw     = (long long)states1 * nnz_per_row;

    long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nnz_raw) return;

    const int s1 = (int)(p / nnz_per_row);
    const int t  = (int)(p - (long long)s1 * nnz_per_row);

    const int base_s1 = 3 * (s1 * ndexc);

    int col = 0;
    cuDoubleComplex val = make_cuDoubleComplex(0.0, 0.0);

    if (t < ndexc) {
        // h1e term: (s1 -> s2)
        const int i = t;

        const int s2      = d_dexc[base_s1 + 3*i + 0];
        const int ijshift = d_dexc[base_s1 + 3*i + 1];
        const int parity1 = d_dexc[base_s1 + 3*i + 2];

        col = s2;

        val = d_h1e[ijshift];
        if (parity1 == -1) { val.x = -val.x; val.y = -val.y; }

    } else {
        // h2e term: (s1 -> target) via s2 row
        const int t2 = t - ndexc;
        const int i  = t2 / ndexc;
        const int j  = t2 - i*ndexc;

        const int s2      = d_dexc[base_s1 + 3*i + 0];
        const int ijshift = d_dexc[base_s1 + 3*i + 1];
        const int parity1 = d_dexc[base_s1 + 3*i + 2];

        const int base_s2 = 3 * (s2 * ndexc);

        const int target  = d_dexc[base_s2 + 3*j + 0];
        const int klshift = d_dexc[base_s2 + 3*j + 1];
        const int parity2 = d_dexc[base_s2 + 3*j + 2];

        col = target;

        val = d_h2e[(long long)ijshift * norbs2 + klshift];
        const int parity = parity1 * parity2;
        if (parity == -1) { val.x = -val.x; val.y = -val.y; }
    }

    // Safety: keep indices in range; if out-of-range, write a 0 entry
    if (col < 0 || col >= states1) {
        col = 0;
        val = make_cuDoubleComplex(0.0, 0.0);
    }

    const unsigned long long key =
        ( (unsigned long long)(unsigned int)s1 << 32 ) |
        ( (unsigned long long)(unsigned int)col );

    d_keys[p] = key;
    d_vals[p] = val;
}

__global__ void same_spin_extract_cols_and_rowcounts_kernel(
    const unsigned long long* __restrict__ d_keys_unique, // [nnz_unique]
    int nnz_unique,
    int* __restrict__ d_col_ind,      // [nnz_unique]
    int* __restrict__ d_row_counts,   // [states1]
    int states1)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nnz_unique) return;

    unsigned long long key = d_keys_unique[k];
    int row = (int)(key >> 32);
    int col = (int)(key & 0xFFFFFFFFull);

    d_col_ind[k] = col;

    if (row >= 0 && row < states1) {
        atomicAdd(&d_row_counts[row], 1);
    }
}

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
    int inc2)
{
    if (states1 <= 0 || states2 <= 0 || ndexc <= 0) return;

    // ------------------------------------------------------------
    // 1) Build RAW COO contributions (duplicates allowed)
    //    raw nnz per row = ndexc + ndexc*ndexc = ndexc*(ndexc+1)
    // ------------------------------------------------------------
    const long long nnz_per_row = (long long)ndexc * (long long)(ndexc + 1);
    const long long nnz_raw_ll  = (long long)states1 * nnz_per_row;

    if (nnz_raw_ll <= 0) return;
    if (nnz_raw_ll > (long long)std::numeric_limits<int>::max()) {
        // You can support larger by switching some ints to int64 in cuSPARSE.
        throw std::runtime_error("nnz_raw too large for this 32-bit implementation");
    }

    const int nnz_raw = (int)nnz_raw_ll;

    thrust::device_vector<unsigned long long> d_keys_raw(nnz_raw);
    thrust::device_vector<cuDoubleComplex>    d_vals_raw(nnz_raw);

    {
        int threads = 256;
        int blocks  = (nnz_raw + threads - 1) / threads;

        same_spin_build_raw_keys_vals_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_keys_raw.data()),
            thrust::raw_pointer_cast(d_vals_raw.data()),
            d_dexc, d_h1e, d_h2e,
            states1, ndexc, norbs);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // ------------------------------------------------------------
    // 2) Sort by (row,col) key
    // ------------------------------------------------------------
    thrust::sort_by_key(d_keys_raw.begin(), d_keys_raw.end(), d_vals_raw.begin());

    // ------------------------------------------------------------
    // 3) Reduce duplicates: (row,col) sums its contributions
    // ------------------------------------------------------------
    thrust::device_vector<unsigned long long> d_keys_uni(nnz_raw);
    thrust::device_vector<cuDoubleComplex>    d_vals_uni(nnz_raw);

    auto end_pair = thrust::reduce_by_key(
        d_keys_raw.begin(), d_keys_raw.end(),
        d_vals_raw.begin(),
        d_keys_uni.begin(),
        d_vals_uni.begin(),
        thrust::equal_to<unsigned long long>(),
        cuCadd_op());

    int nnz_unique = (int)(end_pair.first - d_keys_uni.begin());
    if (nnz_unique <= 0) return;

    d_keys_uni.resize(nnz_unique);
    d_vals_uni.resize(nnz_unique);

    // ------------------------------------------------------------
    // 4) Build CSR row_ptr + col_ind from unique COO
    // ------------------------------------------------------------
    thrust::device_vector<int> d_row_counts(states1);
    thrust::fill(d_row_counts.begin(), d_row_counts.end(), 0);

    thrust::device_vector<int> d_col_ind(nnz_unique);

    {
        int threads = 256;
        int blocks  = (nnz_unique + threads - 1) / threads;

        same_spin_extract_cols_and_rowcounts_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_keys_uni.data()),
            nnz_unique,
            thrust::raw_pointer_cast(d_col_ind.data()),
            thrust::raw_pointer_cast(d_row_counts.data()),
            states1);

        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // CSR row_ptr: exclusive_scan(row_counts) + last element = nnz_unique
    thrust::device_vector<int> d_row_ptr(states1 + 1);
    thrust::exclusive_scan(d_row_counts.begin(), d_row_counts.end(), d_row_ptr.begin());

    CHECK_CUDA(cudaMemcpy(
        thrust::raw_pointer_cast(d_row_ptr.data()) + states1,
        &nnz_unique,
        sizeof(int),
        cudaMemcpyHostToDevice));

    // ------------------------------------------------------------
    // 5) cuSPARSE SpMM: out += A * C
    // A is CSR(states1 x states1), B is dense(states1 x states2), C is dense(states1 x states2)
    // ------------------------------------------------------------
    static cusparseHandle_t handle = nullptr;
    static bool handle_init = false;
    if (!handle_init) {
        CHECK_CUSPARSE(cusparseCreate(&handle));
        handle_init = true;
    }

    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(
        &matA,
        (int64_t)states1, (int64_t)states1, (int64_t)nnz_unique,
        (void*)thrust::raw_pointer_cast(d_row_ptr.data()),
        (void*)thrust::raw_pointer_cast(d_col_ind.data()),
        (void*)thrust::raw_pointer_cast(d_vals_uni.data()),
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO,
        CUDA_C_64F));

    // Dense matrix descriptors:
    // cuSPARSE only supports ROW or COL order (not arbitrary strides).
    cusparseOrder_t order;
    int ld = 0;

    if (inc2 == 1) {
        // row-major: idx = row*inc1 + col*1, so ld should be #cols == states2
        order = CUSPARSE_ORDER_ROW;
        ld = inc1;
    } else if (inc1 == 1) {
        // col-major: idx = row*1 + col*inc2, so ld should be #rows == states1
        order = CUSPARSE_ORDER_COL;
        ld = inc2;
    } else {
        CHECK_CUSPARSE(cusparseDestroySpMat(matA));
        throw std::runtime_error("Unsupported dense layout (need row-major or col-major contiguous)");
    }

    cusparseDnMatDescr_t matB, matC;
    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matB,
        (int64_t)states1, (int64_t)states2, (int64_t)ld,
        (void*)d_C,
        CUDA_C_64F,
        order));

    CHECK_CUSPARSE(cusparseCreateDnMat(
        &matC,
        (int64_t)states1, (int64_t)states2, (int64_t)ld,
        (void*)d_out,
        CUDA_C_64F,
        order));

    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta  = make_cuDoubleComplex(1.0, 0.0); // accumulate into existing out

    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA, matB,
        &beta,
        matC,
        CUDA_C_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize));

    void* dBuffer = nullptr;
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    CHECK_CUSPARSE(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA, matB,
        &beta,
        matC,
        CUDA_C_64F,
        CUSPARSE_SPMM_ALG_DEFAULT,
        dBuffer));

    CHECK_CUDA(cudaFree(dBuffer));

    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

// ==============================================
// Even Newer Diff Spin kernel implementation (v3)
// ==============================================

// ---------------------------------------------------------
// Diff spin v3: cache bdexc row + weights in shared once per block
// ---------------------------------------------------------
__global__ void lm_apply_array12_diff_spin_kernel_v3(
    cuDoubleComplex* __restrict__ d_out,
    const cuDoubleComplex* __restrict__ d_C,
    const int* __restrict__ d_ad_offsets, // [norbs2+1]
    const int* __restrict__ d_ad_coff,    // [total_ex]
    const int* __restrict__ d_ad_boff,    // [total_ex]
    const int* __restrict__ d_ad_sign,    // [total_ex]
    const int* __restrict__ d_bd_idx2,    // [beta_states*nbdexc]
    const int* __restrict__ d_bd_orbkl,   // [beta_states*nbdexc]
    const int* __restrict__ d_bd_parity,  // [beta_states*nbdexc]
    const cuDoubleComplex* __restrict__ d_h2e, // [norbs2*norbs2]
    int alpha_states,
    int beta_states,
    int nbdexc,
    int norbs)
{
    // Warp-centric tile:
    //   x = 32 lanes (one warp) -> alpha excitations within a tile
    //   y = BETA_TILE warps -> multiple s2 handled per block
    constexpr int SIG_TILE  = 32;
    constexpr int BETA_TILE = 8;

    const int norbs2 = norbs * norbs;

    const int orbid = blockIdx.x;
    if (orbid >= norbs2) return;

    const int beta_tile_base = blockIdx.y * BETA_TILE;
    const int local_beta     = threadIdx.y;           // 0..BETA_TILE-1
    const int s2             = beta_tile_base + local_beta;

    // CSR range for this orbid
    const int ad_begin = d_ad_offsets[orbid];
    const int ad_end   = d_ad_offsets[orbid + 1];
    const int nsig     = ad_end - ad_begin;
    if (nsig == 0) return;

    // Pointer to h2e row for this orbid
    const cuDoubleComplex* __restrict__ h2e_block =
        d_h2e + (size_t)orbid * norbs2;

    // ---------------------------
    // Shared memory layout:
    //   int sh_coff[SIG_TILE]
    //   int sh_boff[SIG_TILE]
    //   int sh_sign[SIG_TILE]
    //   int sh_idx2[BETA_TILE * nbdexc]
    //   cuDoubleComplex sh_ttt[BETA_TILE * nbdexc]   (precomputed parity*h2e)
    // ---------------------------
    extern __shared__ unsigned char smem[];

    int* sh_int = reinterpret_cast<int*>(smem);
    int* sh_coff = sh_int;
    int* sh_boff = sh_coff + SIG_TILE;
    int* sh_sign = sh_boff + SIG_TILE;
    int* sh_idx2 = sh_sign + SIG_TILE; // length = BETA_TILE*nbdexc

    // Align to 16 bytes for cuDoubleComplex
    size_t int_bytes = (size_t)(3 * SIG_TILE + BETA_TILE * nbdexc) * sizeof(int);
    size_t int_bytes_aligned = (int_bytes + 15) & ~((size_t)15);

    cuDoubleComplex* sh_ttt =
        reinterpret_cast<cuDoubleComplex*>(smem + int_bytes_aligned);

    // -------------------------------------------------------
    // Stage 0: load bdexc row (idx2) and precompute weights ttt
    // This is done ONCE per block and reused for all sig tiles.
    // Each warp (fixed local_beta) loads its own s2 row.
    // -------------------------------------------------------
    if (s2 < beta_states) {
        for (int jj = threadIdx.x; jj < nbdexc; jj += SIG_TILE) {
            const int flat = s2 * nbdexc + jj;

            const int idx2   = d_bd_idx2[flat];
            const int orbkl  = d_bd_orbkl[flat];
            const int parity = d_bd_parity[flat];

            const int dst = local_beta * nbdexc + jj;
            sh_idx2[dst] = idx2;

            cuDoubleComplex t = h2e_block[orbkl];
            if (parity == -1) { t.x = -t.x; t.y = -t.y; }
            sh_ttt[dst] = t;
        }
    } else {
        // out-of-range s2 lanes: fill dummy
        for (int jj = threadIdx.x; jj < nbdexc; jj += SIG_TILE) {
            const int dst = local_beta * nbdexc + jj;
            sh_idx2[dst] = 0;
            sh_ttt[dst]  = make_cuDoubleComplex(0.0, 0.0);
        }
    }
    __syncthreads();

    // Thread identifiers
    const int local_sig  = threadIdx.x; // 0..31
    const int flat_tid   = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;

    // -------------------------------------------------------
    // Loop over tiles of alpha excitations
    // -------------------------------------------------------
    for (int sig_base = 0; sig_base < nsig; sig_base += SIG_TILE) {

        // Load one alpha tile into shared (coff, boff, sign)
        for (int t = flat_tid; t < SIG_TILE; t += block_size) {
            int g = sig_base + t;
            if (g < nsig) {
                int idx = ad_begin + g;
                sh_coff[t] = d_ad_coff[idx];
                sh_boff[t] = d_ad_boff[idx];
                sh_sign[t] = d_ad_sign[idx];
            }
        }
        __syncthreads();

        const int global_sig = sig_base + local_sig;

        if (s2 < beta_states && global_sig < nsig) {

            const int row_in  = sh_coff[local_sig]; // alpha_to
            const int row_out = sh_boff[local_sig]; // alpha_from
            const int sign    = sh_sign[local_sig]; // ±1

            if ((unsigned)row_in < (unsigned)alpha_states &&
                (unsigned)row_out < (unsigned)alpha_states) {

                const cuDoubleComplex* __restrict__ C_row =
                    d_C + (size_t)row_in * beta_states;

                cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);

                const int base = local_beta * nbdexc;

                // Inner loop: nbdexc dot-product against sparse gather of C_row
                for (int jj = 0; jj < nbdexc; ++jj) {
                    const int idx2 = sh_idx2[base + jj];
                    if ((unsigned)idx2 >= (unsigned)beta_states) continue;

                    const cuDoubleComplex ttt = sh_ttt[base + jj];
                    cuDoubleComplex cval = C_row[idx2];
                    if (sign == -1) { cval.x = -cval.x; cval.y = -cval.y; }

                    // acc += ttt * cval
                    const cuDoubleComplex prod = cuCmul(ttt, cval);
                    acc.x += prod.x;
                    acc.y += prod.y;
                }

                // Scatter-add into output
                const size_t out_idx = (size_t)row_out * beta_states + (size_t)s2;
                atomicAdd(&d_out[out_idx].x, acc.x);
                atomicAdd(&d_out[out_idx].y, acc.y);
            }
        }

        __syncthreads();
    }
}

extern "C" void lm_apply_array12_diff_spin_wrapper_v3(
    cuDoubleComplex* d_out,
    const cuDoubleComplex* d_C,
    const int* d_adexc,
    const int* d_bdexc,
    const cuDoubleComplex* d_h2e,
    int alpha_states,
    int beta_states,
    int nadexc,
    int nbdexc,
    int norbs)
{
    const int norbs2      = norbs * norbs;
    const int nadexc_tot  = alpha_states * nadexc;
    const int betaexc_tot = beta_states * nbdexc;

    // -----------------------------
    // Alpha CSR by orbid (same as v2)
    // -----------------------------
    thrust::device_vector<int> d_counts(norbs2, 0);
    {
        int threads = 256;
        int blocks  = (nadexc_tot + threads - 1) / threads;
        count_alpha_excitations_per_orbid_kernel<<<blocks, threads>>>(
            d_adexc, alpha_states, nadexc, norbs2,
            thrust::raw_pointer_cast(d_counts.data()));
        cudaDeviceSynchronize();
    }

    thrust::device_vector<int> d_ad_offsets(norbs2 + 1);
    thrust::exclusive_scan(d_counts.begin(), d_counts.end(), d_ad_offsets.begin());
    int total_ex = thrust::reduce(d_counts.begin(), d_counts.end(), 0, thrust::plus<int>());
    cudaMemcpy(thrust::raw_pointer_cast(d_ad_offsets.data()) + norbs2,
               &total_ex, sizeof(int), cudaMemcpyHostToDevice);

    thrust::device_vector<int> d_ad_coff(total_ex);
    thrust::device_vector<int> d_ad_boff(total_ex);
    thrust::device_vector<int> d_ad_sign(total_ex);

    thrust::device_vector<int> d_cursors(norbs2);
    thrust::copy(d_ad_offsets.begin(), d_ad_offsets.begin() + norbs2, d_cursors.begin());

    {
        int threads = 256;
        int blocks  = (nadexc_tot + threads - 1) / threads;
        fill_alpha_csr_from_adexc_kernel<<<blocks, threads>>>(
            d_adexc, alpha_states, nadexc, norbs2,
            thrust::raw_pointer_cast(d_ad_offsets.data()),
            thrust::raw_pointer_cast(d_cursors.data()),
            thrust::raw_pointer_cast(d_ad_coff.data()),
            thrust::raw_pointer_cast(d_ad_boff.data()),
            thrust::raw_pointer_cast(d_ad_sign.data()));
        cudaDeviceSynchronize();
    }

    // -----------------------------
    // Beta SoA (same as v2)
    // -----------------------------
    thrust::device_vector<int> d_bd_idx2(betaexc_tot);
    thrust::device_vector<int> d_bd_orbkl(betaexc_tot);
    thrust::device_vector<int> d_bd_parity(betaexc_tot);

    {
        int threads = 256;
        int blocks  = (betaexc_tot + threads - 1) / threads;
        split_bdexc_kernel<<<blocks, threads>>>(
            d_bdexc, beta_states, nbdexc,
            thrust::raw_pointer_cast(d_bd_idx2.data()),
            thrust::raw_pointer_cast(d_bd_orbkl.data()),
            thrust::raw_pointer_cast(d_bd_parity.data()));
        cudaDeviceSynchronize();
    }

    // -----------------------------
    // Launch v3 kernel
    // -----------------------------
    constexpr int SIG_TILE  = 32;
    constexpr int BETA_TILE = 8;

    dim3 block(SIG_TILE, BETA_TILE);
    dim3 grid(norbs2, (beta_states + BETA_TILE - 1) / BETA_TILE);

    // shared bytes:
    //   ints: 3*SIG_TILE + BETA_TILE*nbdexc
    //   complex: BETA_TILE*nbdexc
    size_t int_bytes = (size_t)(3 * SIG_TILE + BETA_TILE * nbdexc) * sizeof(int);
    size_t int_bytes_aligned = (int_bytes + 15) & ~((size_t)15);
    size_t complex_bytes = (size_t)(BETA_TILE * nbdexc) * sizeof(cuDoubleComplex);
    size_t shmem_bytes = int_bytes_aligned + complex_bytes;

    lm_apply_array12_diff_spin_kernel_v3<<<grid, block, shmem_bytes>>>(
        d_out,
        d_C,
        thrust::raw_pointer_cast(d_ad_offsets.data()),
        thrust::raw_pointer_cast(d_ad_coff.data()),
        thrust::raw_pointer_cast(d_ad_boff.data()),
        thrust::raw_pointer_cast(d_ad_sign.data()),
        thrust::raw_pointer_cast(d_bd_idx2.data()),
        thrust::raw_pointer_cast(d_bd_orbkl.data()),
        thrust::raw_pointer_cast(d_bd_parity.data()),
        d_h2e,
        alpha_states,
        beta_states,
        nbdexc,
        norbs);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "lm_apply_array12_diff_spin_wrapper_v3 failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("lm_apply_array12_diff_spin_wrapper_v3 failed");
    }
}
