#include "fci_computer_gpu_kernels.cuh"
#include <cuda_runtime.h>
#include <iostream>


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
    cuDoubleComplex* __restrict__ temp_global) // size states1*states1
{
    // Each thread handles one s1 state
    int s1 = blockIdx.x * blockDim.x + threadIdx.x;
    if (s1 >= states1) return;

    cuDoubleComplex* temp = temp_global + s1 * states1;

    // Initialize temp to zero
    for (int i = 0; i < states1; ++i) {
        temp[i] = make_cuDoubleComplex(0.0, 0.0);
    }

    const int* cdexc = d_dexc + 3 * s1 * ndexc;
    const int* lim1 = cdexc + 3 * ndexc;

    // First loop: build temp array
    for (; cdexc < lim1; cdexc += 3) {
        const int s2 = cdexc[0];
        const int ijshift = cdexc[1];
        const int parity1 = cdexc[2];
        
        const int* cdexc2 = d_dexc + 3 * s2 * ndexc;
        const int* lim2 = cdexc2 + 3 * ndexc;
        const int h2e_id = ijshift * norbs * norbs;
        const cuDoubleComplex* h2etmp = d_h2e + h2e_id;
        
        // Add h1e contribution
        cuDoubleComplex h1e_contrib = cuCmul(
            make_cuDoubleComplex(static_cast<double>(parity1), 0.0),
            d_h1e[ijshift]
        );
        temp[s2] = cuCadd(temp[s2], h1e_contrib);

        // Inner loop over cdexc2
        for (; cdexc2 < lim2; cdexc2 += 3) {
            const int target = cdexc2[0];
            const int klshift = cdexc2[1];
            const int parity = cdexc2[2] * parity1;
            
            cuDoubleComplex pref = cuCmul(
                make_cuDoubleComplex(static_cast<double>(parity), 0.0),
                h2etmp[klshift]
            );
            temp[target] = cuCadd(temp[target], pref);
        }
    }

    // Second loop: accumulate into output using axpy-like operation
    cuDoubleComplex* cout = d_out + s1 * inc1;
    const cuDoubleComplex* xptr = d_C;
    
    for (int ii = 0; ii < states1; ++ii) {
        cuDoubleComplex ttt = temp[ii];
        
        // Perform axpy: cout[j] += ttt * xptr[j] for j in range(states2)
        for (int j = 0; j < states2; ++j) {
            int idx = j * inc2;
            cuDoubleComplex contrib = cuCmul(ttt, xptr[idx]);

            cout[idx].x += contrib.x;
            cout[idx].y += contrib.y;
        }
        xptr += inc1;
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

    int blockSize = 256;
    int gridSize = (states1 + blockSize - 1) / blockSize;

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

// Kernel to compute nsig and populate signs, coff, boff for a given orbid
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

    for (int i = 0; i < nadexc; ++i) {
        int idx = 3 * (s1 * nadexc + i);
        int orbij = d_adexc[idx + 1];
        
        if (orbij == orbid) {
            int pos = atomicAdd(d_nsig, 1);
            d_signs[pos] = d_adexc[idx + 2];
            d_coff[pos] = d_adexc[idx];
            d_boff[pos] = s1;
        }
    }
}

// Kernel to compute ctemp array
__global__ void lm_diff_spin_ctemp_kernel(
    cuDoubleComplex* __restrict__ d_ctemp,
    const cuDoubleComplex* __restrict__ d_C,
    const int* __restrict__ d_signs,
    const int* __restrict__ d_coff,
    int beta_states,
    int nsig)
{
    int isig = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (isig >= nsig) return;

    int offset = d_coff[isig];
    const cuDoubleComplex* cptr = d_C + offset * beta_states;
    cuDoubleComplex zsign = make_cuDoubleComplex(static_cast<double>(d_signs[isig]), 0.0);

    for (int j = 0; j < beta_states; ++j) {
        cuDoubleComplex contrib = cuCmul(zsign, cptr[j]);
        atomicAdd_double(&d_ctemp[j * nsig + isig].x, contrib.x);
        atomicAdd_double(&d_ctemp[j * nsig + isig].y, contrib.y);
    }
}

// Kernel to compute vtemp and accumulate to output
__global__ void lm_diff_spin_vtemp_kernel(
    cuDoubleComplex* __restrict__ d_out,
    const cuDoubleComplex* __restrict__ d_ctemp,
    const int* __restrict__ d_bdexc,
    const cuDoubleComplex* __restrict__ d_h2e,
    const int* __restrict__ d_boff,
    int beta_states,
    int nbdexc,
    int norbs,
    int nsig,
    int orbid)
{
    int s2 = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (s2 >= beta_states) return;

    const int norbs2 = norbs * norbs;
    const cuDoubleComplex* tmperi = d_h2e + orbid * norbs2;

    // Shared memory for vtemp
    extern __shared__ cuDoubleComplex shared_vtemp[];
    cuDoubleComplex* vtemp = &shared_vtemp[threadIdx.x * nsig];

    // Initialize vtemp to zero
    for (int kk = 0; kk < nsig; ++kk) {
        vtemp[kk] = make_cuDoubleComplex(0.0, 0.0);
    }

    // Compute vtemp
    for (int j = 0; j < nbdexc; ++j) {
        int idx2 = d_bdexc[3 * (s2 * nbdexc + j)];
        int parity = d_bdexc[3 * (s2 * nbdexc + j) + 2];
        int orbkl = d_bdexc[3 * (s2 * nbdexc + j) + 1];
        
        cuDoubleComplex ttt = cuCmul(
            make_cuDoubleComplex(static_cast<double>(parity), 0.0),
            tmperi[orbkl]
        );

        const cuDoubleComplex* cctmp = d_ctemp + idx2 * nsig;
        for (int isig = 0; isig < nsig; ++isig) {
            cuDoubleComplex contrib = cuCmul(ttt, cctmp[isig]);
            vtemp[isig] = cuCadd(vtemp[isig], contrib);
        }
    }

    // Accumulate into output
    cuDoubleComplex* tmpout = d_out + s2;
    for (int isig = 0; isig < nsig; ++isig) {
        int out_idx = beta_states * d_boff[isig];
        atomicAdd_double(&tmpout[out_idx].x, vtemp[isig].x);
        atomicAdd_double(&tmpout[out_idx].y, vtemp[isig].y);
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
    const int norbs2 = norbs * norbs;
    const int nadexc_tot = alpha_states * nadexc;

    // Allocate device memory for intermediate arrays (max size)
    int* d_signs;
    int* d_coff;
    int* d_boff;
    int* d_nsig;
    cuDoubleComplex* d_ctemp;

    cudaMalloc(&d_signs, nadexc_tot * sizeof(int));
    cudaMalloc(&d_coff, nadexc_tot * sizeof(int));
    cudaMalloc(&d_boff, nadexc_tot * sizeof(int));
    cudaMalloc(&d_nsig, sizeof(int));
    cudaMalloc(&d_ctemp, nadexc_tot * beta_states * sizeof(cuDoubleComplex));

    // Loop over all orbital pairs
    for (int orbid = 0; orbid < norbs2; ++orbid) {
        // Reset nsig counter
        cudaMemset(d_nsig, 0, sizeof(int));
        cudaMemset(d_ctemp, 0, nadexc_tot * beta_states * sizeof(cuDoubleComplex));

        // Compute nsig and populate arrays for this orbid
        int threadsPerBlock = 256;
        int blocksPerGrid = (alpha_states + threadsPerBlock - 1) / threadsPerBlock;
        
        lm_diff_spin_compute_nsig_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_adexc, d_signs, d_coff, d_boff, d_nsig,
            alpha_states, nadexc, orbid);
        
        cudaDeviceSynchronize();

        // Get nsig value
        int nsig;
        cudaMemcpy(&nsig, d_nsig, sizeof(int), cudaMemcpyDeviceToHost);

        if (nsig == 0) continue;

        // Compute ctemp
        blocksPerGrid = (nsig + threadsPerBlock - 1) / threadsPerBlock;
        lm_diff_spin_ctemp_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_ctemp, d_C, d_signs, d_coff, beta_states, nsig);
        
        cudaDeviceSynchronize();

        // Compute vtemp and accumulate to output
        blocksPerGrid = (beta_states + threadsPerBlock - 1) / threadsPerBlock;
        size_t sharedMemSize = threadsPerBlock * nsig * sizeof(cuDoubleComplex);
        
        lm_diff_spin_vtemp_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(
            d_out, d_ctemp, d_bdexc, d_h2e, d_boff,
            beta_states, nbdexc, norbs, nsig, orbid);
        
        cudaDeviceSynchronize();
    }

    // Free device memory
    cudaFree(d_signs);
    cudaFree(d_coff);
    cudaFree(d_boff);
    cudaFree(d_nsig);
    cudaFree(d_ctemp);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "lm_apply_array12_diff_spin_wrapper failed ("
                  << cudaGetErrorString(err) << ")\n";
        throw std::runtime_error("lm_apply_array12_diff_spin_wrapper execution failed");
    }
}