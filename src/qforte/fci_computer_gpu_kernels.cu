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

    for (int i = 0; i < nadexc; ++i) {
        int idx   = 3 * (s1 * nadexc + i);
        int orbij = d_adexc[idx + 1];
        if (orbij == orbid) {
            int pos = atomicAdd(d_nsig, 1);
            d_signs[pos] = d_adexc[idx + 2]; // ±1
            d_coff[pos]  = d_adexc[idx];     // α_to
            d_boff[pos]  = s1;               // α_from
        }
    }
}

// 2) build ctemp(β, isig) = sign_α * C(α_to, β)
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

    int alpha_to = d_coff[isig];
    const cuDoubleComplex* cptr = d_C + alpha_to * beta_states;
    cuDoubleComplex zsign = make_cuDoubleComplex(double(d_signs[isig]), 0.0);

    for (int beta = 0; beta < beta_states; ++beta) {
        cuDoubleComplex contrib = cuCmul(zsign, cptr[beta]);
        d_ctemp[beta * nsig + isig] = contrib;  // no += needed if d_ctemp was memset to 0
    }
}

// 3) beta-loop + GEMV + scatter
__global__ void lm_diff_spin_vtemp_kernel(
    cuDoubleComplex* __restrict__ d_out,
    const cuDoubleComplex* __restrict__ d_ctemp,
    const int* __restrict__ d_bdexc,
    const cuDoubleComplex* __restrict__ d_h2e,
    const int* __restrict__ d_boff,
    int alpha_states,
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

    cuDoubleComplex* tmpout = d_out + s2;

    for (int j = 0; j < nbdexc; ++j) {
        int base   = 3 * (s2 * nbdexc + j);
        int idx2   = d_bdexc[base + 0]; // beta_to
        int orbkl  = d_bdexc[base + 1]; // (k,l)
        int parity = d_bdexc[base + 2]; // sign_β

        if (idx2 < 0 || idx2 >= beta_states) continue; // TEMP GUARD for debugging

        cuDoubleComplex ttt = tmperi[orbkl];
        if (parity == -1) {
            ttt.x = -ttt.x;
            ttt.y = -ttt.y;
        }

        const cuDoubleComplex* cctmp = d_ctemp + idx2 * nsig;

        for (int isig = 0; isig < nsig; ++isig) {
            cuDoubleComplex contrib = cuCmul(ttt, cctmp[isig]);

            int alpha_from = d_boff[isig];
            if (alpha_from < 0 || alpha_from >= alpha_states) continue; // TEMP GUARD

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

    int* d_signs;
    int* d_coff;
    int* d_boff;
    int* d_nsig;

    cudaMalloc(&d_signs, nadexc_tot * sizeof(int));
    cudaMalloc(&d_coff,  nadexc_tot * sizeof(int));
    cudaMalloc(&d_boff,  nadexc_tot * sizeof(int));
    cudaMalloc(&d_nsig,  sizeof(int));

    // Pre-allocate d_ctemp with maximum possible size (nadexc_tot * beta_states)
    // This avoids repeated malloc/free in the loop
    cuDoubleComplex* d_ctemp;
    cudaMalloc(&d_ctemp, nadexc_tot * beta_states * sizeof(cuDoubleComplex));

    int threadsPerBlock = 256;

    for (int orbid = 0; orbid < norbs2; ++orbid) {
        cudaMemset(d_nsig,  0, sizeof(int));

        // 1) compute nsig
        int blocks = (alpha_states + threadsPerBlock - 1) / threadsPerBlock;
        lm_diff_spin_compute_nsig_kernel<<<blocks, threadsPerBlock>>>(
            d_adexc, d_signs, d_coff, d_boff, d_nsig,
            alpha_states, nadexc, orbid);

        int nsig;
        cudaMemcpy(&nsig, d_nsig, sizeof(int), cudaMemcpyDeviceToHost);
        if (nsig == 0) continue;

        if (nsig > nadexc_tot) {
            std::cerr << "ERROR: nsig > nadexc_tot, something is wrong with adexc\n";
            break;
        }

        // Zero only the portion we need
        cudaMemset(d_ctemp, 0, nsig * beta_states * sizeof(cuDoubleComplex));

        // 2) build ctemp
        blocks = (nsig + threadsPerBlock - 1) / threadsPerBlock;
        lm_diff_spin_ctemp_kernel<<<blocks, threadsPerBlock>>>(
            d_ctemp, d_C, d_signs, d_coff, beta_states, nsig);

        // 3) beta loop + accumulate
        blocks = (beta_states + threadsPerBlock - 1) / threadsPerBlock;
        lm_diff_spin_vtemp_kernel<<<blocks, threadsPerBlock>>>(
            d_out, d_ctemp, d_bdexc, d_h2e, d_boff,
            alpha_states, beta_states, nbdexc, norbs, nsig, orbid);
    }

    // Free d_ctemp once at the end
    cudaFree(d_ctemp);

    cudaFree(d_signs);
    cudaFree(d_coff);
    cudaFree(d_boff);
    cudaFree(d_nsig);

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
    cuDoubleComplex* __restrict__ d_out,
    const cuDoubleComplex* __restrict__ d_C,
    const cuDoubleComplex* __restrict__ d_temp,
    int states1,
    int states2,
    int inc1,
    int inc2)
{
    int s1 = blockIdx.x;   // one block per s1
    if (s1 >= states1) return;

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    extern __shared__ cuDoubleComplex sh_temp[];  // tile buffer

    // Each block handles all j for this s1; j is split across threads
    for (int iiBase = 0; iiBase < states1; iiBase += blockSize) {

        int tileCount = min(blockSize, states1 - iiBase);

        // load this tile of temp into shared memory
        if (tid < tileCount) {
            sh_temp[tid] = d_temp[s1 * states1 + (iiBase + tid)];
        }
        __syncthreads();

        // each thread handles j = tid, tid+blockSize, ...
        for (int j = tid; j < states2; j += blockSize) {

            // accumulate over this tile
            cuDoubleComplex partial = make_cuDoubleComplex(0.0, 0.0);

            for (int local = 0; local < tileCount; ++local) {
                int ii = iiBase + local;

                cuDoubleComplex tval = sh_temp[local];

                // C(ii, j) with generic strides inc1/inc2
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
