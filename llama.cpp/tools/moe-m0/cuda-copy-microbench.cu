#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_OK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

__global__ void copy_u4(const uint4 * src, uint4 * dst, size_t n) {
    size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t) blockDim.x * gridDim.x) {
        dst[i] = src[i];
    }
}

__global__ void bf16_gemv(const __nv_bfloat16 * matrix, const float * vector, float * output, int rows, int cols) {
    int row = blockIdx.x;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        sum += __bfloat162float(matrix[(size_t) row * cols + col]) * vector[col];
    }
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
    __shared__ float warps[8];
    if ((threadIdx.x & 31) == 0) warps[threadIdx.x / 32] = sum;
    __syncthreads();
    if (threadIdx.x < 32) {
        sum = threadIdx.x < 8 ? warps[threadIdx.x] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (threadIdx.x == 0) output[row] = sum;
    }
}

static double elapsed(cudaEvent_t begin, cudaEvent_t end) {
    float ms = 0;
    cudaEventElapsedTime(&ms, begin, end);
    return ms;
}

int main() {
    constexpr size_t bytes = 64ull << 20;
    constexpr int repetitions = 32;
    void * pageable = nullptr;
    if (posix_memalign(&pageable, 4096, bytes) != 0) return 2;
    memset(pageable, 1, bytes);
    void * pinned = nullptr;
    void * mapped = nullptr;
    void * mapped_device = nullptr;
    void * device = nullptr;
    CUDA_OK(cudaHostAlloc(&pinned, bytes, cudaHostAllocPortable));
    CUDA_OK(cudaHostAlloc(&mapped, bytes, cudaHostAllocPortable | cudaHostAllocMapped));
    memset(pinned, 2, bytes);
    memset(mapped, 3, bytes);
    CUDA_OK(cudaHostGetDevicePointer(&mapped_device, mapped, 0));
    CUDA_OK(cudaMalloc(&device, bytes));
    cudaEvent_t begin, end;
    CUDA_OK(cudaEventCreate(&begin));
    CUDA_OK(cudaEventCreate(&end));

    auto run_copy = [&](const char * name, const void * source) {
        cudaDeviceSynchronize();
        cudaEventRecord(begin);
        for (int i = 0; i < repetitions; ++i) cudaMemcpyAsync(device, source, bytes, cudaMemcpyHostToDevice);
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        double ms = elapsed(begin, end);
        printf("%s: %.3f ms, %.2f GiB/s\n", name, ms, bytes * (double) repetitions / ms / 1e6);
    };
    run_copy("pageable cudaMemcpyAsync", pageable);
    run_copy("pinned cudaMemcpyAsync", pinned);

    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaEventRecord(begin));
    for (int i = 0; i < repetitions; ++i) {
        copy_u4<<<256, 256>>>((const uint4 *) mapped_device, (uint4 *) device, bytes / sizeof(uint4));
    }
    CUDA_OK(cudaEventRecord(end));
    CUDA_OK(cudaEventSynchronize(end));
    double ms = elapsed(begin, end);
    printf("mapped host kernel copy: %.3f ms, %.2f GiB/s\n", ms, bytes * (double) repetitions / ms / 1e6);

    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaEventRecord(begin));
    for (int i = 0; i < repetitions; ++i) {
        copy_u4<<<256, 256>>>((const uint4 *) pageable, (uint4 *) device, bytes / sizeof(uint4));
    }
    CUDA_OK(cudaEventRecord(end));
    CUDA_OK(cudaEventSynchronize(end));
    ms = elapsed(begin, end);
    printf("pageable host kernel copy: %.3f ms, %.2f GiB/s\n", ms, bytes * (double) repetitions / ms / 1e6);

    int can_map = 0, pageable_access = 0, pageable_uses_host_pt = 0;
    CUDA_OK(cudaDeviceGetAttribute(&can_map, cudaDevAttrCanMapHostMemory, 0));
    CUDA_OK(cudaDeviceGetAttribute(&pageable_access, cudaDevAttrPageableMemoryAccess, 0));
    CUDA_OK(cudaDeviceGetAttribute(&pageable_uses_host_pt, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0));
    printf("attributes: can_map=%d pageable_access=%d pageable_host_pt=%d mapped_same=%d\n",
           can_map, pageable_access, pageable_uses_host_pt, mapped == mapped_device);

    constexpr int rows = 2048;
    constexpr int cols = 768;
    constexpr size_t matrix_bytes = (size_t) rows * cols * sizeof(__nv_bfloat16);
    constexpr int gemv_repetitions = 512;
    void * matrix_host = nullptr;
    void * matrix_mapped = nullptr;
    void * matrix_pageable = nullptr;
    __nv_bfloat16 * matrix_device = nullptr;
    float * vector_device = nullptr;
    float * output_device = nullptr;
    CUDA_OK(cudaHostAlloc(&matrix_host, matrix_bytes, cudaHostAllocPortable | cudaHostAllocMapped));
    if (posix_memalign(&matrix_pageable, 4096, matrix_bytes) != 0) return 2;
    CUDA_OK(cudaHostGetDevicePointer(&matrix_mapped, matrix_host, 0));
    CUDA_OK(cudaMalloc(&matrix_device, matrix_bytes));
    CUDA_OK(cudaMalloc(&vector_device, cols * sizeof(float)));
    CUDA_OK(cudaMalloc(&output_device, rows * sizeof(float)));
    memset(matrix_host, 0x3f, matrix_bytes);
    memset(matrix_pageable, 0x3f, matrix_bytes);
    CUDA_OK(cudaMemcpy(matrix_device, matrix_host, matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(vector_device, 0, cols * sizeof(float)));

    auto run_gemv = [&](const char * name, const __nv_bfloat16 * matrix, bool copy_first) {
        CUDA_OK(cudaDeviceSynchronize());
        CUDA_OK(cudaEventRecord(begin));
        for (int i = 0; i < gemv_repetitions; ++i) {
            if (copy_first) CUDA_OK(cudaMemcpyAsync(matrix_device, matrix_host, matrix_bytes, cudaMemcpyHostToDevice));
            bf16_gemv<<<rows, 256>>>(matrix, vector_device, output_device, rows, cols);
        }
        CUDA_OK(cudaEventRecord(end));
        CUDA_OK(cudaEventSynchronize(end));
        double gemv_ms = elapsed(begin, end);
        printf("%s: %.4f ms/expert\n", name, gemv_ms / gemv_repetitions);
        return 0;
    };
    if (run_gemv("device BF16 GEMV", matrix_device, false)) return 1;
    if (run_gemv("mapped host BF16 GEMV", (const __nv_bfloat16 *) matrix_mapped, false)) return 1;
    if (run_gemv("pageable host BF16 GEMV", (const __nv_bfloat16 *) matrix_pageable, false)) return 1;
    if (run_gemv("copy + device BF16 GEMV", matrix_device, true)) return 1;

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaFree(device);
    cudaFree(output_device);
    cudaFree(vector_device);
    cudaFree(matrix_device);
    cudaFreeHost(matrix_host);
    free(matrix_pageable);
    cudaFreeHost(mapped);
    cudaFreeHost(pinned);
    free(pageable);
    return 0;
}
