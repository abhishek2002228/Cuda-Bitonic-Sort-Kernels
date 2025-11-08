#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>

#define DTYPE int
#include <limits>


//global bitonic sort kernel (uses global memory only) -> tid to element mapping. 
__global__ void bitonic_sort_global(int *a, int n, int s, int d, bool finalUp) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // grid-stride loop
    for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
        int j = i ^ d;      // partner index (via XOR)
        if (j > i && j < n) {
            bool up = ((i & s) == 0);
            if (!finalUp) up = !up;
            int ai = a[i], aj = a[j];
            if ((ai > aj) == up) {
                a[i] = aj;
                a[j] = ai;
            }
        }
    }
}

// shared memory bitonic sort kernel (uses shared memory to reduce global memory accesses)
// Can sort a subsequence of length upto block size 
__global__ void bitonic_sort_shared(int *a, int n, bool finalUp) {
    extern __shared__ int shared[];
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_tid = threadIdx.x;

    // load data into shared memeory
    if (global_tid < n) {
        shared[local_tid] = a[global_tid];
    } else {
        shared[local_tid] = 10000; // pad with max value
    }
    __syncthreads(); // ensure all data is loaded
    //s is subsequence length
    //d is the distance of elements to be compared
    for (unsigned int s=2; s <= blockDim.x; s <<= 1) {
        for (unsigned int d = s >> 1; d > 0; d >>= 1) {
            unsigned int i = local_tid;
            unsigned int j = i ^ d; // partner index (via XOR)
            if (j > i && j < blockDim.x) {
                bool up = ((global_tid & s) == 0);
                if (!finalUp) up = !up;
                int ai = shared[i], aj = shared[j];
                if ((ai > aj) == up) {
                    shared[i] = aj;
                    shared[j] = ai;
                }
            }
            __syncthreads(); // ensure all threads have completed this stage
        }
    }
    // write sorted data back to global memory
    if (global_tid < n) {
        a[global_tid] = shared[local_tid];
    }
}

// shared memory bitonic sort kernel to fused multiple d stages. 
__global__ void bitonic_sort_shared_fused(int *a, int n, int s, int d, bool finalUp) {
    extern __shared__ int shared[];
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_tid = threadIdx.x;

    // load data into shared memeory
    if (global_tid < n) {
        shared[local_tid] = a[global_tid];
    } else {
        shared[local_tid] = 10000; // pad with max value
    }
    __syncthreads(); // ensure all data is loaded
    //s is subsequence length
    //d is the distance of elements to be compared
    for ( ; d > 0; d >>= 1) {
        unsigned int i = local_tid;
        unsigned int j = i ^ d; // partner index (via XOR)
        if (j > i && j < blockDim.x) {
            bool up = ((global_tid & s) == 0);
            if (!finalUp) up = !up;
            int ai = shared[i], aj = shared[j];
            if ((ai > aj) == up) {
                shared[i] = aj;
                shared[j] = ai;
            }
        }
        __syncthreads(); // ensure all threads have completed this stage
    }
    // write sorted data back to global memory
    if (global_tid < n) {
        a[global_tid] = shared[local_tid];
    }
}

void bitonic_sort_hybrid(int *d_arr, int size, bool finalUp = true) {
    //assume size is a power of 2
    int threadsPerBlock = std::min(1024, size);
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    /// Phase A - first sort subsequences that fit in a block using shared memory kernel
    bitonic_sort_shared<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(DTYPE)>>>(d_arr, size, finalUp);

    // if there are more than 1024 elements, we need to do more passes using global memory kernel
    // Phase B - Then continue with global memory kernel for larger subsequences
    for (int s = threadsPerBlock << 1; s <= size; s <<= 1) {
        // phase B.1 - for d > threadsPerBlock, we need to use global memory kernel
        for (int d = s >> 1; d >= threadsPerBlock; d >>= 1) {
            bitonic_sort_global<<<blocksPerGrid, threadsPerBlock>>>(d_arr, size, s, d, finalUp);
        }
        // phase B.2 - for d <= threadsPerBlock, we can use shared memory kernel
        bitonic_sort_shared_fused<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(DTYPE)>>>(d_arr, size, s, threadsPerBlock >> 1, finalUp);
    }
}

void bitonic_sort(int *d_arr, int size, bool finalUp = true) {
    //assume size is a power of 2
    int threadsPerBlock = std::min(256, size);
    int blocksPerGrid = (size) / threadsPerBlock;

    // Main bitonic sort loop
    for (int s = 2; s <= size; s <<= 1) {
        for (int d = s >> 1; d > 0; d >>= 1) {
            bitonic_sort_global<<<blocksPerGrid, threadsPerBlock>>>(d_arr, size, s, d, finalUp);
        }
    }
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);

    srand(time(NULL));

    DTYPE* arrCpu = (DTYPE*)malloc(size * sizeof(DTYPE));

    for (int i = 0; i < size; i++) {
        arrCpu[i] = rand() % 1000;
    }

    float gpuTime, h2dTime, d2hTime, cpuTime = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

// arCpu contains the input random array
// arrSortedGpu should contain the sorted array copied from GPU to CPU
int *arrSortedGpu = (DTYPE*)malloc(size * sizeof(DTYPE));

// Transfer data (arr_cpu) to device
    ////check if size is a power of 2
    int new_size = size;
    if ((size & (size - 1)) != 0) {
        new_size = 1;
        while (new_size < size) new_size <<= 1;
    }
    int *d_arr;
    cudaMalloc((void**)&d_arr, new_size * sizeof(DTYPE));
    cudaMemcpy(d_arr, arrCpu, size * sizeof(DTYPE), cudaMemcpyHostToDevice); 
    //set the rest of the elements to max DTYPE using cudaMemcpy
    if (new_size > size) {
        DTYPE *temp = (DTYPE*)malloc((new_size - size) * sizeof(DTYPE));
        std::fill(temp, temp + (new_size - size), 1000);
        cudaMemcpy(d_arr + size, temp, (new_size - size) * sizeof(DTYPE), cudaMemcpyHostToDevice);
        free(temp);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);

    cudaEventRecord(start);
    
// Perform bitonic sort on GPU
if (new_size > size)
    bitonic_sort_hybrid(d_arr, new_size);
else
    bitonic_sort_hybrid(d_arr, size);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuTime, start, stop);

    cudaEventRecord(start);


// Transfer sorted data back to host (copied to arrSortedGpu)
    cudaMemcpy(arrSortedGpu, d_arr, size * sizeof(DTYPE), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&d2hTime, start, stop);

    auto startTime = std::chrono::high_resolution_clock::now();
    
    // CPU sort for performance comparison
    std::sort(arrCpu, arrCpu + size);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    cpuTime = cpuTime / 1000;

    int match = 1;
    for (int i = 0; i < size; i++) {
        if (arrSortedGpu[i] != arrCpu[i]) {
            match = 0;
            break;
        }
    }

    free(arrCpu);
    free(arrSortedGpu);

    if (match)
        printf("\033[1;32mFUNCTIONAL SUCCESS\n\033[0m");
    else {
        printf("\033[1;31mFUNCTIONCAL FAIL\n\033[0m");
        return 0;
    }
    
    printf("\033[1;34mArray size         :\033[0m %d\n", size);
    printf("\033[1;34mCPU Sort Time (ms) :\033[0m %f\n", cpuTime);
    float gpuTotalTime = h2dTime + gpuTime + d2hTime;
    int speedup = (gpuTotalTime > cpuTime) ? (gpuTotalTime/cpuTime) : (cpuTime/gpuTotalTime);
    float meps = size / (gpuTotalTime * 0.001) / 1e6;
    printf("\033[1;34mGPU Sort Time (ms) :\033[0m %f\n", gpuTotalTime);
    printf("\033[1;34mGPU Sort Speed     :\033[0m %f million elements per second\n", meps);
    if (gpuTotalTime < cpuTime) {
        printf("\033[1;32mPERF PASSING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;32m %dx \033[1;34mfaster than CPU !!!\033[0m\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
    } else {
        printf("\033[1;31mPERF FAILING\n\033[0m");
        printf("\033[1;34mGPU Sort is \033[1;31m%dx \033[1;34mslower than CPU, optimize further!\n", speedup);
        printf("\033[1;34mH2D Transfer Time (ms):\033[0m %f\n", h2dTime);
        printf("\033[1;34mKernel Time (ms)      :\033[0m %f\n", gpuTime);
        printf("\033[1;34mD2H Transfer Time (ms):\033[0m %f\n", d2hTime);
        return 0;
    }

    return 0;
}
