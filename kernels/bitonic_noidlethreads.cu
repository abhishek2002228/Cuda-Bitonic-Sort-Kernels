#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <cuda_runtime.h>
#include <algorithm>

#define DTYPE int
#include <limits>


// because we do only n/2 comparisions in each kernel call, n/2 threads are idle. 
// we optimize this by using only n/2 threads, and remove the idle threads in each iteration of the outer loop
// each thread now computes its own (i,j) pair using bit manipulations
// because each thread does alot of computation to compute (i, j), we can give each thread more work. 
__global__ void bitonic_sort_kernel(int *a, int n, int s, int d, int t, bool finalUp, int workPerThread=1) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // grid-stride loop
    for (unsigned int thread = tid * workPerThread; thread < n/2; thread += gridDim.x * blockDim.x * workPerThread) {
        #pragma unroll
        for (unsigned int w = 0; w < workPerThread; w++) {
            unsigned int thread_id = thread + w;
            if (thread_id >= n/2) break; 
            //computation of mapping from thread index to (i,j) pair. (i, j) are the indices of the elements to be compared and swapped if needed
            unsigned int i = ((thread_id >> t) << (t + 1)) | (thread_id & (d-1));
            unsigned int j = i + d;

            if (j < n) {
                bool up = ((i & s) == 0);
                if (!finalUp) up = !up;
                int ai = a[i], aj = a[j];
                // use predicated execution to avoid branch divergence
                bool swap = ((ai > aj) == up);
                a[i] = swap ? aj : ai;
                a[j] = swap ? ai : aj;
            }
        }
    }
}


void bitonic_sort(int *d_arr, int size, bool finalUp = true, int workPerThread=1) {
    //assume size is a power of 2
    unsigned int threadsPerBlock = std::min(256, size);
    unsigned int num_pairs = size / 2;

    // number of threads needed with coarsening
    unsigned int threads = (num_pairs + workPerThread - 1) / workPerThread;
    unsigned int blocksPerGrid = (threads + threadsPerBlock - 1) / threadsPerBlock;

    // Main bitonic sort loop
    for (int s = 2; s <= size; s <<= 1) {
        for (int d = s >> 1; d > 0; d >>= 1) {
            unsigned int t = __builtin_ctz(d); // number of trailing zeros in d, also gives log2(d). 
            bitonic_sort_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, size, s, d, t, finalUp, workPerThread);
            cudaDeviceSynchronize();
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
        DTYPE max_val = std::numeric_limits<DTYPE>::max();
        DTYPE *temp = (DTYPE*)malloc((new_size - size) * sizeof(DTYPE));
        std::fill(temp, temp + (new_size - size), max_val);
        cudaMemcpy(d_arr + size, temp, (new_size - size) * sizeof(DTYPE), cudaMemcpyHostToDevice);
        free(temp);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&h2dTime, start, stop);

    cudaEventRecord(start);
    
// Perform bitonic sort on GPU
if (new_size > size)
    bitonic_sort(d_arr, new_size, true, 4);
else
    bitonic_sort(d_arr, size, true, 4);


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
