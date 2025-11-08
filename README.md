# Cuda-Bitonic-Sort-Kernels
A Bunch of CUDA kernels to perform BItonic Sort on very large arrays

## Kernels

- [bitonic_naive.cu](kernels/bitonic_naive.cu) - Basic bitonic sort implementation with naive thread-to-element mapping using global memory
- [bitonic_noidlethreads.cu](kernels/bitonic_noidlethreads.cu) - Optimized bitonic sort that eliminates idle threads
- [bitonic_shared_mem.cu](kernels/bitonic_shared_mem.cu) - Hybrid bitonic sort using shared memory for block-sized subsequences and global memory for larger sequences
- [bitonic_shared_mem_warp.cu](kernels/bitonic_shared_mem_warp.cu) - Advanced bitonic sort with shared memory and warp shuffle instructions for efficient intra-warp comparisons
