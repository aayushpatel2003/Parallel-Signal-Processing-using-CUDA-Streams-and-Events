#ifndef STREAMS_H
#define STREAMS_H

#include <cuda_runtime.h>

// Allocates host memory and fills with random floats
float* allocateHostMemory(int num_elements, int seed);

// Allocates device memory
float* allocateDeviceMemory(int num_elements);

// Deallocates device memory
void deallocateDevMemory(float* dev_mem);

// Determines thread and block dimensions
std::pair<int, int> determineThreadBlockDimensions(int num_elements);

// Asynchronously copies data from host to device
void copyFromHostToDeviceAsync(float* host_mem, float* dev_mem, int num_elements, cudaStream_t stream);

// Asynchronously copies data from device to host
void copyFromDeviceToHostAsync(float* dev_mem, float* host_mem, int num_elements, cudaStream_t stream);

#endif