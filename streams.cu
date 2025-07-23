// streams.cu
#include "streams.h"
#include <iostream>
#include <fstream>
#include <tuple>
#include <cstdlib>
#define random_max 255.0f

__global__ void kernelA1(float *data, int n, float x) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) data[i] += x;
}

__global__ void kernelB1(float *data, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) data[i] *= 2.0f;
}

__global__ void kernelA2(float *data, int n, float x) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) data[i] -= x;
}

__global__ void kernelB2(float *data, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) data[i] /= 2.0f;
}

float* allocateHostMemory(int n, int seed) {
    float* data = new float[n];
    srand(seed);
    for (int i = 0; i < n; ++i)
        data[i] = static_cast<float>(rand()) / RAND_MAX * random_max;
    return data;
}

float* allocateDeviceMemory(int n) {
    float* d;
    cudaMalloc(&d, n * sizeof(float));
    return d;
}

void deallocateDevMemory(float* d) {
    cudaFree(d);
}

void copyFromHostToDeviceAsync(float* h, float* d, int n, cudaStream_t stream) {
    cudaMemcpyAsync(d, h, n * sizeof(float), cudaMemcpyHostToDevice, stream);
}

void copyFromDeviceToHostAsync(float* d, float* h, int n, cudaStream_t stream) {
    cudaMemcpyAsync(h, d, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
}

std::tuple<int, int> determineThreadBlockDimensions(int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    return {threads, blocks};
}

void writeCSVLine(float* data, int n, std::ofstream& out) {
    for (int i = 0; i < n; ++i) {
        out << data[i];
        if (i != n - 1) out << ",";
    }
    out << "\n";
}

float* runStreamsFullAsync(float* host_mem, int n) {
    float* dev_mem = allocateDeviceMemory(n);
    auto [tpb, bpg] = determineThreadBlockDimensions(n);

    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    copyFromHostToDeviceAsync(host_mem, dev_mem, n, s1);

    int seed;
    std::cout << "Enter seed for A1: "; std::cin >> seed;
    srand(seed);
    float x1 = static_cast<float>(rand()) / RAND_MAX * random_max;
    kernelA1<<<bpg, tpb, 0, s1>>>(dev_mem, n, x1);
    kernelB1<<<bpg, tpb, 0, s2>>>(dev_mem, n);

    std::cout << "Enter seed for A2: "; std::cin >> seed;
    srand(seed);
    float x2 = static_cast<float>(rand()) / RAND_MAX * random_max;
    kernelA2<<<bpg, tpb, 0, s1>>>(dev_mem, n, x2);
    kernelB2<<<bpg, tpb, 0, s2>>>(dev_mem, n);

    copyFromDeviceToHostAsync(dev_mem, host_mem, n, s1);
    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    deallocateDevMemory(dev_mem);
    return host_mem;
}

void printHostMemory(float* data, int n) {
    for (int i = 0; i < n; ++i) {
        printf("%.6f", data[i]);
        if (i != n - 1) printf(",");
    }
    printf("\n");
}

int main() {
    int n = 255;
    std::ofstream out("output.txt");
    out.close();

    for (int run = 0; run < 3; ++run) {
        float* host_mem = allocateHostMemory(n, run);
        std::ofstream out("output.txt", std::ios_base::app);
        writeCSVLine(host_mem, n, out);
        out.close();

        host_mem = runStreamsFullAsync(host_mem, n);

        out.open("output.txt", std::ios_base::app);
        writeCSVLine(host_mem, n, out);
        out.close();
    }

    return 0;
}
