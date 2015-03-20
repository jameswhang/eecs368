#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"


__global__ void histogramKernel(uint32_t *input, size_t width, size_t height, uint32_t *bins);
__global__ void wordToByte(uint32_t *input, uint8_t *output);
void* allocateDevice(size_t);
void copyToDevice(void*, void*, size_t);


__global__ void histogramKernel(uint32_t *input, size_t width, size_t height, uint32_t *bins) {
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	const int offset = gridDim.x * blockDim.x;	
	if (tid < 1024) {
		bins[tid] = 0;
	}
	__syncthreads();
	for (int pos = tid; pos < width * height; pos += offset) {
		const int data = input[pos];
		if (bins[data] < UINT8_MAX && data) {
			atomicAdd(&(bins[data]), 1);
		}
	}
	__syncthreads();
}

__global__ void wordToByte(uint32_t *input, uint8_t *output) {
	const int tid = threadIdx.x + blockIdx.x * blockDim.x;
	output[tid] = (uint8_t)((input[tid] < UINT8_MAX) * input[tid]) + (input[tid] >= UINT8_MAX) * UINT8_MAX;
	syncthreads();
}

void  opt_2dhisto(uint32_t *input, size_t width, size_t height, uint32_t *bins, uint8_t *bytebins)
{
	dim3 DimGrid(width, 1, 1);
	dim3 DimBlock(256, 1);
	histogramKernel<<<DimGrid, DimBlock>>>(input, width, height, bins);
	wordToByte<<<2, 512>>>(bins, bytebins);
	cudaThreadSynchronize();
}

void* allocateDevice(size_t size) {
	void *obj;
	cudaMalloc(&obj, size);
	return obj;
}

void copyToDevice(void* device, void* host, size_t size) {
	cudaMemcpy(device, host, size, cudaMemcpyHostToDevice);
}

void copyToHost(void* host, void* device, size_t size) {
	cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost);
}

void freeGPU(void* device) {
	cudaFree(device);
}





