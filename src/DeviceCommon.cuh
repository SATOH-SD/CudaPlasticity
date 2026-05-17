#pragma once

#include "cuda_runtime.h"


#define MAX_BS 1024

#define cudaAddKernelNodeMacro(_func, grid, block, shmem, args, node, graph, deps, depsNum) { \
	cudaKernelNodeParams nodeParams = {}; \
	nodeParams.func = (void*)_func; \
	nodeParams.gridDim = dim3(grid); \
	nodeParams.blockDim = dim3(block); \
	nodeParams.sharedMemBytes = shmem; \
	nodeParams.kernelParams = args; \
	nodeParams.extra = nullptr; \
	cudaGraphAddKernelNode(&node, graph, deps, depsNum, &nodeParams); \
}

cudaError_t cudaAddKernelNode(void* func, dim3 grid, dim3 block, unsigned shmem, \
	void* args[], \
	cudaGraphNode_t& node, cudaGraph_t graph, \
	std::initializer_list<cudaGraphNode_t> dependencies);

cudaError_t cudaAddMemsetNode(void* devPtr, int value, size_t count, \
	cudaGraphNode_t& node, cudaGraph_t graph, \
	std::initializer_list<cudaGraphNode_t> dependencies);

//__device__ inline void warpReduce(volatile float* sdata, int tid) {
//	sdata[tid] += sdata[tid + 32];
//	sdata[tid] += sdata[tid + 16];
//	sdata[tid] += sdata[tid + 8];
//	sdata[tid] += sdata[tid + 4];
//	sdata[tid] += sdata[tid + 2];
//	sdata[tid] += sdata[tid + 1];
//}
//__device__ inline void warpReduce(volatile double* sdata, int tid) {
//	sdata[tid] += sdata[tid + 32];
//	sdata[tid] += sdata[tid + 16];
//	sdata[tid] += sdata[tid + 8];
//	sdata[tid] += sdata[tid + 4];
//	sdata[tid] += sdata[tid + 2];
//	sdata[tid] += sdata[tid + 1];
//}


template<typename T>
__device__ inline void warpReduce(volatile T* sdata, int tid) {
	//MIN BLOCK SIZE 64
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

template<typename T>
__device__ inline void blockReduce(T* sdata, T* sum, int tid) {
	__syncthreads();

	for (unsigned s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if (tid < 32) warpReduce(sdata, tid);
	if (tid == 0) atomicAdd(sum, sdata[0]);
}

__device__ inline double atomicAdd_arch52(double* address, double val);

__global__ void norm2_f(float* x, float* sum);
__global__ void norm2_d(double* x, double* sum);

__global__ void floatToDouble(double* dst, float* src, unsigned N);
__global__ void doubleToFloat(float* dst, double* src, unsigned N);