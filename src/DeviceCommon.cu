#include "DeviceCommon.cuh"

#include "device_launch_parameters.h"

//#ifndef __CUDACC__  
//#define __CUDACC__
//#endif
//#include <device_functions.h>


cudaError_t cudaAddKernelNode(void* func, dim3 grid, dim3 block, unsigned shmem, \
	void* args[], \
	cudaGraphNode_t& node, cudaGraph_t graph, \
	std::initializer_list<cudaGraphNode_t> dependencies) {

	cudaKernelNodeParams nodeParams = {};
	nodeParams.func = func;
	nodeParams.gridDim = grid;
	nodeParams.blockDim = block;
	nodeParams.sharedMemBytes = shmem;
	nodeParams.kernelParams = args;
	nodeParams.extra = nullptr;
	return cudaGraphAddKernelNode(&node, graph, dependencies.begin(), dependencies.size(), &nodeParams);
}

cudaError_t cudaAddMemsetNode(void* devPtr, int value, size_t count, \
	cudaGraphNode_t& node, cudaGraph_t graph, \
	std::initializer_list<cudaGraphNode_t> dependencies) {

	cudaMemsetParams memsetParams = {};
	memsetParams.dst = devPtr;
	memsetParams.value = value;
	memsetParams.pitch = 0;
	memsetParams.elementSize = 1;
	memsetParams.width = count;
	memsetParams.height = 1;
	return cudaGraphAddMemsetNode(&node, graph, dependencies.begin(), dependencies.size(), &memsetParams);
}

//__device__ inline float atomicAdd_arch52(float* address, float val) {
//	return atomicAdd(address, val);
//}
//
//__device__ inline double atomicAdd_arch52(double* address, double val)
//{
//	unsigned long long int* address_as_ull =
//		(unsigned long long int*)address;
//	unsigned long long int old = *address_as_ull, assumed;
//	do {
//		assumed = old;
//		old = atomicCAS(address_as_ull, assumed, \
//			__double_as_longlong(val + __longlong_as_double(assumed)));
//	} while (assumed != old);
//
//	return __longlong_as_double(old);
//}


__global__ void floatToDouble(double* dst, float* src, unsigned N) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
		dst[i] = double(src[i]);
}

__global__ void doubleToFloat(float* dst, double* src, unsigned N) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
		dst[i] = float(src[i]);
}