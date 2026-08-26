#include "CudaChecker.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>


static std::string cudaVersion(int driverVersion) {
	int mainVersion = driverVersion / 1000;
	return std::to_string(mainVersion) + "." + std::to_string((driverVersion - mainVersion * 1000) / 10);
}


CudaChecker::CudaChecker() {
	cudaGetLastError();  // reset last error
	int driverVersion;
	cudaError_t driverError = cudaDriverGetVersion(&driverVersion);

	cudaGetDeviceCount(&gpuCount);
	if (gpuCount) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		std::cout << "Detected ";
		if (gpuCount > 1)
			std::cout << gpuCount << "installed GPUs\nGPU 0: ";
		std::cout << deviceProp.name << " " \
			<< round(deviceProp.totalGlobalMem / double(1024 * 1024)) << " MB\n";
		if (driverVersion >= CUDART_VERSION) {
			success = testGpu(0);
			if (success) {
				globalMem = deviceProp.totalGlobalMem;
				sharedMem = deviceProp.sharedMemPerMultiprocessor;
			}
		}
		else {
			std::cout << "WARNING! Unable to run CUDA due to old installed GPU driver\n";
			std::cout << "Program CUDA version: " << cudaVersion(CUDART_VERSION) << "\n";
			std::cout << "Driver supported CUDA version: " << cudaVersion(driverVersion) << "\n";
			success = false;
		}
	}
	else {
		std::cout << "There is no installed CUDA-capable GPU\n";
		success = false;
	}

	if (success)
		std ::cout << "Running native mode...\n\n";
	else
		std::cout << "Running CPU only...\n\n";
}


static __global__ void addKernel(int* c, const int* a, const int* b) {
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}


#define CUDA_CHECK(error, action) { \
	cudaError_t err = error; \
	if (err) { \
		std::cout << "CUDA TESTING ERROR! Unable to " << action << "\n"; \
		std::cout << cudaGetErrorName(err) << ": " << cudaGetErrorString(err) << "\n"; \
		return false; \
	} \
}


bool CudaChecker::testGpu(int gpuId) {
	cudaSetDevice(gpuId);
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	const int c[arraySize] = { 11, 22, 33, 44, 55 };
	int result[arraySize] = {};
	int* dev_a = nullptr, * dev_b = nullptr, * dev_c = nullptr;

	CUDA_CHECK(cudaMalloc((void**)&dev_a, arraySize * sizeof(int)), "allocate memory");
	CUDA_CHECK(cudaMalloc((void**)&dev_b, arraySize * sizeof(int)), "allocate memory");
	CUDA_CHECK(cudaMalloc((void**)&dev_c, arraySize * sizeof(int)), "allocate memory");

	CUDA_CHECK(cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice), "copy data to device");
	CUDA_CHECK(cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice), "copy data to device");

	addKernel<<<1, arraySize>>>(dev_c, dev_a, dev_b);
	cudaDeviceSynchronize();
	CUDA_CHECK(cudaGetLastError(), "run kernel");

	CUDA_CHECK(cudaMemcpy(result, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost), "copy data from device");

	CUDA_CHECK(cudaFree(dev_a), "free memory");
	CUDA_CHECK(cudaFree(dev_b), "free memory");
	CUDA_CHECK(cudaFree(dev_c), "free memory");

	bool correct = true;
	for (int i = 0; i < arraySize; ++i)
		if (result[i] != c[i]) {
			correct = false;
			break;
		}
	if (!correct) {
		std::cout << "TESTING ERROR! Incorrect calculation\n";
		return false;
	}
	return true;
}