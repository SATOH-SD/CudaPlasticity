#pragma once

class CudaChecker {

public:

	CudaChecker();

	//bool setGpu(int gpuId);

	bool isGpuOn() const {
		return success;
	}

	size_t getGlobalMem() const {
		return globalMem;
	}

	size_t getSharedMem() const {
		return sharedMem;
	}

private:

	size_t globalMem = 0;
	size_t sharedMem = 0;

	int gpuCount = 0;
	int currentGpuId = 0;
	int smCount = 0;

	bool success = false;

	bool testGpu(int gpuId);

};