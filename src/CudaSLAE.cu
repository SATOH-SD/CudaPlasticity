#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Mesh.cuh"

template<typename fp>
class CudaSLAE {

public:

	int N = 0; //размер СЛАУ

	int W = 1; //полуширина ленты

	int memLen = 0;

	//volatile fp* matrix = nullptr;  //массив элементов матрицы
	fp* matrix = nullptr;  //массив элементов матрицы

	fp* rp = nullptr;      //вектор правой части

	CudaSLAE() = default;

	CudaSLAE(int memLength, int size, int width)
		: N(size), W(width), memLen(memLength) {
		cudaMalloc((void**)&rp, memLen * sizeof(fp));
		cudaMalloc((void**)&matrix, memLen * (2 * W + 1) * sizeof(fp));
	};

	~CudaSLAE() {
		cudaFree(rp);
		cudaFree(matrix);
	}

	//Вернуть размер СЛАУ
	int size() const {
		return N;
	}

	//Ширина ленты (в одну сторону)
	int width() const {
		return W;
	}

	void clear();

	void clearMatrix();

	void print();

	fp matrixHash() const;

	fp rpHash() const;

	void saveToFile(const std::string& fileName) const;

};

template<typename fp>
__global__ void clearKF(fp* matrix, fp* rp, int N, int W) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (int j = i * (2 * W + 1); j < (i + 1) * (2 * W + 1); ++j)
		matrix[j] = {};
	rp[i] = {};
	//if (i < N) printf("%d %e\n", i, rp[i]);
}

template<typename fp>
__global__ void printRP(fp* rp, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	rp[i] = rp[i];
	//if (i < N) printf("%d %e\n", i, rp[i]);
}

template<typename fp>
void CudaSLAE<fp>::clear() {
	//std::cout << "clear?\n";
	clearKF<fp><<<memLen / BS, BS>>>(matrix, rp, N, W);
	cudaDeviceSynchronize();
	/*printRP<fp><<<1, BS>>>(rp, N);
	cudaDeviceSynchronize();*/
}

template<typename fp>
__global__ void clearMatr(fp* matrix, fp* rp, int N, int W) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	for (int j = i * (2 * W + 1); j < (i + 1) * (2 * W + 1); ++j)
		matrix[j] = {};
}

template<typename fp>
void CudaSLAE<fp>::clearMatrix() {
	//std::cout << "clear?\n";
	clearMatr<fp><<<memLen / BS, BS>>>(matrix, rp, N, W);
	cudaDeviceSynchronize();
}

template<typename fp>
void CudaSLAE<fp>::print() {
	
	//printRP<fp> << <1, BS >> > (rp, N);
	//cudaDeviceSynchronize();

	int len = N;
	fp* ramMatr = new fp[len * (2 * W + 1)];
	fp* ramRp = new fp[len];
	std::cout << "print?\n";
	//cudaDeviceSynchronize();
	//printRP<fp><<<1, BS>>>(rp, N);
	//std::cout << memLen << "\n";
	//std::cout << "print2?\n";
	cudaMemcpy(ramMatr, matrix, len * (2 * W + 1) * sizeof(fp), cudaMemcpyDeviceToHost);
	cudaMemcpy(ramRp, rp, len * sizeof(fp), cudaMemcpyDeviceToHost);

	/*for (int i = 0; i < len; ++i)
		std::cout << ramRp[i] << "\n";*/

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j <= 2 * W; ++j)
			std::cout << ramMatr[i * (2 * W + 1) + j] << " ";
		std::cout << "\n";
	}

	delete[] ramMatr;
	delete[] ramRp;
}


template<typename fp>
fp CudaSLAE<fp>::matrixHash() const {
	fp* locMatrix = new fp[memLen * (2 * W + 1)];
	cudaMemcpy(locMatrix, matrix, memLen * (2 * W + 1) * sizeof(fp), cudaMemcpyDeviceToHost);
	fp sum = 0.;
	for (int i = 0; i < memLen * (2 * W + 1); ++i)
		sum += locMatrix[i];
	return sum;
}

template<typename fp>
fp CudaSLAE<fp>::rpHash() const {
	fp* locRp = new fp[memLen];
	cudaMemcpy(locRp, rp, memLen * sizeof(fp), cudaMemcpyDeviceToHost);
	fp sum = 0.;
	for (int i = 0; i < memLen; ++i)
		sum += locRp[i];
	return sum;
}

template<typename fp>
void CudaSLAE<fp>::saveToFile(const std::string& fileName) const {
	std::ofstream file(fileName, std::ios_base::out);
	int len = N;
	fp* ramMatr = new fp[len * (2 * W + 1)];
	fp* ramRp = new fp[len];
	//std::cout << "print?\n";
	cudaMemcpy(ramMatr, matrix, len * (2 * W + 1) * sizeof(fp), cudaMemcpyDeviceToHost);
	cudaMemcpy(ramRp, rp, len * sizeof(fp), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j <= 2 * W; ++j)
			file << ramMatr[i * (2 * W + 1) + j] << " ";
		file << "\n";
	}

	delete[] ramMatr;
	delete[] ramRp;
	file.close();
}