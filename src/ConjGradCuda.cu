#pragma once

#include "CudaSparseSLAE.cuh"

template <typename fp>
__device__ void warpReduce(volatile fp* sdata, int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

template<typename fp>
__global__ void cgInitSparseMask(fp* data, int* rows, int* cols, fp* rp, fp* xNext, fp* rNext, fp* zNext, fp* rNextScal, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ fp sdata[BS];
	fp sum = 0.;
	for (int j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * xNext[cols[j]];         //вычисление A.x0
	fp rk = mask[i] * (rp[i] - sum);
	zNext[i] = rNext[i] = rk;
	sdata[tid] = rk * rk;
	__syncthreads();

	//вычисление (r0, r0)
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid < 32) warpReduce(sdata, tid);
	if (tid == 0) atomicAdd(rNextScal, sdata[0]);
}

template<typename fp>
__global__ void cg1sparse(fp* data, int* rows, int* cols, fp* rPrev, fp* zPrev, fp* Az, fp* Az_z) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ fp sdata[BS];

	double sum = 0.;
	for (int j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * zPrev[cols[j]];         //вычисление A.z
	Az[i] = sum;
	sdata[tid] = sum * zPrev[i];

	__syncthreads();

	//вычисление (A.z, z)
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if (tid < 32) warpReduce(sdata, tid);
	if (tid == 0) atomicAdd(Az_z, sdata[0]);
}

template <typename fp>
__global__ void cg2mask(fp* xNext, fp* xPrev, fp* rNext, fp* rPrev, fp* zPrev, fp* Az, fp* Az_z, fp rPrevScal, fp* rNextScal, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ fp sdata[BS];
	fp alpha = rPrevScal / *Az_z;  //вычисление alpha

	xNext[i] = xPrev[i] + alpha * zPrev[i];       //вычисление xk
	fp rk = rPrev[i] - mask[i] * alpha * Az[i];   //вычисление rk
	rNext[i] = rk;
	sdata[tid] = rk * rk;
	__syncthreads();

	//вычисление (rk, rk)
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid < 32) warpReduce(sdata, tid);
	if (tid == 0) atomicAdd(rNextScal, sdata[0]);
}

template <typename fp>
__global__ void cg3(fp* rNext, fp* zNext, fp* zPrev, fp rPrevScal, fp rNextScal) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	fp beta = rNextScal / rPrevScal;                 //вычисление beta
	zNext[i] = rNext[i] + beta * zPrev[i];           //вычисление zk
}

template <typename fp>
__global__ void exitNormR(fp* rNext, fp* norm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	atomicAdd(norm, rNext[i] * rNext[i]);  //(не эффективно)
}

template<typename fp>
class ConjGradCuda {

	CudaSparseSLAE<fp>& slae;

	fp* xNext = nullptr,
		* xPrev = nullptr,
		* rNext = nullptr,
		* rPrev = nullptr,
		* zNext = nullptr,
		* zPrev = nullptr,
		* Az = nullptr;

	fp _normB = {};

	//0 - r, 1 - Az_z
	fp* dev_scalNext = nullptr;

public:

	ConjGradCuda(CudaSparseSLAE<fp>& sparseSlae)
		: slae(sparseSlae) {
		//cudaMalloc((void**)&xNext, slae.memLen * sizeof(fp));

		cudaMalloc((void**)&xPrev, slae.memLen * sizeof(fp));
		cudaMalloc((void**)&rNext, slae.memLen * sizeof(fp));
		cudaMalloc((void**)&rPrev, slae.memLen * sizeof(fp));
		cudaMalloc((void**)&zNext, slae.memLen * sizeof(fp));
		cudaMalloc((void**)&zPrev, slae.memLen * sizeof(fp));
		cudaMalloc((void**)&Az, slae.memLen * sizeof(fp));
		cudaMemset(rNext, 0, slae.memLen * sizeof(fp));
		cudaMemset(rPrev, 0, slae.memLen * sizeof(fp));
		cudaMemset(zNext, 0, slae.memLen * sizeof(fp));
		cudaMemset(zPrev, 0, slae.memLen * sizeof(fp));
		
		_normB = 0.;
		fp* dev_normB = nullptr;
		cudaMalloc((void**)&dev_normB, sizeof(fp));
		cudaMemcpy(dev_normB, &_normB, sizeof(fp), cudaMemcpyHostToDevice);
		exitNormR<<<slae.memLen / BS, BS>>>(slae.rp, dev_normB);
		cudaMemcpy(&_normB, dev_normB, sizeof(fp), cudaMemcpyDeviceToHost);
		_normB = fp(1.) / _normB;
		//std::cout << "normB " << 1. / _normB << "\n";
		cudaFree(dev_normB);

		cudaMalloc((void**)&dev_scalNext, 2 * sizeof(fp));
		cudaMemset(dev_scalNext, 0, 2 * sizeof(fp));
	}

	~ConjGradCuda() {
		//cudaFree(xNext); //?
		cudaFree(xPrev); //?

		cudaFree(rNext);
		cudaFree(rPrev);
		cudaFree(zNext);
		cudaFree(zPrev);
		cudaFree(Az);

		cudaFree(dev_scalNext);
	}

	//DEBUG
	int maskSum(bool* mask, int N) {
		bool* loc_mask = new bool[N];
		cudaMemcpy(loc_mask, mask, N * sizeof(bool), cudaMemcpyDeviceToHost);
		int sum = 0;
		for (int i = 0; i < N; ++i)
			sum += (int)!loc_mask[i];
		for (int i = N - 1024; i < N; ++i) \
			std::cout << loc_mask[i] << " ";
		delete[] loc_mask;
		return sum;
	}

	//DEBUG
	template<typename T>
	void printEnd(T* vec, int N) {
		T* loc = new T[N];
		cudaMemcpy(loc, vec, N * sizeof(T), cudaMemcpyDeviceToHost);
		for (int i = N - 1024; i < N; ++i) \
			std::cout << loc[i] << " ";
		delete[] loc;
	}

	void solve(fp* solution, bool* mask, size_t& iterNum, fp eps) {
		//std::cout << "MASK " << maskSum(mask, slae.memLen) << "\n";
		//printEnd(slae.cols, slae.memLen);

		eps = eps * eps;
		fp rNextScal = 0., rPrevScal = 0.;

		int block = BS; std::min(BS, slae.N);
		int grid = slae.memLen / BS;

		xNext = solution;
		//cudaMemset(xNext, 0, slae.memLen * sizeof(fp));
		//cudaMemcpy(xNext, solution, slae.N * sizeof(fp), cudaMemcpyDeviceToDevice);

		cgInitSparseMask<fp><<<grid, block>>>(slae.data, slae.rows, slae.cols, slae.rp, xNext, rNext, zNext, dev_scalNext, mask);
		cudaDeviceSynchronize();
		cudaMemcpy(&rNextScal, dev_scalNext, sizeof(fp), cudaMemcpyDeviceToHost);

		do {
			std::swap(xNext, xPrev);
			std::swap(rNext, rPrev);
			std::swap(zNext, zPrev);

			cudaDeviceSynchronize();
			cudaMemset(dev_scalNext, 0, 2 * sizeof(fp));
			cg1sparse<fp><<<grid, block>>>(slae.data, slae.rows, slae.cols, rPrev, zPrev, Az, dev_scalNext + 1);
			cudaDeviceSynchronize();
			cg2mask<fp><<<grid, block>>>(xNext, xPrev, rNext, rPrev, zPrev, Az, dev_scalNext + 1, rNextScal, dev_scalNext, mask);
			rPrevScal = rNextScal;
			cudaDeviceSynchronize();
			cudaMemcpy(&rNextScal, dev_scalNext, sizeof(fp), cudaMemcpyDeviceToHost);
			cg3<fp><<<grid, block>>>(rNext, zNext, zPrev, rPrevScal, rNextScal);

			++iterNum;
			//if (iterNum == 1) break;
			//if (iterNum > 1000) break;
			//if (sizeof(fp) == 8) std::cout << rNextScal << "\n";
		} while (rNextScal * _normB > eps);
		//std::cout << "iterNum = " << iterNum << "\n";
		//std::cout << "NORM = " << rNextScal * _normB << "\n";

		//solution = xNext;
		if (iterNum % 2 == 1) std::swap(xNext, xPrev);
		//std::cout << "SLAE " << slae.N << " " << slae.memLen << "\n";
		//cudaMemset(xNext, 255, slae.memLen * sizeof(fp));
		//cudaMemcpy(solution, xNext, slae.N * sizeof(fp), cudaMemcpyDeviceToDevice);
	}
	

};