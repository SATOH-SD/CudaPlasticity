#pragma once

#include "CudaSparseSLAE.cuh"

#include <cooperative_groups.h>

#include "cuda.h"

#include "CGKernels.cuh"

template <typename fp>
static __device__ void warpReduce(volatile fp* sdata, int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

template<typename fp>
static __global__ void cgInitSparseMask(fp* data, int* rows, int* cols, fp* rp, fp* xNext, fp* rNext, fp* zNext, fp* rNextScal, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ fp sdata[BS];
	fp sum = {};
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
static __global__ void cg1sparse(fp* data, int* rows, int* cols, fp* rPrev, fp* zPrev, fp* Az, fp* Az_z) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ fp sdata[BS];

	fp sum = {};
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
static __global__ void cg2mask(fp* xNext, fp* xPrev, fp* rNext, fp* rPrev, fp* zPrev, fp* Az, fp* Az_z, fp rPrevScal, fp* rNextScal, bool* mask) {
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
static __global__ void cg3_(fp* rNext, fp* zNext, fp* zPrev, fp rPrevScal, fp rNextScal) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	fp beta = rNextScal / rPrevScal;                 //вычисление beta
	zNext[i] = rNext[i] + beta * zPrev[i];           //вычисление zk
}

template <typename fp>
static __global__ void exitNormR_(fp* rNext, fp* norm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	atomicAdd(norm, rNext[i] * rNext[i]);  //(не эффективно)
}


//LEGACY (traditional form, memory overconsumption, high delays)
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
		exitNormR_<<<slae.memLen / BS, BS>>>(slae.rp, dev_normB);
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
			cudaMemcpyAsync(&rNextScal, dev_scalNext, sizeof(fp), cudaMemcpyDeviceToHost);
			cg3_<fp><<<grid, block>>>(rNext, zNext, zPrev, rPrevScal, rNextScal);

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


const unsigned CGBS = 512;


template<typename fp>
static __global__ void cgInit(fp* data, int* rows, int* cols, fp* rp, fp* xNext, fp* rNext, fp* zNext, fp* rhoNext, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ fp sdata[CGBS];
	fp sum = {};
	for (int j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * xNext[cols[j]];         //вычисление A.x0
	fp rk = mask[i] * (rp[i] - sum);
	rNext[i] = rk;
	zNext[i] = 0;
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
	if (tid == 0) atomicAdd(rhoNext, sdata[0]);
}

template <typename fp>
static __global__ void cg1(fp* r, fp* z, fp rhoPrev, fp rhoNext) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//if (i == 0) printf("rhoNext %f\n", rhoNext);
	//if (i == 0) printf("rhoPrev %f\n", rhoPrev);
	fp beta = rhoNext / rhoPrev;                  //вычисление beta
	z[i] = r[i] + beta * z[i];        //вычисление zk
	//if (i == 0) printf("beta %f\n", beta);
	//if (i < 32) printf("%f ", zNext[i]);
}

template<typename fp>
static __global__ void cg2(fp* data, int* rows, int* cols, fp* z, fp* s, fp* omega) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ fp sdata[CGBS];

	//if (i == 0) printf("cg2\n");

	fp sum = {};
	for (int j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * z[cols[j]];         //вычисление A.z
	s[i] = sum;
	sdata[tid] = sum * z[i];

	__syncthreads();

	//вычисление (A.z, z)
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if (tid < 32) warpReduce(sdata, tid);
	if (tid == 0) atomicAdd(omega, sdata[0]);
}

template <typename fp>
static __global__ void cg3(fp* x, fp* r, fp* z, fp* s, fp* omega, fp rhoPrev, fp* rhoNext, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ fp sdata[CGBS];
	//if (i == 0) printf("omega   %e\n\n", *omega);
	fp alpha = rhoPrev / *omega;  //вычисление alpha

	//if (i == 0) printf("rhoPrCu %e\n", rhoPrev);
	//if (i == 0) printf("alpha   %e\n", alpha);
	//if (i == 0) printf("rhoNext %f\n", *rhoNext);
	x[i] += alpha * z[i];       //вычисление xk

	fp rk = r[i] -= mask[i] * alpha * s[i];
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
	if (tid == 0) atomicAdd(rhoNext, sdata[0]);
}



template <typename fp>
static __global__ void exitNormR(fp* rNext, fp* norm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	atomicAdd(norm, rNext[i] * rNext[i]);  //(не эффективно)
}

//Reordered iteration (no further opmimization)
template<typename fp>
class ConjGradCuda2 {

	CudaSparseSLAE<fp>& slae;

	fp* xNext = nullptr,
		* rNext = nullptr,
		* zNext = nullptr,
		* s = nullptr;

	fp _normB = {};

	//0 - rho, 1 - omega
	fp* dev_scalars = nullptr;

public:

	ConjGradCuda2(CudaSparseSLAE<fp>& sparseSlae)
		: slae(sparseSlae) {
		//cudaMalloc((void**)&xNext, slae.memLen * sizeof(fp));

		cudaMalloc((void**)&rNext, slae.memLen * sizeof(fp));
		cudaMalloc((void**)&zNext, slae.memLen * sizeof(fp));
		cudaMalloc((void**)&s, slae.memLen * sizeof(fp));
		cudaMemset(rNext, 0, slae.memLen * sizeof(fp));
		cudaMemset(zNext, 0, slae.memLen * sizeof(fp));

		_normB = 0.;
		fp* dev_normB = nullptr;
		cudaMalloc((void**)&dev_normB, sizeof(fp));
		cudaMemcpy(dev_normB, &_normB, sizeof(fp), cudaMemcpyHostToDevice);
		exitNormR<<<slae.memLen / BS, BS>>>(slae.rp, dev_normB);
		cudaMemcpy(&_normB, dev_normB, sizeof(fp), cudaMemcpyDeviceToHost);
		_normB = fp(1.) / _normB;
		//std::cout << "normB " << 1. / _normB << "\n";
		cudaFree(dev_normB);

		cudaMalloc((void**)&dev_scalars, 2 * sizeof(fp));
		cudaMemset(dev_scalars, 0, 2 * sizeof(fp));
	}

	~ConjGradCuda2() {
		cudaFree(rNext);
		cudaFree(zNext);
		cudaFree(s);

		cudaFree(dev_scalars);
	}

	void solve(fp* solution, bool* mask, size_t& iterNum, fp eps) {

		eps = eps * eps;
		fp rhoNext = 0., rhoPrev = 1.;

		int block = CGBS; std::min((int)CGBS, slae.N);
		int grid = slae.memLen / CGBS;

		xNext = solution;

		cudaStream_t stream;
		cudaStreamCreate(&stream);

		cgInit<fp><<<grid, block, 0, stream>>>(slae.data, slae.rows, slae.cols, slae.rp, xNext, rNext, zNext, dev_scalars, mask);
		//cudaDeviceSynchronize();
		cudaStreamSynchronize(stream);
		cudaMemcpy(&rhoNext, dev_scalars, sizeof(fp), cudaMemcpyDeviceToHost);

		while (rhoNext * _normB > eps) {

			//std::cin.get();

			cudaMemset(dev_scalars, 0, 2 * sizeof(fp));

			//double* rr = new double[slae.memLen];

			cg1<fp><<<grid, block>>>(rNext, zNext, rhoPrev, rhoNext);
			//cudaStreamSynchronize(stream);
			cudaDeviceSynchronize();
			
			cg2<fp><<<grid, block>>>(slae.data, slae.rows, slae.cols, zNext, s, dev_scalars + 1);
			//cudaStreamSynchronize(stream);
			cudaDeviceSynchronize();
			
			cg3<fp><<<grid, block>>>(xNext, rNext, zNext, s, dev_scalars + 1, rhoNext, dev_scalars, mask);
			//cudaStreamSynchronize(stream);
			cudaDeviceSynchronize();
			rhoPrev = rhoNext;
			cudaMemcpy(&rhoNext, dev_scalars, sizeof(fp), cudaMemcpyDeviceToHost);

			//delete[] rr;
			
			//if (sizeof(fp) == 8) std::cout << rhoNext << "\n";

			++iterNum;
		}

		cudaStreamDestroy(stream);
	}


	void solve2(fp* solution, bool* mask, size_t& iterNum, fp eps) {

		eps = eps * eps;
		fp rhoNext = 0., rhoPrev = 1.;

		fp* rhoNextPtr, * rhoPrevPtr;
		cudaMallocHost(&rhoNextPtr, sizeof(fp));
		cudaMallocHost(&rhoPrevPtr, sizeof(fp));

		int block = CGBS; std::min((int)CGBS, slae.N);
		int grid = slae.memLen / CGBS;

		xNext = solution;
		fp* dev_omega = dev_scalars + 1;

		cudaStream_t stream;
		cudaStreamCreate(&stream);

		cudaGraph_t graph;
		cudaGraphExec_t execGraph;

		//std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";

		cudaGraphCreate(&graph, 0);
		cudaGraphNode_t memsetNode, cg1Node, cg2Node, cg3Node, rhoNode, memcpyNode;

		
		/*cudaMemsetParams memsetParams;
		memsetParams.dst = dev_scalars;
		memsetParams.value = 0;
		memsetParams.pitch = 0;
		memsetParams.elementSize = sizeof(fp);
		memsetParams.width = 2;
		memsetParams.height = 1;
		cudaGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams);*/
		
		//std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";

		cudaKernelNodeParams cg1Params = { 0 };
		void* cg1Args[4] = { &rNext, &zNext, &rhoPrev, &rhoNext };
		cg1Params.func = (void*)cg1<fp>;
		cg1Params.gridDim = dim3(grid, 1, 1);
		cg1Params.blockDim = dim3(block, 1, 1);
		cg1Params.sharedMemBytes = 0;
		cg1Params.kernelParams = cg1Args;
		cg1Params.extra = nullptr;
		//cudaGraphAddKernelNode(&cg1Node, graph, { &memsetNode }, 1, &kernelNodeParams);
		cudaGraphAddKernelNode(&cg1Node, graph, nullptr, 0, &cg1Params);

		//std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";

		cudaKernelNodeParams cg2Params = { 0 };
		//memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
		void* cg2Args[6] = { &slae.data, &slae.rows, &slae.cols, &zNext, &s, &dev_omega };
		cg2Params.func = (void*)cg2<fp>;
		cg2Params.gridDim = dim3(grid, 1, 1);
		cg2Params.blockDim = dim3(block, 1, 1);
		cg2Params.sharedMemBytes = 0;
		cg2Params.kernelParams = cg2Args;
		cg2Params.extra = nullptr;
		cudaGraphAddKernelNode(&cg2Node, graph, { &cg1Node }, 1, &cg2Params);

		//std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";

		cudaKernelNodeParams cg3Params = { 0 };
		//memset(&kernelNodeParams, 0, sizeof(kernelNodeParams));
		void* cg3Args[8] = { &xNext, &rNext, &zNext, &s, &dev_omega, &rhoNext, &dev_scalars, &mask };
		cg3Params.func = (void*)cg3<fp>;
		cg3Params.gridDim = dim3(grid, 1, 1);
		cg3Params.blockDim = dim3(block, 1, 1);
		cg3Params.sharedMemBytes = 0;
		cg3Params.kernelParams = cg3Args;
		cg3Params.extra = nullptr;
		cudaGraphAddKernelNode(&cg3Node, graph, { &cg2Node }, 1, &cg3Params);

		//std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";

		//cudaGraphAddMemcpyNode1D(&rhoNode, graph, { &cg3Node }, 1, &rhoPrev, &rhoNext, sizeof(fp), cudaMemcpyHostToHost);

		//cudaGraphAddMemcpyNode1D(&memcpyNode, graph, { &rhoNode }, 1, &rhoNext, dev_scalars, sizeof(fp), cudaMemcpyDeviceToHost);
		
		cudaGraphInstantiate(&execGraph, graph, nullptr, nullptr, 0);
		
		cgInit<fp> << <grid, block>> > (slae.data, slae.rows, slae.cols, slae.rp, xNext, rNext, zNext, dev_scalars, mask);
		cudaDeviceSynchronize();
		//cudaStreamSynchronize(stream);
		cudaMemcpy(&rhoNext, dev_scalars, sizeof(fp), cudaMemcpyDeviceToHost);
		//std::cout << "host rho" << rhoNext << "\n";

		//std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";

		/*
		cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

		//cudaStreamSynchronize(stream);
		cg1<fp> << <grid, block, 0, stream>> > (rNext, zNext, rhoPrev, rhoNext);
		//cudaStreamSynchronize(stream);
		//cudaDeviceSynchronize();

		cg2<fp> << <grid, block, 0, stream >> > (slae.data, slae.rows, slae.cols, zNext, s, dev_scalars + 1);
		//cudaStreamSynchronize(stream);
		//cudaDeviceSynchronize();

		cg3<fp> << <grid, block, 0, stream >> > (xNext, rNext, zNext, s, dev_scalars + 1, rhoNext, dev_scalars, mask);
		cudaStreamEndCapture(stream, &graph);
		cudaGraphInstantiate(&execGraph, graph, nullptr, nullptr, 0);
		//cudaStreamSynchronize(stream);
		//cudaDeviceSynchronize();
		rhoPrev = rhoNext;
		//cudaMemcpyAsync(&rhoPrev, &rhoNext, sizeof(fp), cudaMemcpyHostToHost, stream);
		//cudaStreamSynchronize(stream);
		cudaMemcpy(&rhoNext, dev_scalars, sizeof(fp), cudaMemcpyDeviceToHost);
		//cudaStreamSynchronize(stream);
		*/
		

		//size_t num = 0;
		//cudaGraphNode_t* nodes;
		//cudaGraphGetNodes(graph, nodes, &num);
		//std::cout << "nodes: " << num << "\n";

		//rhoNext = 0., rhoPrev = 1.;
		
		

		while (rhoNext * _normB > eps) {

			//std::cin.get();

			cudaGraphExecKernelNodeSetParams(execGraph, cg1Node, &cg1Params);
			cudaGraphExecKernelNodeSetParams(execGraph, cg3Node, &cg3Params);

			//std::cout << "before\n";
			cudaMemset(dev_scalars, 0, 2 * sizeof(fp));
			cudaGraphLaunch(execGraph, stream);
			cudaStreamSynchronize(stream);
			//std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";
			//std::cout << "after\n";

			rhoPrev = rhoNext;
			cudaMemcpy(&rhoNext, dev_scalars, sizeof(fp), cudaMemcpyDeviceToHost);

			//if (sizeof(fp) == 8) std::cout << rhoNext << "\n";

			++iterNum;
		}

		cudaStreamDestroy(stream);
		cudaGraphExecDestroy(execGraph);
		cudaGraphDestroy(graph);

		cudaFreeHost(rhoNextPtr);
		cudaFreeHost(rhoPrevPtr);
	}


};




template<typename fp>
class ConjGradCudaG {

	CudaSparseSLAE<fp>& slae;

	fp* x = nullptr,
		* r = nullptr,
		* z = nullptr,
		* s = nullptr;

	bool* mask = nullptr;

	fp normB = {};

	//0 - omega, 1 - rhoNext, 2 - rhoPrev
	fp* dev_scalars = nullptr;
	fp* dev_omega = nullptr, * dev_rhoNext = nullptr, * dev_rhoPrev = nullptr;
	fp* rho = nullptr;

	cudaGraph_t graph;
	cudaGraphExec_t execGraph;
	cudaStream_t stream;

	cudaGraphNode_t cg1Node, rhoOnDevNode, memsetNode, cg2Node, cg3Node, rhoFromDevNode, memcpyNode;

public:

	ConjGradCudaG(CudaSparseSLAE<fp>& sparseSlae, fp* initSol, bool* _mask)
		: slae(sparseSlae), mask(_mask), x(initSol) {
		//cudaMalloc((void**)&xNext, slae.memLen * sizeof(fp));

		cudaMalloc((void**)&r, slae.memLen * sizeof(fp));
		cudaMalloc((void**)&z, slae.memLen * sizeof(fp));
		cudaMalloc((void**)&s, slae.memLen * sizeof(fp));
		//cudaMemset(r, 0, slae.memLen * sizeof(fp));
		//cudaMemset(z, 0, slae.memLen * sizeof(fp));

		normB = 0.;
		fp* dev_normB = nullptr;
		cudaMalloc((void**)&dev_normB, sizeof(fp));
		cudaMemset(dev_normB, 0, sizeof(fp));
		exitNormR << <slae.memLen / BS, BS >> > (slae.rp, dev_normB);
		cudaMemcpy(&normB, dev_normB, sizeof(fp), cudaMemcpyDeviceToHost);
		//std::cout << "normB " << normB << "\n";
		cudaFree(dev_normB);

		cudaMalloc((void**)&dev_scalars, 3 * sizeof(fp));
		cudaMemset(dev_scalars, 0, 3 * sizeof(fp));
		dev_omega = dev_scalars;
		dev_rhoNext = dev_scalars + 1;
		dev_rhoPrev = dev_scalars + 2;
		cudaMallocHost(&rho, sizeof(fp));

		cudaStreamCreate(&stream);

		int block = CGBS; std::min((int)CGBS, slae.N);
		int grid = slae.memLen / CGBS;

		//build graph
		cudaGraphCreate(&graph, 0);
		
		/*cudaKernelNodeParams kernelParams = { 0 };
		kernelParams.gridDim = dim3(grid, 1, 1);
		kernelParams.blockDim = dim3(block, 1, 1);
		kernelParams.sharedMemBytes = 0;
		kernelParams.extra = nullptr;*/

		cudaKernelNodeParams cg1Params = { 0 };
		void* cg1Args[4] = { &r, &z, &dev_rhoPrev, &dev_rhoNext };
		if constexpr (sizeof(fp) == 4) cg1Params.func = (void*)cg1g_f;
		else cg1Params.func = (void*)cg1g_d;
		cg1Params.gridDim = dim3(grid, 1, 1);
		cg1Params.blockDim = dim3(block, 1, 1);
		cg1Params.sharedMemBytes = 0;
		cg1Params.kernelParams = cg1Args;
		cg1Params.extra = nullptr;
		//cudaGraphAddKernelNode(&cg1Node, graph, { &memsetNode }, 1, &kernelNodeParams);
		cudaGraphAddKernelNode(&cg1Node, graph, nullptr, 0, &cg1Params);

		//std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";

		cudaGraphAddMemcpyNode1D(&rhoOnDevNode, graph, { &cg1Node }, 1, dev_rhoPrev, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToDevice);
		//cudaGraphAddMemcpyNode1D(&rhoOnDevNode, graph, nullptr, 0, dev_rhoPrev, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToDevice);

		cudaMemsetParams memsetParams = { 0 };
		memsetParams.dst = dev_scalars;
		memsetParams.value = 0;
		memsetParams.pitch = 0;
		memsetParams.elementSize = 1;
		memsetParams.width = 2 * sizeof(fp);
		memsetParams.height = 1;
		cudaGraphAddMemsetNode(&memsetNode, graph, { &rhoOnDevNode }, 1, &memsetParams);
		//cudaGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams);

		//memset(&kernelParams, 0, sizeof(kernelParams));
		cudaKernelNodeParams cg2Params = { 0 };
		void* cg2Args[6] = { &slae.data, &slae.rows, &slae.cols, &z, &s, &dev_omega };
		if constexpr (sizeof(fp) == 4) cg2Params.func = (void*)cg2g_f;
		else cg2Params.func = (void*)cg2g_d;
		cg2Params.kernelParams = cg2Args;
		cg2Params.gridDim = dim3(grid, 1, 1);
		cg2Params.blockDim = dim3(block, 1, 1);
		cg2Params.sharedMemBytes = 0;
		cg2Params.extra = nullptr;
		cudaGraphAddKernelNode(&cg2Node, graph, { &memsetNode }, 1, &cg2Params);
		//cudaGraphAddKernelNode(&cg2Node, graph, nullptr, 0, &cg2Params);

		//std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";

		//memset(&kernelParams, 0, sizeof(kernelParams));
		cudaKernelNodeParams cg3Params = { 0 };
		void* cg3Args[6] = { &x, &r, &z, &s, &dev_scalars, &mask };
		if constexpr (sizeof(fp) == 4) cg3Params.func = (void*)cg3g_f;
		else cg3Params.func = (void*)cg3g_d;
		cg3Params.kernelParams = cg3Args;
		cg3Params.gridDim = dim3(grid, 1, 1);
		cg3Params.blockDim = dim3(block, 1, 1);
		cg3Params.sharedMemBytes = 0;
		cg3Params.extra = nullptr;
		//auto error = cudaGraphAddKernelNode(&cg3Node, graph, nullptr, 0, &cg3Params);
		cudaGraphAddKernelNode(&cg3Node, graph, { &cg2Node }, 1, &cg3Params);

		//std::cout << cudaGetErrorString(error) << "\n";
		//cudaGraphAddMemcpyNode1D(&rhoFromDevNode, graph, { &cg3Node }, 1, rho, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToHost);


		cudaGraphInstantiate(&execGraph, graph, nullptr, nullptr, 0);
	}

	~ConjGradCudaG() {
		cudaStreamDestroy(stream);
		cudaGraphExecDestroy(execGraph);
		cudaGraphDestroy(graph);

		cudaFree(r);
		cudaFree(z);
		cudaFree(s);

		cudaFree(dev_scalars);
		cudaFreeHost(rho);
	}

	void solve(fp* solution, bool* mask, size_t& iterNum, fp eps) {

		x = solution;
		eps = eps * eps * normB;
		
		int block = CGBS; std::min((int)CGBS, slae.N);
		int grid = slae.memLen / CGBS;

		fp rhoPrev = 1.;
		cudaMemcpyAsync(dev_rhoPrev, &rhoPrev, sizeof(fp), cudaMemcpyHostToDevice, stream);
		if (sizeof(fp) == 4)
			cgInitG_f<<<grid, block, 0, stream>>>(slae.data, slae.rows, slae.cols, slae.rp, x, r, z, dev_rhoNext, mask);
		else
			cgInitG_d<<<grid, block, 0, stream>>>(slae.data, slae.rows, slae.cols, slae.rp, x, r, z, dev_rhoNext, mask);
		cudaStreamSynchronize(stream);
		cudaMemcpy(rho, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToHost);

		while (*rho > eps) {

			if constexpr (sizeof(fp) == 4) cg1g_f<<<grid, block>>>(r, z, dev_rhoPrev, dev_rhoNext);
			else cg1g_d<<<grid, block>>>(r, z, dev_rhoPrev, dev_rhoNext);
			cudaDeviceSynchronize();

			cudaMemcpy(dev_rhoPrev, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToDevice);
			cudaMemset(dev_scalars, 0, 2 * sizeof(fp));

			if constexpr (sizeof(fp) == 4) cg2g_f<<<grid, block>>>(slae.data, slae.rows, slae.cols, z, s, dev_omega);
			else cg2g_d<<<grid, block>>>(slae.data, slae.rows, slae.cols, z, s, dev_omega);
			cudaDeviceSynchronize();

			if constexpr (sizeof(fp) == 4) cg3g_f<<<grid, block>>>(x, r, z, s, dev_scalars, mask);
			else cg3g_d<<<grid, block>>>(x, r, z, s, dev_scalars, mask);
			cudaDeviceSynchronize();

			cudaMemcpy(rho, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToHost);
			
			
			++iterNum;
		}
	}


	void solve2(fp* solution, bool* mask, size_t& iterNum, fp eps) {

		//x = solution;
		eps = eps * eps * normB;

		int block = CGBS; std::min((int)CGBS, slae.N);
		int grid = slae.memLen / CGBS;

		fp rhoPrev = 1.;
		cudaMemcpyAsync(dev_rhoPrev, &rhoPrev, sizeof(fp), cudaMemcpyHostToDevice, stream);
		if constexpr (sizeof(fp) == 4)
			cgInitG_f<<<grid, block, 0, stream>>>(slae.data, slae.rows, slae.cols, slae.rp, x, r, z, dev_rhoNext, mask);
		else
			cgInitG_d<<<grid, block, 0, stream>>>(slae.data, slae.rows, slae.cols, slae.rp, x, r, z, dev_rhoNext, mask);
		cudaStreamSynchronize(stream);
		cudaMemcpy(rho, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToHost);

		while (*rho > eps) {

			cudaGraphLaunch(execGraph, stream);
			cudaStreamSynchronize(stream);

			cudaMemcpy(rho, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToHost);

			//std::cout << *rho << "\n";

			++iterNum;
		}
	}


};


template<typename fp>
class ConjGradCudaGW {

	CudaSparseSLAE<fp>& slae;

	fp* x = nullptr,
		* r = nullptr,
		* z = nullptr,
		* s = nullptr;

	bool* mask = nullptr;

	fp normB = {};
	fp eps2b = {};

	//0 - omega, 1 - rhoNext, 2 - rhoPrev
	fp* dev_scalars = nullptr;
	fp* dev_omega = nullptr, * dev_rhoNext = nullptr, * dev_rhoPrev = nullptr;
	fp* rho = nullptr;

	size_t* dev_iters = nullptr;

	cudaGraph_t bodyGraph, mainGraph;
	cudaGraphExec_t execGraph;
	cudaStream_t stream;

	cudaGraphConditionalHandle handle;

	cudaGraphNode_t initNode, conditionNode, loopNode, cg1Node, rhoOnDevNode, memsetNode, cg2Node, cg3Node, rhoFromDevNode, memcpyNode;

	cudaKernelNodeParams condParams = {};

	void* condArgs[4] = { &handle, &dev_rhoNext, &eps2b, &dev_iters };

public:

	ConjGradCudaGW(CudaSparseSLAE<fp>& sparseSlae, fp* initSol, bool* _mask)
		: slae(sparseSlae), mask(_mask), x(initSol) {
		//cudaMalloc((void**)&xNext, slae.memLen * sizeof(fp));

		cudaMalloc((void**)&r, slae.memLen * sizeof(fp));
		cudaMalloc((void**)&z, slae.memLen * sizeof(fp));
		cudaMalloc((void**)&s, slae.memLen * sizeof(fp));
		//cudaMemset(r, 0, slae.memLen * sizeof(fp));
		//cudaMemset(z, 0, slae.memLen * sizeof(fp));

		normB = 0.;
		fp* dev_normB = nullptr;
		cudaMalloc((void**)&dev_normB, sizeof(fp));
		cudaMemset(dev_normB, 0, sizeof(fp));
		exitNormR << <slae.memLen / BS, BS >> > (slae.rp, dev_normB);
		cudaMemcpy(&normB, dev_normB, sizeof(fp), cudaMemcpyDeviceToHost);
		//std::cout << "normB " << normB << "\n";
		cudaFree(dev_normB);

		cudaMalloc((void**)&dev_scalars, 3 * sizeof(fp));
		cudaMemset(dev_scalars, 0, 3 * sizeof(fp));
		dev_omega = dev_scalars;
		dev_rhoNext = dev_scalars + 1;
		dev_rhoPrev = dev_scalars + 2;
		cudaMallocHost(&rho, sizeof(fp));

		cudaMalloc(&dev_iters, sizeof(size_t));

		cudaStreamCreate(&stream);

		int block = CGBS; std::min((int)CGBS, slae.N);
		int grid = slae.memLen / CGBS;

		//build graph
		cudaGraphCreate(&mainGraph, 0);
		cudaGraphConditionalHandleCreate(&handle, mainGraph);

		cudaKernelNodeParams initParams = { 0 };
		void* initArgs[9] = { &slae.data, &slae.rows, &slae.cols, &slae.rp, &x, &r, &z, &dev_rhoNext, &mask};
		if constexpr (sizeof(fp) == 4) initParams.func = (void*)cgInitG_f;
		else initParams.func = (void*)cgInitG_d;
		initParams.gridDim = dim3(grid, 1, 1);
		initParams.blockDim = dim3(block, 1, 1);
		initParams.sharedMemBytes = 0;
		initParams.kernelParams = initArgs;
		initParams.extra = nullptr;
		cudaGraphAddKernelNode(&initNode, mainGraph, nullptr, 0, &initParams);

		if constexpr (sizeof(fp) == 4) condParams.func = (void*)loopCondition_f;
		else condParams.func = (void*)loopCondition_d;
		condParams.gridDim = dim3(1, 1, 1);
		condParams.blockDim = dim3(1, 1, 1);
		condParams.sharedMemBytes = 0;
		condParams.kernelParams = condArgs;
		condParams.extra = nullptr;
		cudaGraphAddKernelNode(&conditionNode, mainGraph, &initNode, 1, &condParams);

		cudaGraphNodeParams loopParams = { cudaGraphNodeTypeConditional };
		loopParams.conditional.handle = handle;
		loopParams.conditional.type = cudaGraphCondTypeWhile;
		loopParams.conditional.size = 1;
#if CUDART_VERSION >= 13000
		cudaGraphAddNode(&loopNode, mainGraph, &conditionNode, nullptr, 1, &loopParams);
#else
		cudaGraphAddNode(&loopNode, mainGraph, &conditionNode, 1, &loopParams);
#endif

		
		bodyGraph = loopParams.conditional.phGraph_out[0];


		cudaKernelNodeParams cg1Params = { 0 };
		void* cg1Args[4] = { &r, &z, &dev_rhoPrev, &dev_rhoNext };
		if constexpr (sizeof(fp) == 4) cg1Params.func = (void*)cg1g_f;
		else cg1Params.func = (void*)cg1g_d;
		cg1Params.gridDim = dim3(grid, 1, 1);
		cg1Params.blockDim = dim3(block, 1, 1);
		cg1Params.sharedMemBytes = 0;
		cg1Params.kernelParams = cg1Args;
		cg1Params.extra = nullptr;
		//cudaGraphAddKernelNode(&cg1Node, graph, { &memsetNode }, 1, &kernelNodeParams);
		cudaGraphAddKernelNode(&cg1Node, bodyGraph, nullptr, 0, &cg1Params);

		//std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";

		cudaGraphAddMemcpyNode1D(&rhoOnDevNode, bodyGraph, { &cg1Node }, 1, dev_rhoPrev, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToDevice);
		//cudaGraphAddMemcpyNode1D(&rhoOnDevNode, graph, nullptr, 0, dev_rhoPrev, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToDevice);

		cudaMemsetParams memsetParams = { 0 };
		memsetParams.dst = dev_scalars;
		memsetParams.value = 0;
		memsetParams.pitch = 0;
		memsetParams.elementSize = 1;
		memsetParams.width = 2 * sizeof(fp);
		memsetParams.height = 1;
		cudaGraphAddMemsetNode(&memsetNode, bodyGraph, { &rhoOnDevNode }, 1, &memsetParams);
		//cudaGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams);

		//memset(&kernelParams, 0, sizeof(kernelParams));
		cudaKernelNodeParams cg2Params = { 0 };
		void* cg2Args[6] = { &slae.data, &slae.rows, &slae.cols, &z, &s, &dev_omega };
		if constexpr (sizeof(fp) == 4) cg2Params.func = (void*)cg2g_f;
		else cg2Params.func = (void*)cg2g_d;
		cg2Params.kernelParams = cg2Args;
		cg2Params.gridDim = dim3(grid, 1, 1);
		cg2Params.blockDim = dim3(block, 1, 1);
		cg2Params.sharedMemBytes = 0;
		cg2Params.extra = nullptr;
		cudaGraphAddKernelNode(&cg2Node, bodyGraph, { &memsetNode }, 1, &cg2Params);
		//cudaGraphAddKernelNode(&cg2Node, graph, nullptr, 0, &cg2Params);

		//std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";

		//memset(&kernelParams, 0, sizeof(kernelParams));
		cudaKernelNodeParams cg3Params = { 0 };
		void* cg3Args[6] = { &x, &r, &z, &s, &dev_scalars, &mask };
		if constexpr (sizeof(fp) == 4) cg3Params.func = (void*)cg3g_f;
		else cg3Params.func = (void*)cg3g_d;
		cg3Params.kernelParams = cg3Args;
		cg3Params.gridDim = dim3(grid, 1, 1);
		cg3Params.blockDim = dim3(block, 1, 1);
		cg3Params.sharedMemBytes = 0;
		cg3Params.extra = nullptr;
		//auto error = cudaGraphAddKernelNode(&cg3Node, graph, nullptr, 0, &cg3Params);
		cudaGraphAddKernelNode(&cg3Node, bodyGraph, { &cg2Node }, 1, &cg3Params);

		//std::cout << cudaGetErrorString(error) << "\n";
		//cudaGraphAddMemcpyNode1D(&rhoFromDevNode, graph, { &cg3Node }, 1, rho, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToHost);

		cudaGraphAddKernelNode(&conditionNode, bodyGraph, &cg3Node, 1, &condParams);

		cudaGraphInstantiate(&execGraph, mainGraph, nullptr, nullptr, 0);
	}

	~ConjGradCudaGW() {
		cudaStreamDestroy(stream);
		cudaGraphExecDestroy(execGraph);
		cudaGraphDestroy(mainGraph);
		cudaGraphDestroy(bodyGraph);

		cudaFree(r);
		cudaFree(z);
		cudaFree(s);

		cudaFree(dev_scalars);
		cudaFree(dev_iters);
		cudaFreeHost(rho);
	}

	void solve(fp* solution, bool* mask, size_t& iterNum, fp eps) {

		x = solution;
		eps = eps * eps * normB;

		int block = CGBS; std::min((int)CGBS, slae.N);
		int grid = slae.memLen / CGBS;

		fp rhoPrev = 1.;
		cudaMemcpyAsync(dev_rhoPrev, &rhoPrev, sizeof(fp), cudaMemcpyHostToDevice, stream);
		if constexpr (sizeof(fp) == 4)
			cgInitG_f << <grid, block, 0, stream >> > (slae.data, slae.rows, slae.cols, slae.rp, x, r, z, dev_rhoNext, mask);
		else
			cgInitG_d << <grid, block, 0, stream >> > (slae.data, slae.rows, slae.cols, slae.rp, x, r, z, dev_rhoNext, mask);
		cudaStreamSynchronize(stream);
		cudaMemcpy(rho, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToHost);

		while (*rho > eps) {

			if constexpr (sizeof(fp) == 4) cg1g_f << <grid, block >> > (r, z, dev_rhoPrev, dev_rhoNext);
			else cg1g_d << <grid, block >> > (r, z, dev_rhoPrev, dev_rhoNext);
			cudaDeviceSynchronize();

			cudaMemcpy(dev_rhoPrev, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToDevice);
			cudaMemset(dev_scalars, 0, 2 * sizeof(fp));

			if constexpr (sizeof(fp) == 4) cg2g_f << <grid, block >> > (slae.data, slae.rows, slae.cols, z, s, dev_omega);
			else cg2g_d << <grid, block >> > (slae.data, slae.rows, slae.cols, z, s, dev_omega);
			cudaDeviceSynchronize();

			if constexpr (sizeof(fp) == 4) cg3g_f << <grid, block >> > (x, r, z, s, dev_scalars, mask);
			else cg3g_d << <grid, block >> > (x, r, z, s, dev_scalars, mask);
			cudaDeviceSynchronize();

			cudaMemcpy(rho, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToHost);


			++iterNum;
		}
	}


	void solve2(fp* solution, bool* mask, size_t& iterNum, fp eps) {

		//x = solution;
		eps2b = eps * eps * normB;

		cudaGraphExecKernelNodeSetParams(execGraph, conditionNode, &condParams);

		int block = CGBS; std::min((int)CGBS, slae.N);
		int grid = slae.memLen / CGBS;

		cudaMemset(dev_iters, 0, sizeof(size_t));

		fp rhoPrev = 1.;
		cudaMemcpyAsync(dev_rhoPrev, &rhoPrev, sizeof(fp), cudaMemcpyHostToDevice, stream);
		//cgInitG<fp> << <grid, block, 0, stream >> > (slae.data, slae.rows, slae.cols, slae.rp, x, r, z, dev_rhoNext, mask);
		cudaStreamSynchronize(stream);
		//cudaMemcpy(rho, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToHost);

		cudaGraphLaunch(execGraph, stream);
		cudaStreamSynchronize(stream);

		size_t iters;
		cudaMemcpy(&iters, dev_iters, sizeof(size_t), cudaMemcpyDeviceToHost);
		iterNum = iters - 1;
	}


};



// Vectorized by 2x2
template<typename fp>
class ConjGradCudaV2 {

	CudaSparseSLAE<fp>& slae;

	fp* x = nullptr,
		* r = nullptr,
		* z = nullptr,
		* s = nullptr;

	bool* mask = nullptr;

	fp normB = {};
	fp eps2b = {};

	//0 - omega, 1 - rhoNext, 2 - rhoPrev
	fp* dev_scalars = nullptr;
	fp* dev_omega = nullptr, * dev_rhoNext = nullptr, * dev_rhoPrev = nullptr;
	fp* rho = nullptr;

	size_t* dev_iters = nullptr;

	cudaGraph_t bodyGraph, mainGraph;
	cudaGraphExec_t execGraph;
	cudaStream_t stream;

	cudaGraphConditionalHandle handle;

	cudaGraphNode_t initNode, conditionNode, loopNode, cg1Node, rhoOnDevNode, memsetNode, cg2Node, cg3Node, rhoFromDevNode, memcpyNode;

	cudaKernelNodeParams condParams = {};

	void* condArgs[4] = { &handle, &dev_rhoNext, &eps2b, &dev_iters };

public:

	ConjGradCudaV2(CudaSparseSLAE<fp>& sparseSlae, fp* initSol, bool* _mask)
		: slae(sparseSlae), mask(_mask), x(initSol) {
		//cudaMalloc((void**)&xNext, slae.memLen * sizeof(fp));

		cudaMalloc((void**)&r, slae.memLen * sizeof(fp));
		cudaMalloc((void**)&z, slae.memLen * sizeof(fp));
		cudaMalloc((void**)&s, slae.memLen * sizeof(fp));
		//cudaMemset(r, 0, slae.memLen * sizeof(fp));
		//cudaMemset(z, 0, slae.memLen * sizeof(fp));

		normB = 0.;
		fp* dev_normB = nullptr;
		cudaMalloc((void**)&dev_normB, sizeof(fp));
		cudaMemset(dev_normB, 0, sizeof(fp));
		exitNormR << <slae.memLen / BS, BS >> > (slae.rp, dev_normB);
		cudaMemcpy(&normB, dev_normB, sizeof(fp), cudaMemcpyDeviceToHost);
		//std::cout << "normB " << normB << "\n";
		cudaFree(dev_normB);

		cudaMalloc((void**)&dev_scalars, 3 * sizeof(fp));
		cudaMemset(dev_scalars, 0, 3 * sizeof(fp));
		dev_omega = dev_scalars;
		dev_rhoNext = dev_scalars + 1;
		dev_rhoPrev = dev_scalars + 2;
		cudaMallocHost(&rho, sizeof(fp));

		cudaMalloc(&dev_iters, sizeof(size_t));

		cudaStreamCreate(&stream);

		int block = CGBS; std::min((int)CGBS, slae.N);
		int grid = slae.memLen / CGBS / 2;

		//build graph
		cudaGraphCreate(&mainGraph, 0);
		cudaGraphConditionalHandleCreate(&handle, mainGraph);

		//cudaGraphNode_t initNode;
		cudaKernelNodeParams initParams = { 0 };
		void* initArgs[9] = { &slae.data, &slae.rows, &slae.cols, &slae.rp, &x, &r, &z, &dev_rhoNext, &mask };
		if constexpr (sizeof(fp) == 4) initParams.func = (void*)cgInitGV_f;
		else initParams.func = (void*)cgInitGV_d;
		initParams.gridDim = dim3(2 * grid, 1, 1);
		initParams.blockDim = dim3(block, 1, 1);
		initParams.sharedMemBytes = 0;
		initParams.kernelParams = initArgs;
		initParams.extra = nullptr;
		cudaGraphAddKernelNode(&initNode, mainGraph, nullptr, 0, &initParams);

		if constexpr (sizeof(fp) == 4) condParams.func = (void*)loopCondition_f;
		else condParams.func = (void*)loopCondition_d;
		condParams.gridDim = dim3(1, 1, 1);
		condParams.blockDim = dim3(1, 1, 1);
		condParams.sharedMemBytes = 0;
		condParams.kernelParams = condArgs;
		condParams.extra = nullptr;
		cudaGraphAddKernelNode(&conditionNode, mainGraph, &initNode, 1, &condParams);

		cudaGraphNodeParams loopParams = { cudaGraphNodeTypeConditional };
		loopParams.conditional.handle = handle;
		loopParams.conditional.type = cudaGraphCondTypeWhile;
		loopParams.conditional.size = 1;
#if CUDART_VERSION >= 13000
		cudaGraphAddNode(&loopNode, mainGraph, &conditionNode, nullptr, 1, &loopParams);
#else
		cudaGraphAddNode(&loopNode, mainGraph, &conditionNode, 1, &loopParams);
#endif
		


		bodyGraph = loopParams.conditional.phGraph_out[0];


		cudaKernelNodeParams cg1Params = { 0 };
		void* cg1Args[4] = { &r, &z, &dev_rhoPrev, &dev_rhoNext };
		if constexpr (sizeof(fp) == 4) cg1Params.func = (void*)cg1gv_f;
		else cg1Params.func = (void*)cg1gv_d;
		cg1Params.gridDim = dim3(grid, 1, 1);
		cg1Params.blockDim = dim3(block, 1, 1);
		cg1Params.sharedMemBytes = 0;
		cg1Params.kernelParams = cg1Args;
		cg1Params.extra = nullptr;
		//cudaGraphAddKernelNode(&cg1Node, graph, { &memsetNode }, 1, &kernelNodeParams);
		cudaGraphAddKernelNode(&cg1Node, bodyGraph, nullptr, 0, &cg1Params);

		//std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";

		cudaGraphAddMemcpyNode1D(&rhoOnDevNode, bodyGraph, { &cg1Node }, 1, dev_rhoPrev, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToDevice);
		//cudaGraphAddMemcpyNode1D(&rhoOnDevNode, graph, nullptr, 0, dev_rhoPrev, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToDevice);

		cudaMemsetParams memsetParams = { 0 };
		memsetParams.dst = dev_scalars;
		memsetParams.value = 0;
		memsetParams.pitch = 0;
		memsetParams.elementSize = 1;
		memsetParams.width = 2 * sizeof(fp);
		memsetParams.height = 1;
		cudaGraphAddMemsetNode(&memsetNode, bodyGraph, { &rhoOnDevNode }, 1, &memsetParams);
		//cudaGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams);

		//memset(&kernelParams, 0, sizeof(kernelParams));
		cudaKernelNodeParams cg2Params = { 0 };
		void* cg2Args[6] = { &slae.data, &slae.rows, &slae.cols, &z, &s, &dev_omega };
		if constexpr(sizeof(fp) == 4) cg2Params.func = (void*)cg2g_fv;
		else cg2Params.func = (void*)cg2g_dv;
		cg2Params.kernelParams = cg2Args;
		cg2Params.gridDim = dim3(2 * grid, 1, 1);
		cg2Params.blockDim = dim3(block, 1, 1);
		cg2Params.sharedMemBytes = 0;
		cg2Params.extra = nullptr;
		cudaGraphAddKernelNode(&cg2Node, bodyGraph, { &memsetNode }, 1, &cg2Params);
		//cudaGraphAddKernelNode(&cg2Node, graph, nullptr, 0, &cg2Params);

		//std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";

		//memset(&kernelParams, 0, sizeof(kernelParams));
		cudaKernelNodeParams cg3Params = { 0 };
		void* cg3Args[6] = { &x, &r, &z, &s, &dev_scalars, &mask };
		if constexpr (sizeof(fp) == 4) cg3Params.func = (void*)cg3gv_f;
		else cg3Params.func = (void*)cg3gv_d;
		cg3Params.kernelParams = cg3Args;
		cg3Params.gridDim = dim3(grid, 1, 1);
		cg3Params.blockDim = dim3(block, 1, 1);
		cg3Params.sharedMemBytes = 0;
		cg3Params.extra = nullptr;
		//auto error = cudaGraphAddKernelNode(&cg3Node, graph, nullptr, 0, &cg3Params);
		cudaGraphAddKernelNode(&cg3Node, bodyGraph, &cg2Node, 1, &cg3Params);

		//std::cout << cudaGetErrorString(error) << "\n";
		//cudaGraphAddMemcpyNode1D(&rhoFromDevNode, graph, { &cg3Node }, 1, rho, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToHost);

		cudaGraphAddKernelNode(&conditionNode, bodyGraph, &cg3Node, 1, &condParams);

		cudaGraphInstantiate(&execGraph, mainGraph, nullptr, nullptr, 0);
	}

	~ConjGradCudaV2() {
		cudaStreamDestroy(stream);
		cudaGraphExecDestroy(execGraph);
		cudaGraphDestroy(mainGraph);
		cudaGraphDestroy(bodyGraph);

		cudaFree(r);
		cudaFree(z);
		cudaFree(s);

		cudaFree(dev_scalars);
		cudaFree(dev_iters);
		cudaFreeHost(rho);
	}


	void solve(fp* solution, bool* mask, size_t& iterNum, fp eps) {

		//x = solution;
		eps2b = eps * eps * normB;

		cudaGraphExecKernelNodeSetParams(execGraph, conditionNode, &condParams);

		int block = CGBS; std::min((int)CGBS, slae.N);
		int grid = slae.memLen / CGBS;

		cudaMemset(dev_iters, 0, sizeof(size_t));

		fp rhoPrev = 1.;
		cudaMemcpyAsync(dev_rhoPrev, &rhoPrev, sizeof(fp), cudaMemcpyHostToDevice, stream);
		//cgInitG<fp> << <grid, block, 0, stream >> > (slae.data, slae.rows, slae.cols, slae.rp, x, r, z, dev_rhoNext, mask);
		cudaStreamSynchronize(stream);
		//cudaMemcpy(rho, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToHost);

		cudaGraphLaunch(execGraph, stream);
		cudaStreamSynchronize(stream);

		size_t iters;
		cudaMemcpy(&iters, dev_iters, sizeof(size_t), cudaMemcpyDeviceToHost);
		iterNum = iters - 1;
	}


};



template<typename fp>
static __global__ void cgInitGT(fp* data, int* rows, int* cols, fp* rp, fp* x, fp* r, fp* z, fp* rhoNext, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ fp sdata[BS];
	fp sum = {};
	for (int j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * x[cols[j]];         //вычисление A.x0
	fp rk = mask[i] * (rp[i] - sum);
	z[i] = r[i] = rk;
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
	if (tid == 0) atomicAdd(rhoNext, sdata[0]);
}

template<typename fp>
static __global__ void cg1gt(fp* data, int* rows, int* cols, fp* z, fp* s, fp* omega) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ fp sdata[BS];

	fp sum = {};
	for (int j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * z[cols[j]];         //вычисление s
	s[i] = sum;
	sdata[tid] = sum * z[i];

	__syncthreads();

	//вычисление omega
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if (tid < 32) warpReduce(sdata, tid);
	if (tid == 0) atomicAdd(omega, sdata[0]);
}

template <typename fp>
static __global__ void cg2gt(fp* x, fp* r, fp* z, fp* s, fp* scalars, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	fp omega = *scalars, * rhoNext = scalars + 1, rhoPrev = scalars[2];
	__shared__ fp sdata[BS];
	fp alpha = rhoPrev / omega;  //вычисление alpha

	x[i] += alpha * z[i];       //вычисление xk
	fp rk = r[i] -= mask[i] * alpha * s[i];   //вычисление rk
	sdata[tid] = rk * rk;
	__syncthreads();

	//вычисление (rk, rk)
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if (tid < 32) warpReduce(sdata, tid);
	if (tid == 0) atomicAdd(rhoNext, sdata[0]);
}

template <typename fp>
static __global__ void cg3gt(fp* r, fp* z, fp *rhoPrev, fp *rhoNext) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	fp beta = *rhoNext / *rhoPrev;            //вычисление beta
	z[i] = r[i] + beta * z[i];                //вычисление zk
}


template<typename fp>
class ConjGradCudaGT {

	CudaSparseSLAE<fp>& slae;

	fp* x = nullptr,
		* r = nullptr,
		* z = nullptr,
		* s = nullptr;

	bool* mask = nullptr;

	fp normB = {};

	//0 - omega, 1 - rhoNext, 2 - rhoPrev
	fp* dev_scalars = nullptr;
	fp* dev_omega = nullptr, * dev_rhoNext = nullptr, * dev_rhoPrev = nullptr;
	fp* rho = nullptr;

	cudaGraph_t graph;
	cudaGraphExec_t execGraph;
	cudaStream_t stream;

	cudaGraphNode_t cg1Node, rhoOnDevNode, memsetNode, cg2Node, cg3Node, rhoFromDevNode, memcpyNode;

public:

	ConjGradCudaGT(CudaSparseSLAE<fp>& sparseSlae, fp* initSol, bool* _mask)
		: slae(sparseSlae), mask(_mask), x(initSol) {
		//cudaMalloc((void**)&xNext, slae.memLen * sizeof(fp));

		cudaMalloc((void**)&r, slae.memLen * sizeof(fp));
		cudaMalloc((void**)&z, slae.memLen * sizeof(fp));
		cudaMalloc((void**)&s, slae.memLen * sizeof(fp));
		//cudaMemset(r, 0, slae.memLen * sizeof(fp));
		//cudaMemset(z, 0, slae.memLen * sizeof(fp));

		normB = 0.;
		fp* dev_normB = nullptr;
		cudaMalloc((void**)&dev_normB, sizeof(fp));
		cudaMemset(dev_normB, 0, sizeof(fp));
		exitNormR << <slae.memLen / BS, BS >> > (slae.rp, dev_normB);
		cudaMemcpy(&normB, dev_normB, sizeof(fp), cudaMemcpyDeviceToHost);
		//std::cout << "normB " << normB << "\n";
		cudaFree(dev_normB);

		cudaMalloc((void**)&dev_scalars, 3 * sizeof(fp));
		cudaMemset(dev_scalars, 0, 3 * sizeof(fp));
		dev_omega = dev_scalars;
		dev_rhoNext = dev_scalars + 1;
		dev_rhoPrev = dev_scalars + 2;
		cudaMallocHost(&rho, sizeof(fp));

		cudaStreamCreate(&stream);

		int block = CGBS; std::min((int)CGBS, slae.N);
		int grid = slae.memLen / CGBS;

		//build graph
		cudaGraphCreate(&graph, 0);

		cudaGraphAddMemcpyNode1D(&rhoOnDevNode, graph, nullptr, 0, dev_rhoPrev, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToDevice);

		cudaMemsetParams memsetParams = { 0 };
		memsetParams.dst = dev_scalars;
		memsetParams.value = 0;
		memsetParams.pitch = 0;
		memsetParams.elementSize = 1;
		memsetParams.width = 2 * sizeof(fp);
		memsetParams.height = 1;
		cudaGraphAddMemsetNode(&memsetNode, graph, { &rhoOnDevNode }, 1, &memsetParams);
		//cudaGraphAddMemsetNode(&memsetNode, graph, nullptr, 0, &memsetParams);

		cudaKernelNodeParams cg1Params = { 0 };
		
		void* cg1Args[6] = { &slae.data, &slae.rows, &slae.cols, &z, &s, &dev_omega };
		cg1Params.func = (void*)cg1gt<fp>;
		cg1Params.kernelParams = cg1Args;
		cg1Params.gridDim = dim3(grid, 1, 1);
		cg1Params.blockDim = dim3(block, 1, 1);
		cg1Params.sharedMemBytes = 0;
		cg1Params.extra = nullptr;
		//auto error = cudaGraphAddKernelNode(&cg3Node, graph, nullptr, 0, &cg3Params);
		cudaGraphAddKernelNode(&cg1Node, graph, { &memsetNode }, 1, &cg1Params);

		cudaKernelNodeParams cg2Params = { 0 };
		void* cg2Args[6] = { &x, &r, &z, &s, &dev_scalars, &mask };
		//void* cg2Args[8] = { &x, &r, &z, &s, &dev_omega, &dev_rhoPrev, &dev_rhoNext, &mask };
		cg2Params.func = (void*)cg2gt<fp>;
		cg2Params.kernelParams = cg2Args;
		cg2Params.gridDim = dim3(grid, 1, 1);
		cg2Params.blockDim = dim3(block, 1, 1);
		cg2Params.sharedMemBytes = 0;
		cg2Params.extra = nullptr;
		cudaGraphAddKernelNode(&cg2Node, graph, { &cg1Node }, 1, &cg2Params);
		//cudaGraphAddKernelNode(&cg2Node, graph, nullptr, 0, &cg2Params);

		cudaKernelNodeParams cg3Params = { 0 };
		void* cg3Args[4] = { &r, &z, &dev_rhoPrev, &dev_rhoNext };
		cg3Params.func = (void*)cg3gt<fp>;
		cg3Params.gridDim = dim3(grid, 1, 1);
		cg3Params.blockDim = dim3(block, 1, 1);
		cg3Params.sharedMemBytes = 0;
		cg3Params.kernelParams = cg3Args;
		cg3Params.extra = nullptr;
		cudaGraphAddKernelNode(&cg3Node, graph, { &cg2Node }, 1, &cg3Params);
		//cudaGraphAddKernelNode(&cg1Node, graph, nullptr, 0, &cg1Params);

		//std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";

		//cudaGraphAddMemcpyNode1D(&rhoFromDevNode, graph, { &cg2Node }, 1, rho, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToHost);

		//memset(&kernelParams, 0, sizeof(kernelParams));
		

		//std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";

		//memset(&kernelParams, 0, sizeof(kernelParams));
		

		//std::cout << cudaGetErrorString(error) << "\n";
		


		cudaGraphInstantiate(&execGraph, graph, nullptr, nullptr, 0);
	}

	~ConjGradCudaGT() {
		cudaStreamDestroy(stream);
		cudaGraphExecDestroy(execGraph);
		cudaGraphDestroy(graph);

		cudaFree(r);
		cudaFree(z);
		cudaFree(s);

		cudaFree(dev_scalars);
		cudaFreeHost(rho);
	}

	void solve(fp* solution, bool* mask, size_t& iterNum, fp eps) {

		x = solution;
		eps = eps * eps * normB;

		int block = CGBS; std::min((int)CGBS, slae.N);
		int grid = slae.memLen / CGBS;

		fp rhoPrev = 1.;
		cudaMemcpyAsync(dev_rhoPrev, &rhoPrev, sizeof(fp), cudaMemcpyHostToDevice, stream);
		cgInitGT<fp><<<grid, block, 0, stream>>>(slae.data, slae.rows, slae.cols, slae.rp, x, r, z, dev_rhoNext, mask);
		cudaStreamSynchronize(stream);
		cudaMemcpy(rho, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToHost);

		while (*rho > eps) {

			cudaMemcpy(dev_rhoPrev, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToDevice);
			cudaMemset(dev_scalars, 0, 2 * sizeof(fp));

			cg1gt<fp><<<grid, block>>>(slae.data, slae.rows, slae.cols, z, s, dev_omega);
			cudaDeviceSynchronize();

			cg2gt<fp><<<grid, block>>>(x, r, z, s, dev_omega, dev_rhoPrev, dev_rhoNext, mask);
			cudaDeviceSynchronize();
			
			cudaMemcpyAsync(rho, dev_rhoPrev, sizeof(fp), cudaMemcpyDeviceToHost, stream);
			
			cg3gt<fp><<<grid, block, 0, stream>>>(r, z, dev_rhoPrev, dev_rhoNext);
			cudaStreamSynchronize(stream);

			++iterNum;
		}
	}

	void solve2(fp* solution, bool* mask, size_t& iterNum, fp eps) {

		x = solution;
		eps = eps * eps * normB;

		int block = CGBS; std::min((int)CGBS, slae.N);
		int grid = slae.memLen / CGBS;

		fp rhoPrev = 1.;
		cudaMemcpyAsync(dev_rhoPrev, &rhoPrev, sizeof(fp), cudaMemcpyHostToDevice, stream);
		cgInitGT<fp> << <grid, block, 0, stream >> > (slae.data, slae.rows, slae.cols, slae.rp, x, r, z, dev_rhoNext, mask);
		cudaStreamSynchronize(stream);
		cudaMemcpy(rho, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToHost);

		while (*rho > eps) {

			cudaGraphLaunch(execGraph, stream);
			cudaStreamSynchronize(stream);

			//cudaMemcpy(dev_rhoPrev, dev_rhoNext, sizeof(fp), cudaMemcpyDeviceToDevice);
			//cudaMemset(dev_scalars, 0, 2 * sizeof(fp));

			//cg1gt<fp> << <grid, block >> > (slae.data, slae.rows, slae.cols, z, s, dev_omega);
			//cudaDeviceSynchronize();

			//cg2gt<fp> << <grid, block >> > (x, r, z, s, dev_omega, dev_rhoPrev, dev_rhoNext, mask);
			//cudaDeviceSynchronize();

			cudaMemcpyAsync(rho, dev_rhoPrev, sizeof(fp), cudaMemcpyDeviceToHost, stream);

			//cg3gt<fp> << <grid, block, 0, stream >> > (r, z, dev_rhoPrev, dev_rhoNext);
			cudaStreamSynchronize(stream);

			++iterNum;
		}
	}
};


template <typename fp>
static __global__ void cg__1(fp* xNext, fp* xPrev,
	fp* rNext, fp* rPrev,
	fp* zNext, fp* zPrev,
	fp* data, int* rows, int* cols, bool* mask,
	fp rhoPrev, fp rhoCur, fp* s, fp* scalars) {

	fp* rhoNext = scalars, * omega = scalars + 1;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ fp sdata[CGBS];

	fp beta = rhoCur / rhoPrev;                  //вычисление beta
	zNext[i] = rPrev[i] + beta * zPrev[i];        //вычисление zk
	//if (i == 0) printf("%f\n", beta);

	//if (i < 32) printf("%f ", zNext[i]);

	//if (threadIdx.x == 0) printf("%d ", blockIdx.x);
	asm volatile("barrier.sync 0;" ::);
	//if (i == 0) printf("\n");

	fp sum = {};
	for (int j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * zNext[cols[j]];         //вычисление A.z
	s[i] = sum;
	sdata[tid] = sum * zNext[i];

	__syncthreads();

	//вычисление (A.z, z)
	for (unsigned s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if (tid < 32) warpReduce(sdata, tid);

	//asm volatile("barrier.sync 0;" ::);

	if (tid == 0) atomicAdd(omega, sdata[0]);
	//if (tid == 0) *omega += sdata[0];

	//if (threadIdx.x == 0) printf("%d ", blockIdx.x);
	asm volatile("barrier.sync 1;" ::);
	//if (i == 0) printf("\n");

	//if (i == 0) printf("\n\nomega   %e\n\n", *omega);

	fp alpha = rhoCur / *omega;  //вычисление alpha

	//if (i == 0) printf("rhoCur  %e\n", rhoCur);
	//if (i == 0) printf("alpha   %e\n", alpha);
	//if (i == 0) printf("rhoNext %f\n", *rhoNext);

	xNext[i] = xPrev[i] + alpha * zNext[i];       //вычисление xk
	fp rk = rPrev[i] - mask[i] * alpha * s[i];    //вычисление rk
	rNext[i] = rk;

	//if (i > 75 && i < 96) printf("%e ", *omega);
}

template <typename fp>
static __global__ void cg__2(fp* xNext, fp* xPrev,
	fp* rNext, fp* rPrev,
	fp* zNext, fp* zPrev,
	fp* data, int* rows, int* cols, bool* mask,
	fp rhoPrev, fp rhoCur, fp* s, fp* scalars) {

	fp* rhoNext = scalars, * omega = scalars + 1;

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ fp sdata[CGBS];

	fp alpha = rhoCur / *omega;  //вычисление alpha
	
	fp rk = rPrev[i] - mask[i] * alpha * s[i];    //вычисление rk
	rNext[i] = rk;

	//if (i == 0) printf("\n\nomega   %e\n\n", *omega);
	//if (i == 0) printf("alpha   %e\n", alpha);
	//if (i == 0) printf("rhoNext %f\n", *rhoNext);
	
	//if (i > 75 && i < 96) printf("%e ", *omega);
	//rk = rNext[i];
	sdata[tid] = rk * rk;
	//sdata[tid] = (i < 12160) ? rk * rk : 0.;
	//sdata[tid] = (i < 2048) ? rk * rk : 0.;
	__syncthreads();

	//вычисление (rk, rk)
	for (unsigned s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid < 32) warpReduce(sdata, tid);
	if (tid == 0) atomicAdd(rhoNext, sdata[0]);
}


template <typename fp>
static __global__ void cg__(fp* xNext, fp* xPrev,
	fp* rNext, fp* rPrev,
	fp* zNext, fp* zPrev,
	fp* data, int* rows, int* cols, bool* mask,
	fp rhoPrev, fp rhoCur, fp* s, fp* scalars) {

	fp* rhoNext = scalars, * omega = scalars + 1;

	cooperative_groups::grid_group grid = cooperative_groups::this_grid();
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ fp sdata[CGBS];

	fp beta = rhoCur / rhoPrev;                  //вычисление beta
	zNext[i] = rPrev[i] + beta * zPrev[i];        //вычисление zk
	//if (i == 0) printf("%f\n", beta);

	//if (i < 32) printf("%f ", zNext[i]);

	//if (threadIdx.x == 0) printf("%d ", blockIdx.x);
	grid.sync();
	//if (i == 0) printf("\n");

	fp sum = {};
	for (int j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * zNext[cols[j]];         //вычисление A.z
	s[i] = sum;
	sdata[tid] = sum * zNext[i];

	__syncthreads();

	//вычисление (A.z, z)
	for (unsigned s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if (tid < 32) warpReduce(sdata, tid);

	//asm volatile("barrier.sync 0;" ::);

	if (tid == 0) atomicAdd(omega, sdata[0]);
	//if (tid == 0) *omega += sdata[0];

	//if (threadIdx.x == 0) printf("%d ", blockIdx.x);
	grid.sync();
	//if (i == 0) printf("\n");

	//if (i == 0) printf("\n\nomega   %e\n\n", *omega);

	fp alpha = rhoCur / *omega;  //вычисление alpha

	//if (i == 0) printf("rhoCur  %e\n", rhoCur);
	//if (i == 0) printf("alpha   %e\n", alpha);
	//if (i == 0) printf("rhoNext %f\n", *rhoNext);

	xNext[i] = xPrev[i] + alpha * zNext[i];       //вычисление xk
	fp rk = rPrev[i] - mask[i] * alpha * s[i];    //вычисление rk
	rNext[i] = rk;

	//if (i > 75 && i < 96) printf("%e ", *omega);

	//fp alpha = rhoCur / *omega;  //вычисление alpha

	//fp rk = rPrev[i] - mask[i] * alpha * s[i];    //вычисление rk
	//rNext[i] = rk;

	//if (i == 0) printf("\n\nomega   %e\n\n", *omega);
	//if (i == 0) printf("alpha   %e\n", alpha);
	//if (i == 0) printf("rhoNext %f\n", *rhoNext);

	//if (i > 75 && i < 96) printf("%e ", *omega);
	//rk = rNext[i];
	sdata[tid] = rk * rk;
	//sdata[tid] = (i < 12160) ? rk * rk : 0.;
	//sdata[tid] = (i < 2048) ? rk * rk : 0.;
	__syncthreads();

	//вычисление (rk, rk)
	for (unsigned s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	if (tid < 32) warpReduce(sdata, tid);
	if (tid == 0) atomicAdd(rhoNext, sdata[0]);
}


template<typename fp>
class ConjGradCudaSync {

	CudaSparseSLAE<fp>& slae;

	fp* xNext = nullptr,
		* xPrev = nullptr,
		* rNext = nullptr,
		* rPrev = nullptr,
		* zNext = nullptr,
		* zPrev = nullptr,
		* s = nullptr;

	fp _normB = {};

	//0 - rho, 1 - omega
	fp* dev_scalars = nullptr;

public:

	ConjGradCudaSync(CudaSparseSLAE<fp>& sparseSlae)
		: slae(sparseSlae) {
		//cudaMalloc((void**)&xNext, slae.memLen * sizeof(fp));

		cudaMalloc((void**)&xPrev, slae.memLen * sizeof(fp));
		cudaMalloc((void**)&rNext, slae.memLen * sizeof(fp));
		cudaMalloc((void**)&rPrev, slae.memLen * sizeof(fp));
		cudaMalloc((void**)&zNext, slae.memLen * sizeof(fp));
		cudaMalloc((void**)&zPrev, slae.memLen * sizeof(fp));
		cudaMalloc((void**)&s, slae.memLen * sizeof(fp));
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

		cudaMalloc((void**)&dev_scalars, 2 * sizeof(fp));
		cudaMemset(dev_scalars, 0, 2 * sizeof(fp));
	}

	~ConjGradCudaSync() {
		//cudaFree(xNext); //?
		cudaFree(xPrev); //?

		cudaFree(rNext);
		cudaFree(rPrev);
		cudaFree(zNext);
		cudaFree(zPrev);
		cudaFree(s);

		cudaFree(dev_scalars);
	}

	void solve(fp* solution, bool* mask, size_t& iterNum, fp eps) {

		eps = eps * eps;
		fp rhoNext = 0., rhoPrev = 1.;

		int block = BS; std::min(BS, slae.N);
		int grid = slae.memLen / BS;

		xNext = solution;

		void* kernel_args[14] = {
			&xNext, &xPrev, &rNext, &rPrev, &zNext, &zPrev,
			&slae.data, &slae.rows, &slae.cols, &mask, &rhoPrev, &rhoNext,
			&s, &dev_scalars
		};

		cudaStream_t stream;
		cudaStreamCreate(&stream);

		cgInit<fp><<<grid, block, 0, stream>>>(slae.data, slae.rows, slae.cols, slae.rp, xNext, rNext, zNext, dev_scalars, mask);
		//cudaDeviceSynchronize();
		cudaStreamSynchronize(stream);
		cudaMemcpy(&rhoNext, dev_scalars, sizeof(fp), cudaMemcpyDeviceToHost);

		while (rhoNext * _normB > eps) {

			std::swap(xNext, xPrev);
			std::swap(rNext, rPrev);
			std::swap(zNext, zPrev);

			cudaMemsetAsync(dev_scalars, 0, 2 * sizeof(fp));

			//cg_1<fp><<<grid, block>>>(rPrev, zNext, zPrev, rhoPrev, rhoNext);
			//cudaStreamSynchronize(stream);
			//cudaDeviceSynchronize();
			
			//cg_2<fp><<<grid, block>>>(slae.data, slae.rows, slae.cols, zNext, s, dev_scalars + 1);
			//cudaStreamSynchronize(stream);
			//cudaDeviceSynchronize();
			
			//cg_3<fp><<<grid, block>>>(xNext, xPrev, rNext, rPrev, zNext, s, dev_scalars + 1, rhoPrev, dev_scalars, mask);
			//cudaStreamSynchronize(stream);

			//fp* r1 = new fp[slae.memLen];
			//fp* r2 = new fp[slae.memLen];

			/*cudaLaunchCooperativeKernel((void*)cg__1<fp>, grid, block, kernel_args, 0, stream);
			cudaStreamSynchronize(stream);
			cudaLaunchCooperativeKernel((void*)cg__2<fp>, grid, block, kernel_args, 0, stream);
			cudaStreamSynchronize(stream);*/

			cudaLaunchCooperativeKernel((void*)cg__<fp>, grid, block, kernel_args, 0, stream);
			cudaStreamSynchronize(stream);

			//cg__<fp><<<grid, block>>>(xNext, xPrev, rNext, rPrev, zNext, zPrev, \
				slae.data, slae.rows, slae.cols, mask, rhoPrev, rhoNext, dev_scalars, s, dev_scalars + 1);
			//cudaDeviceSynchronize();
			//cudaMemcpy(r1, rNext, slae.memLen * sizeof(fp), cudaMemcpyDeviceToHost);
			//cg__2<fp><<<grid, block>>>(xNext, xPrev, rNext, rPrev, zNext, zPrev, \
				slae.data, slae.rows, slae.cols, mask, rhoPrev, rhoNext, dev_scalars, s, dev_scalars + 1);
			//cudaDeviceSynchronize();
			//cudaMemcpy(r2, rNext, slae.memLen * sizeof(fp), cudaMemcpyDeviceToHost);
			rhoPrev = rhoNext;
			cudaMemcpy(&rhoNext, dev_scalars, sizeof(fp), cudaMemcpyDeviceToHost);

			/*for (int i = 0; i < slae.memLen; ++i) {
				if (r1[i] != 0 || r2[i] != 0)
					std::cout << i << "  " << r1[i] << "  " << r2[i] << "\n";
			}*/
			

			++iterNum;
			//if (iterNum == 1) break;
			//if (iterNum > 400) break;
			//if (sizeof(fp) == 8) std::cout << rhoNext << "\n";
			//std::cin.get();
		}

		//solution = xNext;
		if (iterNum % 2 == 1) std::swap(xNext, xPrev);
		//std::cout << "SLAE " << slae.N << " " << slae.memLen << "\n";
		//cudaMemset(xNext, 255, slae.memLen * sizeof(fp));
		//cudaMemcpy(solution, xNext, slae.N * sizeof(fp), cudaMemcpyDeviceToDevice);

		cudaStreamDestroy(stream);
	}


};