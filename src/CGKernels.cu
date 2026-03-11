#include "CGKernels.cuh"

#ifndef __CUDACC__  
#define __CUDACC__
#endif
#include <device_functions.h>
#include <cuda.h>

template <typename fp>
static __device__ void warpReduce(volatile fp* sdata, int tid) {
	sdata[tid] += sdata[tid + 32];
	sdata[tid] += sdata[tid + 16];
	sdata[tid] += sdata[tid + 8];
	sdata[tid] += sdata[tid + 4];
	sdata[tid] += sdata[tid + 2];
	sdata[tid] += sdata[tid + 1];
}

const unsigned CGBS = 512;

__global__ void cgInitG_f(float* data, int* rows, int* cols, float* rp, float* x, float* r, float* z, float* rhoNext, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ float sdata[CGBS];
	float sum = {};
	for (int j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * x[cols[j]];         //вычисление A.x0
	float rk = mask[i] * (rp[i] - sum);
	r[i] = rk;
	z[i] = 0;
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

__global__ void cgInitG_d(double* data, int* rows, int* cols, double* rp, double* x, double* r, double* z, double* rhoNext, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ double sdata[CGBS];
	double sum = {};
	for (int j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * x[cols[j]];         //вычисление A.x0
	double rk = mask[i] * (rp[i] - sum);
	r[i] = rk;
	z[i] = 0;
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

__global__ void cg1g_f(float* r, float* z, float* rhoPrev, float* rhoNext) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float beta = *rhoNext / *rhoPrev;                  //вычисление beta
	z[i] = r[i] + beta * z[i];        //вычисление zk
}

__global__ void cg1g_d(double* r, double* z, double* rhoPrev, double* rhoNext) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double beta = *rhoNext / *rhoPrev;                  //вычисление beta
	z[i] = r[i] + beta * z[i];        //вычисление zk
}

__global__ void cg2g_f(float* data, int* rows, int* cols, float* z, float* s, float* omega) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ float sdata[CGBS];

	//if (i == 0) printf("cg2\n");

	float sum = {};
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

__global__ void cg2g_d(double* data, int* rows, int* cols, double* z, double* s, double* omega) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ double sdata[CGBS];

	//if (i == 0) printf("cg2\n");

	double sum = {};
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

__global__ void cg2g_dv(double* data, int* rows, int* cols, double* z, double* s, double* omega) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ double sdata[CGBS];

	double sum = {};
	for (int j = rows[i]; j < rows[i + 1]; j += 2) {
		double2 zv = *reinterpret_cast<double2*>(z + cols[j]);
		double2 dv = *reinterpret_cast<double2*>(data + j);
		sum += zv.x * dv.x + zv.y * dv.y;
		//sum += data[j] * z[cols[j]];         //вычисление A.z
	}
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

__global__ void cg2g_fv(float* data, int* rows, int* cols, float* z, float* s, float* omega) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ float sdata[CGBS];

	float sum = {};
	for (int j = rows[i]; j < rows[i + 1]; j += 2) {
		float2 zv = *reinterpret_cast<float2*>(z + cols[j]);
		float2 dv = *reinterpret_cast<float2*>(data + j);
		sum += zv.x * dv.x + zv.y * dv.y;
		//sum += data[j] * z[cols[j]];         //вычисление A.z
	}
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


__global__ void cg3g_f(float* x, float* r, float* z, float* s, float* scalars, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	float omega = *scalars, * rhoNext = scalars + 1, rhoPrev = scalars[2];
	__shared__ float sdata[CGBS];
	//if (i == 0) printf("omega   %e\n\n", omega);
	float alpha = rhoPrev / omega;  //вычисление alpha

	//if (i == 0) printf("rhoPrCu %e\n", rhoPrev);
	//if (i == 0) printf("alpha   %e\n", alpha);
	//if (i == 0) printf("rhoNext %f\n", *rhoNext);
	x[i] += alpha * z[i];       //вычисление xk

	float rk = r[i] -= mask[i] * alpha * s[i];
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

__global__ void cg3g_d(double* x, double* r, double* z, double* s, double* scalars, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	double omega = *scalars, * rhoNext = scalars + 1, rhoPrev = scalars[2];
	__shared__ double sdata[CGBS];
	//if (i == 0) printf("omega   %e\n\n", omega);
	double alpha = rhoPrev / omega;  //вычисление alpha

	//if (i == 0) printf("rhoPrCu %e\n", rhoPrev);
	//if (i == 0) printf("alpha   %e\n", alpha);
	//if (i == 0) printf("rhoNext %f\n", *rhoNext);
	x[i] += alpha * z[i];       //вычисление xk

	double rk = r[i] -= mask[i] * alpha * s[i];
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

__global__ void cgInitGV_f(float* data, int* rows, int* cols, float* rp, float* x, float* r, float* z, float* rhoNext, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ float sdata[CGBS];
	float sum = {};
	for (int j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * x[cols[j]];         //вычисление A.x0
	float rk = mask[i] * (rp[i] - sum);
	r[i] = rk;
	z[i] = 0;
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

__global__ void cgInitGV_d(double* data, int* rows, int* cols, double* rp, double* x, double* r, double* z, double* rhoNext, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ double sdata[CGBS];
	double sum = {};
	for (int j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * x[cols[j]];         //вычисление A.x0
	double rk = mask[i] * (rp[i] - sum);
	r[i] = rk;
	z[i] = 0;
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

__global__ void cg1gv_d(double* r, double* z, double* rhoPrev, double* rhoNext) {
	int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	double beta = *rhoNext / *rhoPrev;                  //вычисление beta
	z[i] = r[i] + beta * z[i];        //вычисление zk
	z[i + 1] = r[i + 1] + beta * z[i + 1];        //вычисление zk

	double2 zn = {}, zv = {}, rv = {};
	*reinterpret_cast<float4*>(&zv) = *reinterpret_cast<float4*>(z + i);
	*reinterpret_cast<float4*>(&rv) = *reinterpret_cast<float4*>(r + i);

	zn.x = rv.x + beta * zv.x;
	zn.y = rv.y + beta * zv.y;
}


__global__ void cg1gv_f(float* r, float* z, float* rhoPrev, float* rhoNext) {
	int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	float beta = __fdividef(*rhoNext, *rhoPrev);                  //вычисление beta
	z[i] = r[i] + beta * z[i];        //вычисление zk
	z[i + 1] = r[i + 1] + beta * z[i + 1];        //вычисление zk

	float2 zn = {}, zv = {}, rv = {};
	*reinterpret_cast<float2*>(&zv) = *reinterpret_cast<float2*>(z + i);
	*reinterpret_cast<float2*>(&rv) = *reinterpret_cast<float2*>(r + i);

	zn.x = rv.x + beta * zv.x;
	zn.y = rv.y + beta * zv.y;
}

__global__ void cg2gv_f(float* data, int* rows, int* cols, float* z, float* s, float* omega) {
	int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	int tid = threadIdx.x;
	__shared__ float sdata[CGBS / 2];

	float sum1 = {}, sum2 = {};
	int shift = rows[i + 1] - rows[i];
	for (int j = rows[i]; j < rows[i + 1]; ++j) {
		float ze = z[cols[j]];
		sum1 += data[j] * ze;         //вычисление A.z
		sum2 += data[j + shift] * ze;
	}
	s[i] = sum1;
	s[i + 1] = sum2;
	sdata[tid] = sum1 * z[i] + sum2 * z[i + 1];

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


__global__ void cg2gv_d(double* data, int* rows, int* cols, double* z, double* s, double* omega) {
	int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	int tid = threadIdx.x;
	__shared__ double sdata[CGBS];

	double2 zv = {}, data1 = {}, data2 = {}, sum = {};
	int2 row = reinterpret_cast<int2*>(&rows[i])[0];

	int shift = row.y - row.x;
	//for (int j = row.x; j < row.y; j += 2) {
	//	*reinterpret_cast<float4*>(&zv) = *reinterpret_cast<float4*>(z + cols[j]);
	//	*reinterpret_cast<float4*>(&data1) = *reinterpret_cast<float4*>(data + j);
	//	*reinterpret_cast<float4*>(&data2) = *reinterpret_cast<float4*>(data + j + shift);
	//	sum.x += data1.x * zv.x + data1.y * zv.y;         //вычисление A.z
	//	sum.y += data2.x * zv.x + data2.y * zv.y;
	//}
	for (int j = rows[i]; j < rows[i + 1]; ++j) {
		double ze = z[cols[j]];
		sum.x += data[j] * ze;         //вычисление A.z
		sum.y += data[j + shift] * ze;
	}
	reinterpret_cast<float4*>(&s[i])[0] = reinterpret_cast<float4*>(&sum)[0];
	*reinterpret_cast<float4*>(&zv) = *reinterpret_cast<float4*>(z + i);
	//sdata[tid] = sum.x * z[i] + sum.y * z[i + 1];
	sdata[tid] = sum.x * zv.x + sum.y * zv.y;

	__syncthreads();

	//вычисление (A.z, z)
	for (unsigned s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s)
			sdata[tid] += sdata[tid + s];
		__syncthreads();
	}
	if (tid < 32) warpReduce(sdata, tid);
	if (tid == 0) atomicAdd(omega, sdata[0]);
}


__global__ void cg3gv_d(double* x, double* r, double* z, double* s, double* scalars, bool* mask) {
	int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	int tid = threadIdx.x;
	double omega = *scalars, * rhoNext = scalars + 1, rhoPrev = scalars[2];
	__shared__ double sdata[CGBS];
	double alpha = rhoPrev / omega;  //вычисление alpha

	double2 v = *reinterpret_cast<double2*>(z + i);

	x[i] += alpha * v.x;       //вычисление xk
	x[i + 1] += alpha * v.y;

	v = *reinterpret_cast<double2*>(s + i);

	double rk1 = r[i] -= mask[i] * alpha * v.x;
	double rk2 = r[i + 1] -= mask[i + 1] * alpha * v.y;
	sdata[tid] = rk1 * rk1 + rk2 * rk2;
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

__global__ void cg3gv_f(float* x, float* r, float* z, float* s, float* scalars, bool* mask) {
	int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	int tid = threadIdx.x;
	float omega = *scalars, * rhoNext = scalars + 1, rhoPrev = scalars[2];
	__shared__ float sdata[CGBS];
	float alpha = __fdividef(rhoPrev, omega);  //вычисление alpha

	float2 v = *reinterpret_cast<float2*>(z + i);

	x[i] += alpha * v.x;       //вычисление xk
	x[i + 1] += alpha * v.y;

	v = *reinterpret_cast<float2*>(s + i);

	float rk1 = r[i] -= mask[i] * alpha * v.x;
	float rk2 = r[i + 1] -= mask[i + 1] * alpha * v.y;
	sdata[tid] = rk1 * rk1 + rk2 * rk2;
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


__global__ void loopCondition_f(cudaGraphConditionalHandle handle, float* rho, float eps, size_t* iterations) {
	cudaGraphSetConditional(handle, *rho > eps);
	++(*iterations);
	//printf("%d ", *iterations);
}

__global__ void loopCondition_d(cudaGraphConditionalHandle handle, double* rho, double eps, size_t* iterations) {
	cudaGraphSetConditional(handle, *rho > eps);
	++(*iterations);
	//printf("%d ", *iterations);
}