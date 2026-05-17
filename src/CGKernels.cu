#include "CGKernels.cuh"

#ifndef __CUDACC__  
#define __CUDACC__
#endif
#include <device_functions.h>
#include <cuda.h>

#include <cmath>

#include <stdio.h>  // DEBUG

#include "DeviceCommon.cuh"

const unsigned CGBS = 512;


__global__ void isCloseToZero_f(cudaGraphConditionalHandle handle, float* num) {
	cudaGraphSetConditional(handle, *num < 1e-35f);
}

__global__ void isCloseToZero_d(cudaGraphConditionalHandle handle, double* num) {
	cudaGraphSetConditional(handle, *num < 1e-200);
}


__global__ void norm2_f(float* x, float* sum) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tid = threadIdx.x;
	__shared__ float sdata[CGBS];

	float xk = x[i];
	sdata[tid] = xk * xk;
	blockReduce(sdata, sum, tid);
}

__global__ void norm2_d(double* x, double* sum) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tid = threadIdx.x;
	__shared__ double sdata[CGBS];

	double xk = x[i];
	sdata[tid] = xk * xk;
	blockReduce(sdata, sum, tid);
}


__global__ void norm2v2_f(float* x, float* sum) {
	unsigned i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	unsigned tid = threadIdx.x;
	__shared__ float sdata[CGBS];

	float2 xk = *reinterpret_cast<float2*>(x + i);
	sdata[tid] = xk.x * xk.x + xk.y * xk.y;
	blockReduce(sdata, sum, tid);
}

__global__ void norm2v2_d(double* x, double* sum) {
	unsigned i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	unsigned tid = threadIdx.x;
	__shared__ double sdata[CGBS];

	double2 xk = *reinterpret_cast<double2*>(x + i);
	sdata[tid] = xk.x * xk.x + xk.y + xk.y;
	blockReduce(sdata, sum, tid);
}


__global__ void norm2v3_f(float* x, float* sum) {
	unsigned i = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
	unsigned tid = threadIdx.x;
	__shared__ float sdata[CGBS];

	float3 xk = *reinterpret_cast<float3*>(x + i);
	sdata[tid] = xk.x * xk.x + xk.y * xk.y + xk.z + xk.z;
	blockReduce(sdata, sum, tid);
}

__global__ void norm2v3_d(double* x, double* sum) {
	unsigned i = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
	unsigned tid = threadIdx.x;
	__shared__ double sdata[CGBS];

	double3 xk = *reinterpret_cast<double3*>(x + i);
	sdata[tid] = xk.x * xk.x + xk.y + xk.y + xk.z + xk.z;
	blockReduce(sdata, sum, tid);
}


__global__ void norm2P_f(float* rhs, float* DR, float* sum) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tid = threadIdx.x;
	__shared__ float sdata[CGBS];

	float rk = rhs[i];
	sdata[tid] = DR[i] * rk * rk;
	blockReduce(sdata, sum, tid);
}

__global__ void norm2P_d(double* rhs, double* DR, double* sum) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tid = threadIdx.x;
	__shared__ double sdata[CGBS];

	double rk = rhs[i];
	sdata[tid] = DR[i] * rk * rk;
	blockReduce(sdata, sum, tid);
}


__global__ void dotProduct_f(float* a, float* b, float* sum) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tid = threadIdx.x;
	__shared__ float sdata[CGBS];

	sdata[tid] = a[i] * b[i];
	blockReduce(sdata, sum, tid);
}

__global__ void dotProduct_d(double* a, double* b, double* sum) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tid = threadIdx.x;
	__shared__ double sdata[CGBS];

	sdata[tid] = a[i] * b[i];
	blockReduce(sdata, sum, tid);
}

__global__ void dotProductV2_f(float* a, float* b, float* sum) {
	unsigned i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	unsigned tid = threadIdx.x;
	__shared__ float sdata[CGBS];

	float2 ak = *reinterpret_cast<float2*>(a + i);
	float2 bk = *reinterpret_cast<float2*>(b + i);
	sdata[tid] = ak.x * bk.x + ak.y * bk.y;
	blockReduce(sdata, sum, tid);
}

__global__ void dotProductV2_d(double* a, double* b, double* sum) {
	unsigned i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	unsigned tid = threadIdx.x;
	__shared__ double sdata[CGBS];

	double2 ak = *reinterpret_cast<double2*>(a + i);
	double2 bk = *reinterpret_cast<double2*>(b + i);
	sdata[tid] = ak.x * bk.x + ak.y * bk.y;
	blockReduce(sdata, sum, tid);
}

__global__ void dotProductV3_f(float* a, float* b, float* sum) {
	unsigned i = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
	unsigned tid = threadIdx.x;
	__shared__ float sdata[CGBS];

	float3 ak = *reinterpret_cast<float3*>(a + i);
	float3 bk = *reinterpret_cast<float3*>(b + i);
	sdata[tid] = ak.x * bk.x + ak.y * bk.y + ak.z * bk.z;
	blockReduce(sdata, sum, tid);
}

__global__ void dotProductV3_d(double* a, double* b, double* sum) {
	unsigned i = 3 * (blockIdx.x * blockDim.x + threadIdx.x);
	unsigned tid = threadIdx.x;
	__shared__ double sdata[CGBS];

	double3 ak = *reinterpret_cast<double3*>(a + i);
	double3 bk = *reinterpret_cast<double3*>(b + i);
	sdata[tid] = ak.x * bk.x + ak.y * bk.y + ak.z * bk.z;
	blockReduce(sdata, sum, tid);
}


__global__ void precond_f(unsigned N, float* data, unsigned* rows, float* DR) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

	float sum = 0.;
	for (unsigned j = rows[i]; j < rows[i + 1]; ++j) {
		float el = data[j];
		sum += el * el;
	}
	DR[i] = i < N ? 1.f / sqrtf(sum) : 0.f;
}

__global__ void precond_d(unsigned N, double* data, unsigned* rows, double* DR) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;

	double sum = 0.;
	for (unsigned j = rows[i]; j < rows[i + 1]; ++j) {
		double el = data[j];
		sum += el * el;
	}
	DR[i] = i < N ? 1. / sqrt(sum) : 0.;
}


__global__ void precondAndNorm_f(unsigned N, float* data, unsigned* rows, float* DR, float* rhs, float* normB) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tid = threadIdx.x;
	__shared__ float sdata[CGBS];

	float sum = 0.;
	for (unsigned j = rows[i]; j < rows[i + 1]; ++j) {
		float el = data[j];
		sum += el * el;
	}
	float dr = i < N ? 1. / sqrt(sum) : 0.;
	DR[i] = dr;

	float rk = rhs[i];
	sdata[tid] = dr * rk * rk;
	blockReduce(sdata, normB, tid);
}

__global__ void precondAndNorm_d(unsigned N, double* data, unsigned* rows, double* DR, double* rhs, double* normB) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tid = threadIdx.x;
	__shared__ double sdata[CGBS];

	double sum = 0.;
	for (unsigned j = rows[i]; j < rows[i + 1]; ++j) {
		double el = data[j];
		sum += el * el;
	}
	double dr = i < N ? 1. / sqrt(sum) : 0.;
	DR[i] = dr;

	double rk = rhs[i];
	sdata[tid] = dr * rk * rk;
	blockReduce(sdata, normB, tid);
}


__global__ void updateEps_f(float* cond) {
	float e = cond[1];
	cond[2] = e * e * cond[0];
	//printf("%e %e %e", cond[0], cond[1], cond[2]);
}

__global__ void updateEps_d(double* cond) {
	double e = cond[1];
	cond[2] = e * e * cond[0];
	//printf("%e %e %e", cond[0], cond[1], cond[2]);
}


__global__ void cgInit_f(float* data, int* rows, int* cols, float* rhs, float* x, float* r, float* z, float* rhoNext, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ float sdata[CGBS];
	float sum = {};
	for (int j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * x[cols[j]];         //вычисление A.x0
	float rk = mask[i] * (rhs[i] - sum);
	r[i] = rk;
	z[i] = 0;
	sdata[tid] = rk * rk;  //вычисление (r0, r0)
	blockReduce(sdata, rhoNext, tid);
}

__global__ void cgInit_d(double* data, int* rows, int* cols, double* rhs, double* x, double* r, double* z, double* rhoNext, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ double sdata[CGBS];
	double sum = {};
	for (int j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * x[cols[j]];         //вычисление A.x0
	double rk = mask[i] * (rhs[i] - sum);
	r[i] = rk;
	z[i] = 0;
	sdata[tid] = rk * rk;  //вычисление (r0, r0)
	blockReduce(sdata, rhoNext, tid);
}

__global__ void cgInitV2_f(float* data, int* rows, int* cols, float* rhs, float* x, float* r, float* z, float* rhoNext, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ float sdata[CGBS];
	float sum = {};
	for (int j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * x[cols[j]];         //вычисление A.x0
	float rk = mask[i] * (rhs[i] - sum);
	r[i] = rk;
	z[i] = 0;
	sdata[tid] = rk * rk;  //вычисление (r0, r0)
	blockReduce(sdata, rhoNext, tid);
}

__global__ void cgInitV2_d(double* data, int* rows, int* cols, double* rhs, double* x, double* r, double* z, double* rhoNext, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ double sdata[CGBS];
	double sum = {};
	for (int j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * x[cols[j]];         //вычисление A.x0
	double rk = mask[i] * (rhs[i] - sum);
	r[i] = rk;
	z[i] = 0;
	sdata[tid] = rk * rk;  //вычисление (r0, r0)
	blockReduce(sdata, rhoNext, tid);
}


__global__ void cgInitP_f(float* data, int* rows, int* cols, float* rhs, float* x, float* r, float* z, float* q, float* DR, float* rhoNext, bool* mask) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tid = threadIdx.x;
	__shared__ float sdata[CGBS];
	float sum = {};
	for (unsigned j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * x[cols[j]];         //вычисление A.x0

	float rk = mask[i] * (rhs[i] - sum);
	r[i] = rk;
	float qk = q[i] = DR[i] * rk;
	z[i] = 0;
	sdata[tid] = qk * rk;  //вычисление (r0, r0)
	blockReduce(sdata, rhoNext, tid);
}

__global__ void cgInitP_d(double* data, int* rows, int* cols, double* rhs, double* x, double* r, double* z, double* q, double* DR, double* rhoNext, bool* mask) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tid = threadIdx.x;
	__shared__ double sdata[CGBS];
	double sum = {};
	for (unsigned j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * x[cols[j]];         //вычисление A.x0

	double rk = mask[i] * (rhs[i] - sum);
	r[i] = rk;
	double qk = q[i] = DR[i] * rk;
	z[i] = 0;
	sdata[tid] = qk * rk;  //вычисление (r0, r0)
	blockReduce(sdata, rhoNext, tid);
}


__global__ void cgInitLines_f(float* data, int* rows, int* cols, float* rhs, float* x, float* r, float* z, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	float sum = {};
	for (int j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * x[cols[j]];         //вычисление A.x0
	float rk = mask[i] * (rhs[i] - sum);
	r[i] = rk;
	z[i] = 0;
}

__global__ void cgInitLines_d(double* data, int* rows, int* cols, double* rhs, double* x, double* r, double* z, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	double sum = {};
	for (int j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * x[cols[j]];         //вычисление A.x0
	double rk = mask[i] * (rhs[i] - sum);
	r[i] = rk;
	z[i] = 0;
}


__global__ void cgInitLinesP_f(float* data, int* rows, int* cols, float* rhs, float* x, float* r, float* z, float* q, float* DR, bool* mask) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tid = threadIdx.x;
	float sum = {};
	for (unsigned j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * x[cols[j]];         //вычисление A.x0

	float rk = mask[i] * (rhs[i] - sum);
	r[i] = rk;
	float qk = q[i] = DR[i] * rk;
	z[i] = 0;
}

__global__ void cgInitLinesP_d(double* data, int* rows, int* cols, double* rhs, double* x, double* r, double* z, double* q, double* DR, bool* mask) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tid = threadIdx.x;
	double sum = {};
	for (unsigned j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * x[cols[j]];         //вычисление A.x0

	double rk = mask[i] * (rhs[i] - sum);
	r[i] = rk;
	double qk = q[i] = DR[i] * rk;
	z[i] = 0;
}


//TODO: оптимизировать обращение к линиям
__global__ void linesV2_f(unsigned lineCount, float* r, float* lines, unsigned* lineRows) {
	unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= lineCount) return;
	unsigned i = lineRows[k];
	float2 rk = *reinterpret_cast<float2*>(r + i);
	float2 lk = *reinterpret_cast<float2*>(lines + 2 * k);
	float len = rk.x * lk.x + rk.y * lk.y;
	rk.x = lk.x * len;
	rk.y = lk.y * len;
	*reinterpret_cast<float2*>(r + i) = rk;
}

__global__ void linesV2_d(unsigned lineCount, double* r, double* lines, unsigned* lineRows) {
	unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= lineCount) return;
	unsigned i = lineRows[k];
	double2 rk = *reinterpret_cast<double2*>(r + i);
	double2 lk = *reinterpret_cast<double2*>(lines + 2 * k);
	double len = rk.x * lk.x + rk.y * lk.y;
	rk.x = lk.x * len;
	rk.y = lk.y * len;
	*reinterpret_cast<double2*>(r + i) = rk;
}

__global__ void linesP2_f(unsigned lineCount, float* r, float* q, float* lines, unsigned* lineRows) {
	unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= lineCount) return;
	unsigned i = lineRows[k];
	float2 rk = *reinterpret_cast<float2*>(r + i);
	float2 lk = *reinterpret_cast<float2*>(lines + 2 * k);
	float len = rk.x * lk.x + rk.y * lk.y;
	rk.x = lk.x * len;
	rk.y = lk.y * len;
	*reinterpret_cast<float2*>(r + i) = rk;
	rk = *reinterpret_cast<float2*>(q + i);
	len = rk.x * lk.x + rk.y * lk.y;
	rk.x = lk.x * len;
	rk.y = lk.y * len;
	*reinterpret_cast<float2*>(q + i) = rk;
}

__global__ void linesP2_d(unsigned lineCount, double* r, double* q, double* lines, unsigned* lineRows) {
	unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
	if (k >= lineCount) return;
	unsigned i = lineRows[k];
	double2 rk = *reinterpret_cast<double2*>(r + i);
	double2 lk = *reinterpret_cast<double2*>(lines + 2 * k);
	double len = rk.x * lk.x + rk.y * lk.y;
	rk.x = lk.x * len;
	rk.y = lk.y * len;
	*reinterpret_cast<double2*>(r + i) = rk;
	rk = *reinterpret_cast<double2*>(q + i);
	len = rk.x * lk.x + rk.y * lk.y;
	rk.x = lk.x * len;
	rk.y = lk.y * len;
	*reinterpret_cast<double2*>(q + i) = rk;
}


__global__ void cg1_f(float* r, float* z, float* rhoPrev, float* rhoNext) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float beta = *rhoNext / *rhoPrev;                  //вычисление beta
	z[i] = r[i] + beta * z[i];        //вычисление zk
	//if (i == 0) printf("beta = %e\n", beta);
	//if (i == 0) printf("rhoPrev = %e\n", *rhoPrev);
}

__global__ void cg1_d(double* r, double* z, double* rhoPrev, double* rhoNext) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	double beta = *rhoNext / *rhoPrev;                  //вычисление beta
	z[i] = r[i] + beta * z[i];        //вычисление zk
}

__global__ void cg1v2_d(double* r, double* z, double* rhoPrev, double* rhoNext) {
	int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	double beta = *rhoNext / *rhoPrev;                  //вычисление beta
	z[i] = r[i] + beta * z[i];        //вычисление zk
	z[i + 1] = r[i + 1] + beta * z[i + 1];        //вычисление zk

	double2 zn = {}, zv = {}, rv = {};
	zv = *reinterpret_cast<double2*>(z + i);
	rv = *reinterpret_cast<double2*>(r + i);

	zn.x = rv.x + beta * zv.x;
	zn.y = rv.y + beta * zv.y;
}


__global__ void cg1v2_f(float* r, float* z, float* rhoPrev, float* rhoNext) {
	int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	float beta = __fdividef(*rhoNext, *rhoPrev);                  //вычисление beta
	z[i] = r[i] + beta * z[i];        //вычисление zk
	z[i + 1] = r[i + 1] + beta * z[i + 1];        //вычисление zk

	float2 zn = {}, zv = {}, rv = {};
	zv = *reinterpret_cast<float2*>(z + i);
	rv = *reinterpret_cast<float2*>(r + i);

	zn.x = rv.x + beta * zv.x;
	zn.y = rv.y + beta * zv.y;
}


__global__ void cg2_f(float* data, int* rows, int* cols, float* z, float* s, float* omega) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ float sdata[CGBS];

	//if (i == 0) printf("cg2\n");

	float sum = {};
	for (int j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * z[cols[j]];         //вычисление A.z
	s[i] = sum;
	sdata[tid] = sum * z[i];  //вычисление (A.z, z)
	blockReduce(sdata, omega, tid);
}

__global__ void cg2_d(double* data, int* rows, int* cols, double* z, double* s, double* omega) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ double sdata[CGBS];

	//if (i == 0) printf("cg2\n");

	double sum = {};
	for (int j = rows[i]; j < rows[i + 1]; ++j)
		sum += data[j] * z[cols[j]];         //вычисление A.z
	s[i] = sum;
	sdata[tid] = sum * z[i];  //вычисление (A.z, z)
	blockReduce(sdata, omega, tid);
}

__global__ void cg2_dv(double* data, int* rows, int* cols, double* z, double* s, double* omega) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ double sdata[CGBS];

	double sum = {};
	for (int j = rows[i]; j < rows[i + 1]; j += 2) {
		double2 zv = *reinterpret_cast<double2*>(z + cols[j]);
		double2 dv = *reinterpret_cast<double2*>(data + j);
		sum += zv.x * dv.x;
		sum += zv.y * dv.y;
		//sum += data[j] * z[cols[j]];         //вычисление A.z
	}
	s[i] = sum;
	sdata[tid] = sum * z[i];  //вычисление (A.z, z)
	blockReduce(sdata, omega, tid);
}

__global__ void cg2_fv(float* data, int* rows, int* cols, float* z, float* s, float* omega) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ float sdata[CGBS];

	float sum = {};
	for (int j = rows[i]; j < rows[i + 1]; j += 2) {
		float2 zv = *reinterpret_cast<float2*>(z + cols[j]);
		float2 dv = *reinterpret_cast<float2*>(data + j);
		sum += zv.x * dv.x;
		sum += zv.y * dv.y;
		//sum += data[j] * z[cols[j]];         //вычисление A.z
	}
	s[i] = sum;
	sdata[tid] = sum * z[i];  //вычисление (A.z, z)
	blockReduce(sdata, omega, tid);
}


__global__ void cg2v2_f(float* data, int* rows, int* cols, float* z, float* s, float* omega) {
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
	sdata[tid] = sum1 * z[i] + sum2 * z[i + 1];  //вычисление (A.z, z)
	blockReduce(sdata, omega, tid);
}


__global__ void cg2v2_d(double* data, int* rows, int* cols, double* z, double* s, double* omega) {
	int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	int tid = threadIdx.x;
	__shared__ double sdata[CGBS / 2];

	double2 zv = {}, data1 = {}, data2 = {}, sum = {};
	int2 row = reinterpret_cast<int2*>(&rows[i])[0];

	int shift = row.y - row.x;
	//for (int j = row.x; j < row.y; j += 2) {
	//	zv = *reinterpret_cast<double2*>(z + cols[j]);
	//	data1 = *reinterpret_cast<double2*>(data + j);
	//	data2 = *reinterpret_cast<double2*>(data + j + shift);
	//	sum.x += data1.x * zv.x + data1.y * zv.y;         //вычисление A.z
	//	sum.y += data2.x * zv.x + data2.y * zv.y;
	//}
	for (int j = rows[i]; j < rows[i + 1]; ++j) {
		double ze = z[cols[j]];
		sum.x += data[j] * ze;         //вычисление A.z
		sum.y += data[j + shift] * ze;
	}
	reinterpret_cast<double2*>(&s[i])[0] = reinterpret_cast<double2*>(&sum)[0];
	zv = *reinterpret_cast<double2*>(z + i);
	//sdata[tid] = sum.x * z[i] + sum.y * z[i + 1];
	sdata[tid] = sum.x * zv.x + sum.y * zv.y;   //вычисление (A.z, z)
	blockReduce(sdata, omega, tid);
}


__global__ void cg3_f(float* x, float* r, float* z, float* s, float* scalars, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	float omega = *scalars, * rhoNext = scalars + 1, rhoPrev = scalars[2];
	__shared__ float sdata[CGBS];
	float alpha = rhoPrev / omega;  //вычисление alpha
	x[i] += alpha * z[i];       //вычисление xk

	float rk = r[i] -= mask[i] * alpha * s[i];
	sdata[tid] = rk * rk;   //вычисление (rk, rk)
	blockReduce(sdata, rhoNext, tid);
}

__global__ void cg3_d(double* x, double* r, double* z, double* s, double* scalars, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	double omega = *scalars, * rhoNext = scalars + 1, rhoPrev = scalars[2];
	__shared__ double sdata[CGBS];
	double alpha = rhoPrev / omega;  //вычисление alpha
	x[i] += alpha * z[i];       //вычисление xk

	double rk = r[i] -= mask[i] * alpha * s[i];
	sdata[tid] = rk * rk;  //вычисление (rk, rk)
	blockReduce(sdata, rhoNext, tid);
}


__global__ void cg3P_f(float* x, float* r, float* z, float* s, float* q, float* DR, float* scalars, bool* mask) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tid = threadIdx.x;
	float omega = *scalars, * rhoNext = scalars + 1, rhoPrev = scalars[2];
	__shared__ float sdata[CGBS];
	float alpha = __fdividef(rhoPrev, omega);  //вычисление alpha

	x[i] += alpha * z[i];       //вычисление xk

	float rk = r[i] -= mask[i] * alpha * s[i];
	float qk = q[i] = DR[i] * rk;
	sdata[tid] = qk * rk;  //вычисление (rk, rk)
	blockReduce(sdata, rhoNext, tid);
}

__global__ void cg3P_d(double* x, double* r, double* z, double* s, double* q, double* DR, double* scalars, bool* mask) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tid = threadIdx.x;
	double omega = *scalars, * rhoNext = scalars + 1, rhoPrev = scalars[2];
	__shared__ double sdata[CGBS];
	double alpha = rhoPrev / omega;  //вычисление alpha

	x[i] += alpha * z[i];       //вычисление xk

	double rk = r[i] -= mask[i] * alpha * s[i];
	double qk = q[i] = DR[i] * rk;
	sdata[tid] = qk * rk;
	blockReduce(sdata, rhoNext, tid);
}


__global__ void cg3v2_d(double* x, double* r, double* z, double* s, double* scalars, bool* mask) {
	int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	int tid = threadIdx.x;
	double omega = *scalars, * rhoNext = scalars + 1, rhoPrev = scalars[2];
	__shared__ double sdata[CGBS / 2];
	double alpha = rhoPrev / omega;  //вычисление alpha

	double2 v = *reinterpret_cast<double2*>(z + i);

	x[i] += alpha * v.x;       //вычисление xk
	x[i + 1] += alpha * v.y;

	v = *reinterpret_cast<double2*>(s + i);

	double rk1 = r[i] -= mask[i] * alpha * v.x;
	double rk2 = r[i + 1] -= mask[i + 1] * alpha * v.y;
	sdata[tid] = rk1 * rk1 + rk2 * rk2;  //вычисление (rk, rk)
	blockReduce(sdata, rhoNext, tid);
}

__global__ void cg3v2_f(float* x, float* r, float* z, float* s, float* scalars, bool* mask) {
	int i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	int tid = threadIdx.x;
	float omega = *scalars, * rhoNext = scalars + 1, rhoPrev = scalars[2];
	__shared__ float sdata[CGBS / 2];
	float alpha = __fdividef(rhoPrev, omega);  //вычисление alpha

	float2 v = *reinterpret_cast<float2*>(z + i);

	x[i] += alpha * v.x;       //вычисление xk
	x[i + 1] += alpha * v.y;

	v = *reinterpret_cast<float2*>(s + i);

	float rk1 = r[i] -= mask[i] * alpha * v.x;
	float rk2 = r[i + 1] -= mask[i + 1] * alpha * v.y;
	sdata[tid] = rk1 * rk1 + rk2 * rk2;  //вычисление (rk, rk)
	blockReduce(sdata, rhoNext, tid);
}


__global__ void cg3P2_f(float* x, float* r, float* z, float* s, float* q, float* DR, float* scalars, bool* mask) {
	unsigned i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	unsigned tid = threadIdx.x;
	float omega = *scalars, * rhoNext = scalars + 1, rhoPrev = scalars[2];
	__shared__ float sdata[CGBS / 2];
	float alpha = rhoPrev / omega;  //вычисление alpha

	float2 v = *reinterpret_cast<float2*>(z + i);
	x[i] += alpha * v.x;       //вычисление xk
	x[i + 1] += alpha * v.y;

	v = *reinterpret_cast<float2*>(s + i);
	float rk1 = r[i] -= mask[i] * alpha * v.x;
	float rk2 = r[i + 1] -= mask[i + 1] * alpha * v.y;

	v = *reinterpret_cast<float2*>(DR + i);
	float qk1 = q[i] = v.x * rk1;
	float qk2 = q[i + 1] = v.y * rk2;

	sdata[tid] = qk1 * rk1 + qk2 * rk2;
	blockReduce(sdata, rhoNext, tid);
}

__global__ void cg3P2_d(double* x, double* r, double* z, double* s, double* q, double* DR, double* scalars, bool* mask) {
	unsigned i = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
	unsigned tid = threadIdx.x;
	double omega = *scalars, * rhoNext = scalars + 1, rhoPrev = scalars[2];
	__shared__ double sdata[CGBS / 2];
	double alpha = rhoPrev / omega;  //вычисление alpha

	double2 v = *reinterpret_cast<double2*>(z + i);
	x[i] += alpha * v.x;       //вычисление xk
	x[i + 1] += alpha * v.y;

	v = *reinterpret_cast<double2*>(s + i);
	float rk1 = r[i] -= mask[i] * alpha * v.x;
	float rk2 = r[i + 1] -= mask[i + 1] * alpha * v.y;

	v = *reinterpret_cast<double2*>(DR + i);
	double qk1 = q[i] = v.x * rk1;
	double qk2 = q[i + 1] = v.y * rk2;

	sdata[tid] = qk1 * rk1 + qk2 * rk2;
	blockReduce(sdata, rhoNext, tid);
}


__global__ void cg3Lines_f(float* x, float* r, float* z, float* s, float* scalars, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	float omega = *scalars, rhoPrev = scalars[2];
	float alpha = rhoPrev / omega;  //вычисление alpha
	x[i] += alpha * z[i];       //вычисление xk

	r[i] -= mask[i] * alpha * s[i];
}

__global__ void cg3Lines_d(double* x, double* r, double* z, double* s, double* scalars, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	double omega = *scalars, rhoPrev = scalars[2];
	double alpha = rhoPrev / omega;  //вычисление alpha
	x[i] += alpha * z[i];       //вычисление xk

	r[i] -= mask[i] * alpha * s[i];
}


__global__ void cg3LinesP_f(float* x, float* r, float* z, float* s, float* q, float* DR, float* scalars, bool* mask) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tid = threadIdx.x;
	float omega = *scalars, rhoPrev = scalars[2];
	float alpha = __fdividef(rhoPrev, omega);  //вычисление alpha

	x[i] += alpha * z[i];       //вычисление xk

	float rk = r[i] -= mask[i] * alpha * s[i];
	q[i] = DR[i] * rk;
}

__global__ void cg3LinesP_d(double* x, double* r, double* z, double* s, double* q, double* DR, double* scalars, bool* mask) {
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned tid = threadIdx.x;
	double omega = *scalars, rhoPrev = scalars[2];
	double alpha = rhoPrev / omega;  //вычисление alpha

	x[i] += alpha * z[i];       //вычисление xk

	double rk = r[i] -= mask[i] * alpha * s[i];
	q[i] = DR[i] * rk;
}



__global__ void loopCondition_f_(cudaGraphConditionalHandle handle, float* rho, float eps, size_t* iterations) {
	cudaGraphSetConditional(handle, *rho > eps);
	++(*iterations);
	//printf("%d ", *iterations);
}

__global__ void loopCondition_d_(cudaGraphConditionalHandle handle, double* rho, double eps, size_t* iterations) {
	cudaGraphSetConditional(handle, *rho > eps);
	++(*iterations);
	//printf("%d ", *iterations);
}


__global__ void loopCondition_f(cudaGraphConditionalHandle handle, float* rho, float* eps, unsigned* iterations, unsigned N) {
	//printf("d %d: eps = %e, rho = %e\n", *iterations, *eps, *rho);
	cudaGraphSetConditional(handle, *rho > *eps && *iterations < N);
	++(*iterations);
}

__global__ void loopCondition_d(cudaGraphConditionalHandle handle, double* rho, double* eps, unsigned* iterations, unsigned N) {
	//printf("d %d: eps = %e, rho = %e\n", *iterations, *eps, *rho);
	cudaGraphSetConditional(handle, *rho > *eps && *iterations < N);
	++(*iterations);
}