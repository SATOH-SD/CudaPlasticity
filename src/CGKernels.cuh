#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"

__global__ void cgInitG_f(float* data, int* rows, int* cols, float* rp, float* x, float* r, float* z, float* rhoNext, bool* mask);
__global__ void cgInitG_d(double* data, int* rows, int* cols, double* rp, double* x, double* r, double* z, double* rhoNext, bool* mask);

__global__ void cg1g_f(float* r, float* z, float* rhoPrev, float* rhoNext);
__global__ void cg1g_d(double* r, double* z, double* rhoPrev, double* rhoNext);

__global__ void cg2g_f(float* data, int* rows, int* cols, float* z, float* s, float* omega);
__global__ void cg2g_d(double* data, int* rows, int* cols, double* z, double* s, double* omega);

__global__ void cg2g_dv(double* data, int* rows, int* cols, double* z, double* s, double* omega);
__global__ void cg2g_fv(float* data, int* rows, int* cols, float* z, float* s, float* omega);

__global__ void cg3g_f(float* x, float* r, float* z, float* s, float* scalars, bool* mask);
__global__ void cg3g_d(double* x, double* r, double* z, double* s, double* scalars, bool* mask);

__global__ void cgInitGV_f(float* data, int* rows, int* cols, float* rp, float* x, float* r, float* z, float* rhoNext, bool* mask);
__global__ void cgInitGV_d(double* data, int* rows, int* cols, double* rp, double* x, double* r, double* z, double* rhoNext, bool* mask);

__global__ void cg1gv_d(double* r, double* z, double* rhoPrev, double* rhoNext);
__global__ void cg1gv_f(float* r, float* z, float* rhoPrev, float* rhoNext);

__global__ void cg2gv_f(float* data, int* rows, int* cols, float* z, float* s, float* omega);
__global__ void cg2gv_d(double* data, int* rows, int* cols, double* z, double* s, double* omega);

__global__ void cg3gv_d(double* x, double* r, double* z, double* s, double* scalars, bool* mask);
__global__ void cg3gv_f(float* x, float* r, float* z, float* s, float* scalars, bool* mask);

__global__ void loopCondition_f(cudaGraphConditionalHandle handle, float* rho, float eps, size_t* iterations);
__global__ void loopCondition_d(cudaGraphConditionalHandle handle, double* rho, double eps, size_t* iterations);