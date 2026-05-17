#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void isCloseToZero_f(cudaGraphConditionalHandle handle, float* num);
__global__ void isCloseToZero_d(cudaGraphConditionalHandle handle, double* num);

__global__ void norm2_f(float* x, float* sum);
__global__ void norm2_d(double* x, double* sum);

__global__ void norm2v2_f(float* x, float* sum);
__global__ void norm2v2_d(double* x, double* sum);

__global__ void norm2v3_f(float* x, float* sum);
__global__ void norm2v3_d(double* x, double* sum);

__global__ void norm2P_f(float* rhs, float* DR, float* sum);
__global__ void norm2P_d(double* rhs, double* DR, double* sum);

__global__ void dotProduct_f(float* a, float* b, float* sum);
__global__ void dotProduct_d(double* a, double* b, double* sum);

__global__ void dotProductV2_f(float* a, float* b, float* sum);
__global__ void dotProductV2_d(double* a, double* b, double* sum);

__global__ void dotProductV3_f(float* a, float* b, float* sum);
__global__ void dotProductV3_d(double* a, double* b, double* sum);

__global__ void precond_f(unsigned N, float* data, unsigned* rows, float* DR);
__global__ void precond_d(unsigned N, double* data, unsigned* rows, double* DR);

__global__ void precondAndNorm_f(unsigned N, float* data, unsigned* rows, float* DR, float* rhs, float* normB);
__global__ void precondAndNorm_d(unsigned N, double* data, unsigned* rows, double* DR, double* rhs, double* normB);

__global__ void updateEps_f(float* cond);
__global__ void updateEps_d(double* cond);

__global__ void cgInit_f(float* data, int* rows, int* cols, float* rhs, float* x, float* r, float* z, float* rhoNext, bool* mask);
__global__ void cgInit_d(double* data, int* rows, int* cols, double* rhs, double* x, double* r, double* z, double* rhoNext, bool* mask);

// LEGACY
__global__ void cgInitV2_f(float* data, int* rows, int* cols, float* rhs, float* x, float* r, float* z, float* rhoNext, bool* mask);
__global__ void cgInitV2_d(double* data, int* rows, int* cols, double* rhs, double* x, double* r, double* z, double* rhoNext, bool* mask);

__global__ void cgInitP_f(float* data, int* rows, int* cols, float* rhs, float* x, float* r, float* z, float* q, float* DR, float* rhoNext, bool* mask);
__global__ void cgInitP_d(double* data, int* rows, int* cols, double* rhs, double* x, double* r, double* z, double* q, double* DR, double* rhoNext, bool* mask);


__global__ void cgInitLines_f(float* data, int* rows, int* cols, float* rhs, float* x, float* r, float* z, bool* mask);
__global__ void cgInitLines_d(double* data, int* rows, int* cols, double* rhs, double* x, double* r, double* z, bool* mask);

__global__ void cgInitLinesP_f(float* data, int* rows, int* cols, float* rhs, float* x, float* r, float* z, float* q, float* DR, bool* mask);
__global__ void cgInitLinesP_d(double* data, int* rows, int* cols, double* rhs, double* x, double* r, double* z, double* q, double* DR, bool* mask);

__global__ void linesV2_f(unsigned lineCount, float* r, float* lines, unsigned* lineRows);
__global__ void linesV2_d(unsigned lineCount, double* r, double* lines, unsigned* lineRows);

__global__ void linesV3_f();
__global__ void linesV3_d();

__global__ void linesP2_f(unsigned lineCount, float* r, float* q, float* lines, unsigned* lineRows);
__global__ void linesP2_d(unsigned lineCount, double* r, double* q, double* lines, unsigned* lineRows);

__global__ void linesP3_f();
__global__ void linesP3_d();


__global__ void cg1_f(float* r, float* z, float* rhoPrev, float* rhoNext);
__global__ void cg1_d(double* r, double* z, double* rhoPrev, double* rhoNext);

__global__ void cg1v2_d(double* r, double* z, double* rhoPrev, double* rhoNext);
__global__ void cg1v2_f(float* r, float* z, float* rhoPrev, float* rhoNext);

__global__ void cg1v3_d(double* r, double* z, double* rhoPrev, double* rhoNext);
__global__ void cg1v3_f(float* r, float* z, float* rhoPrev, float* rhoNext);


__global__ void cg2_f(float* data, int* rows, int* cols, float* z, float* s, float* omega);
__global__ void cg2_d(double* data, int* rows, int* cols, double* z, double* s, double* omega);

__global__ void cg2_dv(double* data, int* rows, int* cols, double* z, double* s, double* omega);
__global__ void cg2_fv(float* data, int* rows, int* cols, float* z, float* s, float* omega);

__global__ void cg2v2_f(float* data, int* rows, int* cols, float* z, float* s, float* omega);
__global__ void cg2v2_d(double* data, int* rows, int* cols, double* z, double* s, double* omega);

__global__ void cg2v3_f(float* data, int* rows, int* cols, float* z, float* s, float* omega);
__global__ void cg2v3_d(double* data, int* rows, int* cols, double* z, double* s, double* omega);


__global__ void cg3_f(float* x, float* r, float* z, float* s, float* scalars, bool* mask);
__global__ void cg3_d(double* x, double* r, double* z, double* s, double* scalars, bool* mask);

__global__ void cg3P_f(float* x, float* r, float* z, float* s, float* q, float* DR, float* scalars, bool* mask);
__global__ void cg3P_d(double* x, double* r, double* z, double* s, double* q, double* DR, double* scalars, bool* mask);

__global__ void cg3v2_d(double* x, double* r, double* z, double* s, double* scalars, bool* mask);
__global__ void cg3v2_f(float* x, float* r, float* z, float* s, float* scalars, bool* mask);

__global__ void cg3P2_f(float* x, float* r, float* z, float* s, float* q, float* DR, float* scalars, bool* mask);
__global__ void cg3P2_d(double* x, double* r, double* z, double* s, double* q, double* DR, double* scalars, bool* mask);

__global__ void cg3v3_d(double* x, double* r, double* z, double* s, double* scalars, bool* mask);
__global__ void cg3v3_f(float* x, float* r, float* z, float* s, float* scalars, bool* mask);

__global__ void cg3P3_f();
__global__ void cg3P3_d();


__global__ void cg3Lines_f(float* x, float* r, float* z, float* s, float* scalars, bool* mask);
__global__ void cg3Lines_d(double* x, double* r, double* z, double* s, double* scalars, bool* mask);

__global__ void cg3LinesP_f(float* x, float* r, float* z, float* s, float* q, float* DR, float* scalars, bool* mask);
__global__ void cg3LinesP_d(double* x, double* r, double* z, double* s, double* q, double* DR, double* scalars, bool* mask);

__global__ void cg3Lines2_f(float* x, float* r, float* z, float* s, float* scalars, bool* mask);
__global__ void cg3Lines2_d(double* x, double* r, double* z, double* s, double* scalars, bool* mask);

__global__ void cg3LinesP2_f(float* x, float* r, float* z, float* s, float* q, float* DR, float* scalars, bool* mask);
__global__ void cg3LinesP2_d(double* x, double* r, double* z, double* s, double* q, double* DR, double* scalars, bool* mask);

__global__ void cg3Lines3_f();
__global__ void cg3Lines3_d();

__global__ void cg3LinesP3_f();
__global__ void cg3LinesP3_d();


// LEGACY
__global__ void loopCondition_f_(cudaGraphConditionalHandle handle, float* rho, float eps, size_t* iterations);
__global__ void loopCondition_d_(cudaGraphConditionalHandle handle, double* rho, double eps, size_t* iterations);

__global__ void loopCondition_f(cudaGraphConditionalHandle handle, float* rho, float* eps, unsigned* iterations, unsigned N);
__global__ void loopCondition_d(cudaGraphConditionalHandle handle, double* rho, double* eps, unsigned* iterations, unsigned N);