#pragma once

#include <functional>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Mesh.cuh"
#include "StripSLAE.h"
#include "SparseSLAE.h"
#include "CudaSLAE.cu"

//Метод Гаусса для ленточной СЛАУ
std::vector<double> Gauss(StripSLAE& slae);

//Метод сопряжённых градиентов
void conjugateGradient(StripSLAE& slae, std::vector<double>& solution, size_t& iterNum, double eps);

void conjugateGradientCuda(StripSLAE& slae, std::vector<double>& solution, size_t& iterNum, double eps);

void conjugateGradient(SparseSLAE& slae, double* solution, bool* mask, size_t& iterNum, double eps);

//template<typename fp>
void conjugateGradient(CudaSLAE<double>& slae, double* solution, size_t& iterNum, double eps);

void conjugateGradient(CudaSLAE<float>& slae, float* solution, size_t& iterNum, float eps);

void conjugateGradient(CudaSLAE<double>& slae, double* solution, bool* mask, size_t& iterNum, double eps);