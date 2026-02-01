#pragma once

#include <iostream>

#include "cuda_runtime.h"


struct Tensor2D {

	double x, y, xy;

};

struct Tensor2Df {

	float x, y, xy;

};


struct TensorAxiSym {

	double r, z, f, rz;

};

struct TensorAxiSymf {

	float r, z, f, rz;

};

struct Tensor3D {

	double x, y, z, xy, xz, yz;

};

struct Tensor3Df {

	float x, y, z, xy, xz, yz;

};