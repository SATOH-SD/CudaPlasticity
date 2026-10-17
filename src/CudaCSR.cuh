#pragma once

#include "ptrs.h"


class NodeAdjStruct;

class FiniteElement;


class CudaCSR {

public:

	unsigned N = 0;
	unsigned dataSize = 0;

	dptr<unsigned> rows;
	dptr<unsigned> cols;

	dptr<double> data;
	dptr<float> data_f;

	unsigned elemTypes = 0;
	dptr<unsigned> elemIns;
	dptr<unsigned> elemInsIdx;
	dptr<unsigned> elemInsSize;
	dptr<unsigned> nodeLine;

	CudaCSR(const NodeAdjStruct& nodeAdjStruct, unsigned dim);

	// TODO: constructor for FEs with const dim

	CudaCSR(const NodeAdjStruct& nodeAdjStruct, unsigned elemTypes, const FiniteElement* fe, const unsigned* elem, unsigned blockSize = 512);

	void mallocDouble();

	void mallocFloat();

};

__device__
void cudaCSR_insertKe(double* data, const unsigned* rows,
	const unsigned* elemIns, const unsigned* elemInsIdx,
	const unsigned* elemInsSize, const unsigned* nodeLine,
	unsigned elemType, unsigned e,
	const double* Ke, const unsigned* elem, unsigned nodeCount, unsigned nodeDim);

__device__
void cudaCSR_insertKe(float* data, const unsigned* rows,
	const unsigned* elemIns, const unsigned* elemInsIdx,
	const unsigned* elemInsSize, const unsigned* nodeLine,
	unsigned elemType, unsigned e,
	const float* Ke, const unsigned* elem, unsigned nodeCount, unsigned nodeDim);

__device__
void cudaCSR_insertKeSym(double* data, const unsigned* rows,
	const unsigned* elemIns, const unsigned* elemInsIdx,
	const unsigned* elemInsSize, const unsigned* nodeLine,
	unsigned elemType, unsigned e,
	const double* Ke, const unsigned* elem, unsigned nodeCount, unsigned nodeDim);

__device__
void cudaCSR_insertKeSym(float* data, const unsigned* rows,
	const unsigned* elemIns, const unsigned* elemInsIdx,
	const unsigned* elemInsSize, const unsigned* nodeLine,
	unsigned elemType, unsigned e,
	const float* Ke, const unsigned* elem, unsigned nodeCount, unsigned nodeDim);