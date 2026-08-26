#pragma once

#include "ptrs.h"


class NodeAdjStruct;


class CudaCSR {

public:

	unsigned N = 0;
	unsigned dataSize = 0;

	dptr<unsigned> rows;
	dptr<unsigned> cols;

	dptr<double> data;
	dptr<float> data_f;

	CudaCSR(const NodeAdjStruct& nodeAdjStruct, unsigned dim);

	void mallocDouble();

	void mallocFloat();

};