#pragma once

#include "ptrs.h"


class NodeAdjStruct;

class FiniteElement;


// Разреженная матрица формата Compressed Sparse Row (CSR)
class CSR {

public:

	unsigned N = 0;

	hptr<double> data;  // возможно придётся сделать внешнее хранение данных
	hptr<unsigned> rows;
	hptr<unsigned> cols;

	unsigned elemTypes = 0;
	hptr<unsigned> elemIns;
	hptr<unsigned> elemInsIdx;
	hptr<unsigned> elemInsSize;
	hptr<unsigned> nodeLine;

	CSR(const NodeAdjStruct& nodeAdjStruct, unsigned dim);

	CSR(const NodeAdjStruct& nodeAdjStruct, unsigned elemTypes, const FiniteElement* fe, const unsigned* elem);

	void insertKe(unsigned elemType, unsigned e, double* Ke, unsigned* elem, unsigned nodeCount, unsigned nodeDim);

	void insertKeSym(unsigned elemType, unsigned e, double* Ke, unsigned* elem, unsigned nodeCount, unsigned nodeDim);

	unsigned dataSize() {
		return rows[N];
	}

	double& getItem(unsigned i, unsigned j);

	void printStruct() const;

};