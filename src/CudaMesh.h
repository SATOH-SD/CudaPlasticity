#pragma once

#include "ptrs.h"


class Mesh;

class CudaFiniteElement;


class CudaMesh {

public:

	unsigned elemTypes = 0;

	unsigned totalElems = 0;

	hptr<CudaFiniteElement> elemInfo;

	dptr<unsigned> elem;

	unsigned dim = 2;
	unsigned nodeCount = 0;

	dptr<double> node;
	dptr<float> node_f;

	dptr<unsigned> colorMap;

	bool useFloat = false;

	CudaMesh();

	CudaMesh(CudaMesh&& mesh);

	CudaMesh(const Mesh& mesh);

	~CudaMesh();

	void fromMesh(const Mesh& mesh);

	//void addFloat();

	void addFloat(const Mesh& mesh);

	bool empty() const;

};