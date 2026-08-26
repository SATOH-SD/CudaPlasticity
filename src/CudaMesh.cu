#include "CudaMesh.h"

#include "Mesh.h"
#include "FiniteElement.h"
#include "DeviceCommon.cuh"


CudaMesh::CudaMesh() = default;

CudaMesh::~CudaMesh() = default;


CudaMesh::CudaMesh(const Mesh& mesh) {
	fromMesh(mesh);
}


void CudaMesh::fromMesh(const Mesh& mesh) {
	totalElems = mesh.totalElems;
	elemTypes = mesh.elemTypes;
	dim = mesh.dim;
	nodeCount = mesh.nodeCount;

	node.realloc(nodeCount * dim);
	hostToDevice(node, mesh.node, nodeCount * dim);

	elemInfo.realloc(elemTypes);
	for (unsigned i = 0; i < elemTypes; ++i)
		elemInfo[i] = CudaFiniteElement(mesh.elemInfo[i]);

	CudaFiniteElement& lastType = elemInfo[elemTypes - 1];
	unsigned elemMem = lastType.memIdx + lastType.elemCount * lastType.nodeCount;
	elem.realloc(elemMem);
	hostToDevice(elem, mesh.elem, elemMem);
}


void CudaMesh::addFloat() {
	for (unsigned i = 0; i < elemTypes; ++i)
		elemInfo[i].addFloat();
	const unsigned BS = 1024;
	unsigned nodeMem = nodeCount * dim;
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	doubleToFloat<<<(nodeCount + BS - 1) / BS, BS>>>(node_f, node, nodeMem);
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
}


void CudaMesh::addFloat(const Mesh& mesh) {
	for (unsigned i = 0; i < elemTypes; ++i)
		elemInfo[i].addFloat(mesh.elemInfo[i]);
	const unsigned BS = 1024;
	unsigned nodeMem = nodeCount * dim;
	cudaStream_t stream;
	cudaStreamCreate(&stream);
	doubleToFloat<<<(nodeCount + BS - 1) / BS, BS>>>(node_f, node, nodeMem);
	cudaStreamSynchronize(stream);
	cudaStreamDestroy(stream);
}


bool CudaMesh::empty() const {
	return !(elemTypes || nodeCount);
}