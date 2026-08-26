#include "FiniteElement.h"


CudaFiniteElement::CudaFiniteElement() = default;


CudaFiniteElement::CudaFiniteElement(const FiniteElement& fe) {
	blockId = fe.blockId;
	nodeCount = fe.nodeCount;
	memIdx = fe.memIdx;
	poolIdx = fe.poolIdx;
	elemCount = fe.elemCount;
	elemType = fe.elemType;
	geomType = fe.geomType;
	elemDim = fe.elemDim;
	intPointsCount = fe.intPointsCount;
	intCoefs.malloc(intPointsCount);
	hostToDevice(intCoefs, fe.intCoefs, intPointsCount);
	unsigned intMem = intPointsCount * elemDim;
	intPoints.malloc(intMem);
	hostToDevice(intPoints, fe.intPoints, intMem);
}


void CudaFiniteElement::addFloat() {
	hptr<double> coefs_d(intPointsCount);
	hptr<float> coefs_f(intPointsCount);
	deviceToHost(coefs_d, intCoefs, intPointsCount);
	for (unsigned i = 0; i < intPointsCount; ++i)
		coefs_f[i] = float(coefs_d[i]);
	intCoefs_f.realloc(intPointsCount);
	hostToDevice(intCoefs_f, coefs_f, intPointsCount);

	unsigned intMem = intPointsCount * elemDim;
	hptr<double> points_d(intMem);
	hptr<float> points_f(intMem);
	deviceToHost(points_d, intPoints, intMem);
	for (unsigned i = 0; i < intMem; ++i)
		points_f[i] = float(points_d[i]);
	intPoints_f.realloc(intMem);
	hostToDevice(intPoints_f, points_f, intMem);
}


void CudaFiniteElement::addFloat(const FiniteElement& fe) {
	hptr<float> coefs_f(intPointsCount);
	for (unsigned i = 0; i < intPointsCount; ++i)
		coefs_f[i] = float(fe.intCoefs[i]);
	intCoefs_f.realloc(intPointsCount);
	hostToDevice(intCoefs_f, coefs_f, intPointsCount);

	unsigned intMem = intPointsCount * elemDim;
	hptr<float> points_f(intMem);
	for (unsigned i = 0; i < intMem; ++i)
		points_f[i] = float(fe.intPoints[i]);
	intPoints_f.realloc(intMem);
	hostToDevice(intPoints_f, points_f, intMem);
}