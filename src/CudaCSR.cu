#include "CudaCSR.cuh"

#include "NodeAdjStruct.h"
#include "FiniteElement.h"

#include <stdio.h> // DEBUG


#if _OPENMP >= 200805
typedef unsigned omp_for_t;
#else
typedef int omp_for_t;
#endif


CudaCSR::CudaCSR(const NodeAdjStruct& nodeAdjStruct, unsigned dim) {
	N = dim * nodeAdjStruct.N;
	rows.malloc(N + 1);
	hptr<unsigned> host_rows(N + 1);
	unsigned dataSize = dim * dim * nodeAdjStruct.adjIdx[nodeAdjStruct.N];
	cols.malloc(dataSize);
	hptr<unsigned> host_cols(dataSize);

	host_rows[0] = 0;
#pragma omp parallel for
	for (omp_for_t node = 0; node < nodeAdjStruct.N; ++node) {
		unsigned begin = nodeAdjStruct.adjIdx[node], size = nodeAdjStruct.adjIdx[node + 1] - begin;
		for (unsigned i = 0; i < dim; ++i)
			host_rows[dim * node + i + 1] = begin * dim * dim + (i + 1) * size * dim + (dim * node + i + 1) * dim;
	}

#pragma omp parallel for
	for (omp_for_t node = 0; node < nodeAdjStruct.N; ++node) {
		unsigned begin = nodeAdjStruct.adjIdx[node], size = nodeAdjStruct.adjIdx[node + 1] - begin;

		for (unsigned i = 0; i < dim; ++i) {
			for (unsigned j = 0; j < dim; ++j) //диагональные столбцы
				host_cols[host_rows[dim * node + i] + j] = dim * node + j;
			//остальные столбцы
			for (unsigned link = 0; link < size; ++link) {
				unsigned insBegin = host_rows[dim * node + i] + (link + 1) * dim;
				for (unsigned j = 0; j < dim; ++j)
					host_cols[insBegin + j] = nodeAdjStruct.adj[begin + link] * dim + j;
			}
		}
	}
	hostToDevice(rows, host_rows, N + 1);
	hostToDevice(cols, host_cols, dataSize);
}


CudaCSR::CudaCSR(const NodeAdjStruct& nodeAdjStruct, unsigned elemTypes, const FiniteElement* fe, const unsigned* elem, unsigned blockSize) {
	hptr<unsigned> nodeMaxDim(nodeAdjStruct.N);
	nodeMaxDim.setZero(nodeAdjStruct.N);
	for (unsigned t = 0; t < elemTypes; ++t) {
		const FiniteElement& type = fe[t];
		for (unsigned i = type.memIdx; i < type.memIdx + type.nodeCount * type.elemCount; ++i) {
			unsigned node = elem[i];
			nodeMaxDim[node] = std::max(type.nodeDim, nodeMaxDim[node]);
		}
	}
	hptr<unsigned> host_nodeLine(nodeAdjStruct.N);
	host_nodeLine[0] = 0;
	for (unsigned i = 1; i < nodeAdjStruct.N; ++i)
		host_nodeLine[i] = host_nodeLine[i - 1] + nodeMaxDim[i - 1];
	N = host_nodeLine[nodeAdjStruct.N - 1] + nodeMaxDim[nodeAdjStruct.N - 1];
	//std::cout << "N " << N << "\n";

	hptr<unsigned> nodeLineWidth(nodeAdjStruct.N);
#pragma omp parallel for
	for (omp_for_t i = 0; i < nodeAdjStruct.N; ++i) {
		nodeLineWidth[i] = nodeMaxDim[i];
		for (unsigned j = nodeAdjStruct.adjIdx[i]; j < nodeAdjStruct.adjIdx[i + 1]; ++j)
			nodeLineWidth[i] += nodeMaxDim[nodeAdjStruct.adj[j]];
	}
	/*for (unsigned i = 0; i < nodeAdjStruct.N; ++i)
		std::cout << nodeLineWidth[i] << " ";*/

	hptr<unsigned> host_rows(N + 1);
	host_rows[0] = 0;
	unsigned line = 0;
	for (unsigned i = 0; i < nodeAdjStruct.N; ++i)
		for (unsigned d = 0; d < nodeMaxDim[i]; ++d, ++line)
			host_rows[line + 1] = host_rows[line] + nodeLineWidth[i];
	dataSize = host_rows[N];

	/*for (unsigned i = 0; i <= N; ++i)
		std::cout << host_rows[i] << " ";*/

	hptr<unsigned> host_cols(dataSize);
	for (unsigned i = 0; i < nodeAdjStruct.N; ++i) {
		unsigned dim = nodeMaxDim[i];
		unsigned linei = host_nodeLine[i];
		for (unsigned dj = 0; dj < dim; ++dj) {
			unsigned col = linei + dj;
			for (unsigned di = 0; di < dim; ++di)
				host_cols[host_rows[linei + di] + dj] = col;
		}
		unsigned linej = dim;
		for (unsigned j = nodeAdjStruct.adjIdx[i]; j < nodeAdjStruct.adjIdx[i + 1]; ++j) {
			unsigned node = nodeAdjStruct.adj[j];
			unsigned dim = nodeMaxDim[node];
			for (unsigned dj = 0; dj < dim; ++dj) {
				unsigned col = host_nodeLine[node] + dj;
				for (unsigned di = 0; di < dim; ++di) {
					host_cols[host_rows[linei + di] + linej] = col;
				}
				++linej;
			}
		}
	}

	CudaCSR::elemTypes = elemTypes;
	hptr<unsigned> host_elemInsIdx(elemTypes + 1);
	hptr<unsigned> host_elemInsSize(elemTypes);
	host_elemInsIdx[0] = 0;
	for (unsigned t = 0; t < elemTypes; ++t) {
		const FiniteElement& type = fe[t];
		host_elemInsSize[t] = type.nodeCount * type.nodeCount;
		host_elemInsIdx[t + 1] = host_elemInsIdx[t] + host_elemInsSize[t] * type.elemCount;
	}
	hptr<unsigned> host_elemIns(host_elemInsIdx[elemTypes]);
	for (unsigned t = 0; t < elemTypes; ++t) {
		const FiniteElement& type = fe[t];
		const unsigned* locElem = elem + type.memIdx;
		unsigned* ins = host_elemIns + host_elemInsIdx[t];
#pragma omp parallel for
		for (omp_for_t e = 0; e < type.elemCount; ++e) {
			const unsigned* curElem = locElem + e * type.nodeCount;
			unsigned* curIns = ins + host_elemInsSize[t] * e;
			for (unsigned i = 0; i < type.nodeCount; ++i)
				curIns[i * type.nodeCount + i] = 0;

			for (unsigned i = 1; i < type.nodeCount; ++i)
				for (unsigned j = 0; j < i; ++j) {
					unsigned ni = curElem[i], nj = curElem[j];
					unsigned gi = host_nodeLine[ni], gj = host_nodeLine[nj];
					unsigned* curCols = host_cols + host_rows[gi];
					for (unsigned jj = nodeMaxDim[ni]; jj < nodeLineWidth[ni]; ++jj)
						if (curCols[jj] == gj) {
							curIns[i * type.nodeCount + j] = jj;
							break;
						}
					curCols = host_cols + host_rows[gj];
					for (unsigned ii = nodeMaxDim[nj]; ii < nodeLineWidth[nj]; ++ii)
						if (curCols[ii] == gi) {
							curIns[j * type.nodeCount + i] = ii;
							break;
						}
				}
		}
	}
	nodeLine.malloc(nodeAdjStruct.N);  // TODO: async memcpy and page-pinned host memory
	unsigned rowsMemLen = (N + blockSize - 1) / blockSize * blockSize + 1;
	rows.realloc(rowsMemLen);
	cols.malloc(host_rows[N]);
	elemInsIdx.malloc(elemTypes + 1);
	elemInsSize.malloc(elemTypes);
	elemIns.malloc(host_elemInsIdx[elemTypes]);
	hostToDevice(nodeLine, host_nodeLine, nodeAdjStruct.N);
	cudaMemset(rows + N + 1, 0, std::max(0LL, (long long)rowsMemLen - (long long)(N + 1)));
	hostToDevice(rows, host_rows, N + 1);
	hostToDevice(cols, host_cols, host_rows[N]);
	hostToDevice(elemInsIdx, host_elemInsIdx, elemTypes + 1);
	hostToDevice(elemInsSize, host_elemInsSize, elemTypes);
	hostToDevice(elemIns, host_elemIns, host_elemInsIdx[elemTypes]);
}


void CudaCSR::mallocDouble() {
	data.realloc(dataSize);
}


void CudaCSR::mallocFloat() {
	data_f.realloc(dataSize);
}


template<typename fp>
__device__ static inline
void insertKe(fp* data, const unsigned* rows,
	const unsigned* elemIns, const unsigned* elemInsIdx,
	const unsigned* elemInsSize, const unsigned* nodeLine,
	unsigned elemType, unsigned e,
	const fp* Ke, const unsigned* elem, unsigned nodeCount, unsigned nodeDim) {

	const unsigned* ins = elemIns + elemInsIdx[elemType] + e * elemInsSize[elemType];
	unsigned KeSize = nodeCount * nodeDim;
	for (unsigned i = 0; i < nodeCount; ++i) {
		unsigned di = i * nodeDim;
		unsigned linei = nodeLine[elem[i]];
		for (unsigned j = 0; j < nodeCount; ++j) {
			unsigned colj = ins[i * nodeCount + j];
			unsigned dj = j * nodeDim;
			for (unsigned ii = 0; ii < nodeDim; ++ii) {
				for (unsigned jj = 0; jj < nodeDim; ++jj) {
					data[rows[linei + ii] + colj + jj] += Ke[(di + ii) * KeSize + dj + jj];
				}
			}
		}
	}
}

template<typename fp>
__device__ static inline
void insertKeSym(fp* data, const unsigned* rows,
	const unsigned* elemIns, const unsigned* elemInsIdx,
	const unsigned* elemInsSize, const unsigned* nodeLine,
	unsigned elemType, unsigned e,
	const fp* Ke, const unsigned* elem, unsigned nodeCount, unsigned nodeDim) {

	const unsigned* ins = elemIns + elemInsIdx[elemType] + e * elemInsSize[elemType];
	for (unsigned i = 0; i < nodeCount; ++i) {
		unsigned di = i * nodeDim;
		unsigned linei = nodeLine[elem[i]];
		unsigned coli = ins[i * nodeCount + i];
		for (unsigned ii = 0; ii < nodeDim; ++ii) {
			unsigned ki = di + ii;
			data[rows[linei + ii] + coli + ii] += Ke[ki * (ki + 1) / 2 + ki];
			for (unsigned jj = 0; jj < ii; ++jj) {
				double val = Ke[ki * (ki + 1) / 2 + di + jj];
				data[rows[linei + ii] + coli + jj] += val;
				data[rows[linei + jj] + coli + ii] += val;
			}
		}
		for (unsigned j = 0; j < i; ++j) {
			unsigned dj = j * nodeDim;
			unsigned linej = nodeLine[elem[j]];
			unsigned colj = ins[i * nodeCount + j];
			coli = ins[j * nodeCount + i];
			for (unsigned ii = 0; ii < nodeDim; ++ii) {
				unsigned ki = di + ii;
				for (unsigned jj = 0; jj < nodeDim; ++jj) {
					double val = Ke[ki * (ki + 1) / 2 + dj + jj];
					data[rows[linei + ii] + colj + jj] += val;
					data[rows[linej + jj] + coli + ii] += val;
				}
			}
		}
	}
}


__device__
void cudaCSR_insertKe(double* data, const unsigned* rows,
	const unsigned* elemIns, const unsigned* elemInsIdx,
	const unsigned* elemInsSize, const unsigned* nodeLine,
	unsigned elemType, unsigned e,
	const double* Ke, const unsigned* elem, unsigned nodeCount, unsigned nodeDim) {
	insertKe(data, rows, elemIns, elemInsIdx, elemInsSize, nodeLine, \
		elemType, e, Ke, elem, nodeCount, nodeDim);
}

__device__
void cudaCSR_insertKe(float* data, const unsigned* rows,
	const unsigned* elemIns, const unsigned* elemInsIdx,
	const unsigned* elemInsSize, const unsigned* nodeLine,
	unsigned elemType, unsigned e,
	const float* Ke, const unsigned* elem, unsigned nodeCount, unsigned nodeDim) {
	insertKe(data, rows, elemIns, elemInsIdx, elemInsSize, nodeLine, \
		elemType, e, Ke, elem, nodeCount, nodeDim);
}

__device__
void cudaCSR_insertKeSym(double* data, const unsigned* rows,
	const unsigned* elemIns, const unsigned* elemInsIdx,
	const unsigned* elemInsSize, const unsigned* nodeLine,
	unsigned elemType, unsigned e,
	const double* Ke, const unsigned* elem, unsigned nodeCount, unsigned nodeDim) {
	insertKeSym(data, rows, elemIns, elemInsIdx, elemInsSize, nodeLine, \
		elemType, e, Ke, elem, nodeCount, nodeDim);
}

__device__
void cudaCSR_insertKeSym(float* data, const unsigned* rows,
	const unsigned* elemIns, const unsigned* elemInsIdx,
	const unsigned* elemInsSize, const unsigned* nodeLine,
	unsigned elemType, unsigned e,
	const float* Ke, const unsigned* elem, unsigned nodeCount, unsigned nodeDim) {
	insertKeSym(data, rows, elemIns, elemInsIdx, elemInsSize, nodeLine, \
		elemType, e, Ke, elem, nodeCount, nodeDim);
}