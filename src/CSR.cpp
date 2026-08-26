#include "CSR.h"

#include <iostream>

#include "NodeAdjStruct.h"
#include "FiniteElement.h"


#if _OPENMP >= 200805
typedef unsigned omp_for_t;
#else
typedef int omp_for_t;
#endif


CSR::CSR(const NodeAdjStruct& nodeAdjStruct, unsigned dim) {
	N = dim * nodeAdjStruct.N;
	rows.malloc(N + 1);
	unsigned dataSize = dim * dim * (nodeAdjStruct.adjIdx[nodeAdjStruct.N] + nodeAdjStruct.N);
	//std::cout << nodeAdjStruct.adjIdx[nodeAdjStruct.N] << "\n";
	//std::cout << "data size " << dataSize << "\n";

	data.malloc(dataSize);
	cols.malloc(dataSize);

	rows[0] = 0;
#pragma omp parallel for
	for (omp_for_t node = 0; node < nodeAdjStruct.N; ++node) {
		unsigned begin = nodeAdjStruct.adjIdx[node], size = nodeAdjStruct.adjIdx[node + 1] - begin;
		for (unsigned i = 0; i < dim; ++i)
			rows[dim * node + i + 1] = begin * dim * dim + (i + 1) * size * dim + (dim * node + i + 1) * dim;
	}

#pragma omp parallel for
	for (omp_for_t node = 0; node < nodeAdjStruct.N; ++node) {
		unsigned begin = nodeAdjStruct.adjIdx[node], size = nodeAdjStruct.adjIdx[node + 1] - begin;

		for (unsigned i = 0; i < dim; ++i) {
			for (unsigned j = 0; j < dim; ++j) //диагональные столбцы
				cols[rows[dim * node + i] + j] = dim * node + j;
			//остальные столбцы
			for (unsigned link = 0; link < size; ++link) {
				unsigned insBegin = rows[dim * node + i] + (link + 1) * dim;
				for (unsigned j = 0; j < dim; ++j)
					cols[insBegin + j] = nodeAdjStruct.adj[begin + link] * dim + j;
			}
		}
	}
}


CSR::CSR(const NodeAdjStruct& nodeAdjStruct, unsigned elemTypes, const FiniteElement* fe, const unsigned* elem) {

	hptr<unsigned> nodeMaxDim(nodeAdjStruct.N);
	nodeMaxDim.setZero(nodeAdjStruct.N);
	for (unsigned t = 0; t < elemTypes; ++t) {
		const FiniteElement& type = fe[t];
		for (unsigned i = type.memIdx; i < type.memIdx + type.nodeCount * type.elemCount; ++i) {
			unsigned node = elem[i];
			nodeMaxDim[node] = std::max(type.nodeDim, nodeMaxDim[node]);
		}
	}
	nodeLine.malloc(nodeAdjStruct.N);
	nodeLine[0] = 0;
	for (unsigned i = 1; i < nodeAdjStruct.N; ++i)
		nodeLine[i] = nodeLine[i - 1] + nodeMaxDim[i - 1];
	N = nodeLine[nodeAdjStruct.N - 1] + nodeMaxDim[nodeAdjStruct.N - 1];
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

	rows.malloc(N + 1);
	rows[0] = 0;
	unsigned line = 0;
	for (unsigned i = 0; i < nodeAdjStruct.N; ++i)
		for (unsigned d = 0; d < nodeMaxDim[i]; ++d, ++line)
			rows[line + 1] = rows[line] + nodeLineWidth[i];

	/*for (unsigned i = 0; i <= N; ++i)
		std::cout << rows[i] << " ";*/

	data.malloc(rows[N]);
	cols.malloc(rows[N]);
	for (unsigned i = 0; i < nodeAdjStruct.N; ++i) {
		unsigned dim = nodeMaxDim[i];
		unsigned linei = nodeLine[i];
		for (unsigned dj = 0; dj < dim; ++dj) {
			unsigned col = linei + dj;
			for (unsigned di = 0; di < dim; ++di)
				cols[rows[linei + di] + dj] = col;
		}
		unsigned linej = dim;
		for (unsigned j = nodeAdjStruct.adjIdx[i]; j < nodeAdjStruct.adjIdx[i + 1]; ++j) {
			unsigned node = nodeAdjStruct.adj[j];
			unsigned dim = nodeMaxDim[node];
			for (unsigned dj = 0; dj < dim; ++dj) {
				unsigned col = nodeLine[node] + dj;
				for (unsigned di = 0; di < dim; ++di) {
					cols[rows[linei + di] + linej] = col;
				}
				++linej;
			}
		}
	}

	CSR::elemTypes = elemTypes;
	elemInsIdx.malloc(elemTypes + 1);
	elemInsSize.malloc(elemTypes);
	elemInsIdx[0] = 0;
	for (unsigned t = 0; t < elemTypes; ++t) {
		const FiniteElement& type = fe[t];
		elemInsSize[t] = type.nodeCount * type.nodeCount;
		elemInsIdx[t + 1] = elemInsIdx[t] + elemInsSize[t] * type.elemCount;
	}
	elemIns.malloc(elemInsIdx[elemTypes]);
	for (unsigned t = 0; t < elemTypes; ++t) {
		const FiniteElement& type = fe[t];
		const unsigned* locElem = elem + type.memIdx;
		unsigned* ins = elemIns + elemInsIdx[t];
#pragma omp parallel for
		for (omp_for_t e = 0; e < type.elemCount; ++e) {
			const unsigned* curElem = locElem + e * type.nodeCount;
			unsigned* curIns = ins + elemInsSize[t] * e;
			for (unsigned i = 0; i < type.nodeCount; ++i)
				curIns[i * type.nodeCount + i] = 0;

			for (unsigned i = 1; i < type.nodeCount; ++i)
				for (unsigned j = 0; j < i; ++j) {
					unsigned ni = curElem[i], nj = curElem[j];
					unsigned gi = nodeLine[ni], gj = nodeLine[nj];
					unsigned* curCols = cols + rows[gi];
					for (unsigned jj = nodeMaxDim[ni]; jj < nodeLineWidth[ni]; ++jj)
						if (curCols[jj] == gj) {
							curIns[i * type.nodeCount + j] = jj;
							break;
						}
					curCols = cols + rows[gj];
					for (unsigned ii = nodeMaxDim[nj]; ii < nodeLineWidth[nj]; ++ii)
						if (curCols[ii] == gi) {
							curIns[j * type.nodeCount + i] = ii;
							break;
						}
				}
		}
	}

	// DEBUG
	/*for (unsigned t = 0; t < elemTypes; ++t) {
		const FiniteElement& type = fe[t];
		unsigned* ins = elemIns + elemInsIdx[t];
		for (unsigned e = 0; e < type.elemCount; ++e) {
			unsigned* curIns = ins + elemInsSize[t] * e;
			printf("\n");
			for (unsigned i = 0; i < type.nodeCount; ++i) {
				for (unsigned j = 0; j < type.nodeCount; ++j)
					printf("%5d ", curIns[i * type.nodeCount + j]);
				printf("\n");
			}
			printf("\n");
		}
	}*/
	//for (unsigned t = 0; t < elemTypes; ++t) {
	//	const FiniteElement& type = fe[t];
	//	const unsigned* locElem = elem + type.memIdx;
	//	unsigned* ins = elemIns + elemInsIdx[t];
	//	for (unsigned e = 0; e < type.elemCount; ++e) {
	//		const unsigned* curElem = locElem + e * type.nodeCount;
	//		std::cout << e << ":  ";
	//		for (unsigned i = 0; i < type.nodeCount; ++i)
	//			std::cout << curElem[i] << " ";
	//		unsigned* curIns = ins + elemInsSize[t] * e;
	//		printf("\n");
	//		for (unsigned i = 0; i < type.nodeCount; ++i) {
	//			for (unsigned j = 0; j < type.nodeCount; ++j) {
	//				unsigned ni = curElem[i], nj = curElem[j];
	//				unsigned gi = nodeLine[ni], gj = nodeLine[nj];
	//				printf("%5d ", cols[rows[gi] + curIns[i * type.nodeCount + j]] / 2);
	//				//printf("%5d ", curIns[i * type.nodeCount + j]);
	//			}
	//			printf("\n");
	//		}
	//		printf("\n");
	//	}
	//}
	// DEBUG
}


void CSR::insertKe(unsigned elemType, unsigned e, double* Ke, unsigned* elem, unsigned nodeCount, unsigned nodeDim) {
	unsigned* ins = elemIns + elemInsIdx[elemType] + e * elemInsSize[elemType];
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


void CSR::insertKeSym(unsigned elemType, unsigned e, double* Ke, unsigned* elem, unsigned nodeCount, unsigned nodeDim) {
	unsigned* ins = elemIns + elemInsIdx[elemType] + e * elemInsSize[elemType];
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


double& CSR::getItem(unsigned i, unsigned j) {
	for (unsigned idx = rows[i]; idx < rows[i + 1]; ++idx)
		if (cols[idx] == j) {
			//std::cout << i << " " << j << "   " << idx - rows[i] << "\n";
			return data[idx];
		}
	return data[0];
}


void CSR::printStruct() const {
	for (unsigned i = 0; i < N; ++i) {
		std::cout << i << ":  ";
		for (unsigned j = rows[i]; j < rows[i + 1]; ++j)
			std::cout << cols[j] << " ";
		std::cout << " (" << rows[i + 1] - rows[i] << ")\n";
	}
}