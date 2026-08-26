#include "CudaCSR.h"

#include "NodeAdjStruct.h"


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


void CudaCSR::mallocDouble() {
	data.realloc(dataSize);
}


void CudaCSR::mallocFloat() {
	data_f.realloc(dataSize);
}