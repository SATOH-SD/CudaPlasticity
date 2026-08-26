#include "NodeAdjStruct.h"

#include <iostream>
#include <algorithm>
#include <omp.h>

#include "FiniteElement.h"


#if _OPENMP >= 200805
typedef unsigned omp_for_t;
#else
typedef int omp_for_t;
#endif


NodeAdjStruct::NodeAdjStruct(const NodeAdjStruct& nodeAdjStruct) {
	if (nodeAdjStruct.N) {
		N = nodeAdjStruct.N;
		adjIdx.malloc(N + 1);
		memcpy(adjIdx, nodeAdjStruct.adjIdx, (N + 1) * sizeof(unsigned));
		adj.malloc(adjIdx[N]);
		memcpy(adj, nodeAdjStruct.adj, adjIdx[N] * sizeof(unsigned));
	}
}


NodeAdjStruct::NodeAdjStruct(unsigned nodeCount, unsigned elemTypes, const FiniteElement* elemInfo, const unsigned* elem) {
	init(nodeCount, elemTypes, elemInfo, elem);
}


void NodeAdjStruct::init(unsigned nodeCount, unsigned elemTypes, const FiniteElement* elemInfo, const unsigned* elem) {
	N = nodeCount;

	hptr<unsigned> linkCounts(N);
	linkCounts.setZero(N);

	for (unsigned i = 0; i < elemTypes; ++i) {
		const FiniteElement& type = elemInfo[i];
#pragma omp parallel for
		for (omp_for_t j = type.memIdx; j < type.memIdx + type.elemCount * type.nodeCount; ++j)
#pragma omp atomic
			linkCounts[elem[j]] += type.nodeCount - 1;  // подсчёт связей с повторениями
	}

	unsigned dataWidth = 0;
#pragma omp parallel for
	for (omp_for_t i = 0; i < N; ++i) {
		unsigned value = linkCounts[i];
		if (value > dataWidth)
#pragma omp critical
			if (value > dataWidth)
				dataWidth = value;  // вычисление требуемой памяти
	}

	hptr<unsigned> allLinks(dataWidth * N);
	linkCounts.setZero(N);

	hptr<omp_lock_t> locks(N);
	for (unsigned i = 0; i < N; ++i)
		omp_init_lock(locks + i);

	for (unsigned t = 0; t < elemTypes; ++t) {
		const FiniteElement& type = elemInfo[t];
#pragma omp parallel for
		for (omp_for_t e = type.memIdx; e < type.memIdx + type.elemCount * type.nodeCount; e += type.nodeCount) {
			for (unsigned i = e; i < e + type.nodeCount; ++i) {
				unsigned node = elem[i];
				omp_set_lock(locks + node);
				unsigned dataPos = node * dataWidth + linkCounts[node];
				linkCounts[node] += type.nodeCount - 1;
				omp_unset_lock(locks + node);
				for (unsigned j = e; j < e + type.nodeCount; ++j)
					if (i != j) {
						unsigned link = elem[j];
						allLinks[dataPos++] = link;  // заполнение связей с повторениями
					}
			}
		}
	}
	for (unsigned i = 0; i < N; ++i)
		omp_destroy_lock(locks + i);
	locks.free();

	hptr<unsigned> unLinkCount(N);
#pragma omp parallel for
	for (omp_for_t node = 0; node < N; ++node) {
		unLinkCount[node] = 0;
		for (unsigned i = 0; i < linkCounts[node]; ++i) {
			unsigned link = allLinks[node * dataWidth + i];
			bool no = true;
			for (unsigned j = 0; j < unLinkCount[node]; ++j) {
				if (allLinks[dataWidth * node + j] == link) {
					no = false;
					break;
				}
			}
			if (no) {  // заполнение связей без повторений с избытком по памяти
				allLinks[dataWidth * node + unLinkCount[node]] = link;
				++unLinkCount[node];
			}
		}
	}
	adjIdx.realloc(N + 1);
	unsigned adjDataSize = 0;
	adjIdx[0] = 0;
	for (unsigned i = 0; i < N; ++i) {
		adjDataSize += unLinkCount[i];
		adjIdx[i + 1] = adjDataSize;
	}

	adj.realloc(adjDataSize);
#pragma omp parallel for
	for (int node = 0; node < N; ++node) {
		int begin = adjIdx[node];         // сжатие в структуру без избытка и сортировка
		memcpy(adj + begin, allLinks + node * dataWidth, unLinkCount[node] * sizeof(unsigned));
		std::sort(adj + begin, adj + begin + unLinkCount[node]);
	}
}


void NodeAdjStruct::print() const {
	for (unsigned i = 0; i < N; ++i) {
		std::cout << i << ":  ";
		for (unsigned j = adjIdx[i]; j < adjIdx[i + 1]; ++j)
			std::cout << adj[j] << " ";
		std::cout << " (" << adjIdx[i + 1] - adjIdx[i] << ")\n";
	}
}


void NodeAdjStruct::clear() {
	N = 0;
	adj.free();
	adjIdx.free();
}