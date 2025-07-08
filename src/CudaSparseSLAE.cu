#include "cuda.h"

#include "CudaSparseSLAE.cuh"

__global__ void calcLinks4(int* linkCounts, int* elem, int count4) {
	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e < count4) {
		int i = 4 * e;
		/*linkCounts[elem[i++]] += 3;
		linkCounts[elem[i++]] += 3;
		linkCounts[elem[i++]] += 3;
		linkCounts[elem[i++]] += 3;*/
		atomicAdd(linkCounts + elem[i++], 3);
		atomicAdd(linkCounts + elem[i++], 3);
		atomicAdd(linkCounts + elem[i++], 3);
		atomicAdd(linkCounts + elem[i++], 3);
		//if (count4 == 2880) printf("%d\n", linkCounts[elem[i - 3]]);
		//if (count4 == 2880) printf("%d %d %d %d\n", elem[4 * e], elem[4 * e + 1], elem[4 * e + 2], elem[4 * e + 3]);
	}
}
__global__ void calcLinks8(int* linkCounts, int* elem, int count8) {
	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e < count8) {
		//printf("%d\n", e);
		int i = 8 * e;
		//printf("%d:  %d %d %d %d %d %d %d %d\n", e, elem[i], elem[i + 1], elem[i + 2], elem[i + 3], elem[i + 4], elem[i + 5], elem[i + 6], elem[i + 7]);
		atomicAdd(linkCounts + elem[i++], 7);
		//linkCounts[elem[i++]] += 7;
		//printf("%d\n", linkCounts[elem[i - 1]]);
		atomicAdd(linkCounts + elem[i++], 7);
		atomicAdd(linkCounts + elem[i++], 7);
		atomicAdd(linkCounts + elem[i++], 7);
		atomicAdd(linkCounts + elem[i++], 7);
		atomicAdd(linkCounts + elem[i++], 7);
		atomicAdd(linkCounts + elem[i++], 7);
		atomicAdd(linkCounts + elem[i++], 7);
		
	}
}

__global__ void clearLinks(int* linkCounts) {
	linkCounts[blockIdx.x * blockDim.x + threadIdx.x] = 0;
}

//__global__ void fillAllLinks(int* allLinks, int* linkCounts, int* elem, int count4, int dataWidth) {
//	int e = blockIdx.x * blockDim.x + threadIdx.x;
//	if (e < count4) {
//		for (int i = e; i < e + 4; ++i) {
//			int node = elem[i];
//			//if (i == e) printf("-- %d\n", node);
//			for (int j = e; j < e + 4; ++j) {   //заполнение связей с повторениями
//				if (i != j) {
//					int link = elem[j];
//					allLinks[node * dataWidth + linkCounts[node]] = link;
//					//++linkCounts[node];
//					atomicAdd(linkCounts + node, 1);
//				}
//			}
//		}
//	}
//}

__global__ void fillUniqueLinks(int* allLinks, int* uniqueLinks, int* linkCounts, int* unLinkCount, int dataWidth) {
	int node = blockIdx.x * blockDim.x + threadIdx.x;
	unLinkCount[node] = 0;
	for (int i = 0; i < linkCounts[node]; ++i) {
		int link = allLinks[node * dataWidth + i];
		bool no = true;
		for (int j = 0; j < unLinkCount[node]; ++j) {
			if (uniqueLinks[dataWidth * node + j] == link) {
				no = false;
				break;
			}
		}
		if (no) {  //заполнение связей без повторонений с избытком по памяти
			uniqueLinks[dataWidth * node + unLinkCount[node]] = link;
			++unLinkCount[node];
		}
	}
}

__global__ void fillAdjRows(int* adj, int* adjPos, int* rows, int* unLinkCount, int* uniqueLinks, int dataWidth, int dim, int nodeCount) {
	int node = blockIdx.x * blockDim.x + threadIdx.x;
	int begin = adjPos[node], size = adjPos[node + 1] - begin;
	if (node >= nodeCount) return;

	for (int link = 0; link < unLinkCount[node]; ++link) {
		adj[begin + link] = uniqueLinks[node * dataWidth + link];

	}
	for (int i = 0; i < dim; ++i)
		rows[dim * node + i + 1] = begin * dim * dim + (i + 1) * size * dim + (dim * node + i + 1) * dim;
}

__global__ void fillCols(int* rows, int* cols, int* adj, int* adjPos, int* unLinkCount, int dim, int nodeCount) {
	int node = blockIdx.x * blockDim.x + threadIdx.x;
	int begin = adjPos[node], size = adjPos[node + 1] - begin;
	if (node >= nodeCount) return;

	for (int i = 0; i < dim; ++i) {
		for (int j = 0; j < dim; ++j) { //диагональные столбцы
			cols[rows[dim * node + i] + j] = dim * node + j;
		}
		//остальные столбцы
		for (int link = 0; link < unLinkCount[node]; ++link) {
			int insBegin = rows[dim * node + i] + (link + 1) * dim;
			for (int j = 0; j < dim; ++j)
				cols[insBegin + j] = adj[begin + link] * dim + j;
		}
	}
}

__device__
inline int strip(int i, int j, int W) {
	/*if (j > i + W || j < i - W)
		printf("!!! %d %d\n", i, j);*/
	return i * (2 * W + 1) + j + W - i;
}

__global__ void copyRowBack(double* data, int* rows, int* cols, double* matrix, int W, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("-- %d", i);
	if (i < N)
		for (int j = rows[i]; j < rows[i + 1]; ++j)
			matrix[strip(i, cols[j], W)] = data[j];
}

__global__ void copyRow(double* data, int* rows, int* cols, double* matrix, int W, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("-- %d", i);
	if (i < N)
		for (int j = rows[i]; j < rows[i + 1]; ++j)
			data[j] = matrix[strip(i, cols[j], W)];
}

__global__ void copyRow(float* data, int* rows, int* cols, float* matrix, int W, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
		for (int j = rows[i]; j < rows[i + 1]; ++j)
			data[j] = matrix[strip(i, cols[j], W)];
}

__global__ void clearRow(int* rows, int* cols, double* matrix, int W, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
		for (int j = rows[i]; j < rows[i + 1]; ++j)
			matrix[strip(i, cols[j], W)] = 0.;
}

__global__ void clearRow(int* rows, int* cols, float* matrix, int W, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N)
		for (int j = rows[i]; j < rows[i + 1]; ++j)
			matrix[strip(i, cols[j], W)] = 0.f;
}