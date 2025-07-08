#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Mesh.cuh"
#include "CudaSLAE.cu"

//#include "CudaSparseSLAE.cu"

__global__ void calcLinks3(int* linkCounts, int* elem, int count4);
__global__ void calcLinks4(int* linkCounts, int* elem, int count4);
__global__ void calcLinks8(int* linkCounts, int* elem, int count4);

__global__ void clearLinks(int* linkCounts);

__global__ void fillUniqueLinks(int* allLinks, int* uniqueLinks, int* linkCounts, int* unLinkCount, int dataWidth);

__global__ void fillAdjRows(int* adj, int* adjPos, int* rows, int* unLinkCount, int* uniqueLinks, int dataWidth, int dim, int nodeCount);

__global__ void fillCols(int* rows, int* cols, int* adj, int* adjPos, int* unLinkCount, int dim, int nodeCount);

__global__ void copyRow(double* data, int* rows, int* cols, double* matrix, int W, int N);

__global__ void copyRow(float* data, int* rows, int* cols, float* matrix, int W, int N);

__global__ void clearRow(int* rows, int* cols, double* matrix, int W, int N);

__global__ void clearRow(int* rows, int* cols, float* matrix, int W, int N);

__global__ void copyRowBack(double* data, int* rows, int* cols, double* matrix, int W, int N);

template<typename fp>
class CudaSparseSLAE {

private:

	void constructor(Mesh& mesh, int dim);

	void constructorByCPU(Mesh& mesh, int dim);

public:

	int N = 0; //размер СЛАУ
	int memLen = 0;
	int dim = 1;

	fp* rp = nullptr;

	fp* data = nullptr;
	int* rows = nullptr;
	int* cols = nullptr;

	int* adj = nullptr;
	int* adjPos = nullptr;

	CudaSparseSLAE() = default;

	CudaSparseSLAE(Mesh& mesh, int dim = 2) {
		//constructor(mesh, dim);
		constructorByCPU(mesh, dim);
	}

	~CudaSparseSLAE() {
		cudaFree(rp);
		cudaFree(data);
		cudaFree(rows);
		cudaFree(cols);
		cudaFree(adj);
		cudaFree(adjPos);
	}

	void copy(CudaSLAE<fp>& slae);

	void copyBack(CudaSLAE<fp>& slae);

	void clearStrip(CudaSLAE<fp>& slae);

	void printData() const;

	fp matrixHash() const;

	fp rpHash() const;

};

template<typename fp>
void CudaSparseSLAE<fp>::copy(CudaSLAE<fp>& slae) {
	//std::cout << "memlen " << memLen << "\n";
	copyRow<<<memLen / BS, BS>>>(data, rows, cols, slae.matrix, slae.W, slae.N);
	cudaDeviceSynchronize();
}

template<typename fp>
void CudaSparseSLAE<fp>::copyBack(CudaSLAE<fp>& slae) {
	//std::cout << "memlen " << memLen << "\n";
	copyRowBack<<<memLen / BS, BS>>>(data, rows, cols, slae.matrix, slae.W, slae.N);
	cudaDeviceSynchronize();
}

template<typename fp>
void CudaSparseSLAE<fp>::constructor(Mesh& mesh, int dim) {
	CudaSparseSLAE::dim = dim;
	N = mesh.nodeCount * dim;
	memLen = (N + BS - 1) / BS * BS;
	int nodeMemLen = (mesh.nodeCount + BS - 1) / BS * BS;
	int elem4MemLen = (mesh.count4 + BS - 1) / BS * BS;
	int elem8MemLen = (mesh.count8 + BS - 1) / BS * BS;
	int* dev_linkCounts = nullptr;
	cudaDeviceSynchronize();
	cudaMalloc((void**)&dev_linkCounts, nodeMemLen * sizeof(int));
	//clearLinks<<<nodeMemLen / BS, BS>>>(dev_linkCounts);
	cudaMemset(dev_linkCounts, 0, nodeMemLen * sizeof(int));
	//cudaMemcpy(mesh.elem4, mesh.dev_elem4, 4 * mesh.count4 * sizeof(int), cudaMemcpyDeviceToHost);
	//DEBUG
	/*if (mesh.nodeCount == 3000) {
		int* loc_elem4 = new int[4 * elem4MemLen];
		cudaMemcpy(loc_elem4, mesh.dev_elem4, 4 * mesh.count4 * sizeof(int), cudaMemcpyDeviceToHost);
		for (int i = 0; i < mesh.count4; ++i) {
			for (int j = 0; j < 4; ++j) {
				std::cout << loc_elem4[4 * i + j] << " ";
			}
			std::cout << "\n";
		}
		delete[] loc_elem4;
	}*/
	//DEBUG

	//std::cout << "elem4MemLen " << elem4MemLen << "\n";
	//std::cout << "nodeMemLen " << nodeMemLen << "\n";
	//std::cout << "count4 " << mesh.count4 << "\n";
	calcLinks4<<<elem4MemLen / BS, BS>>>(dev_linkCounts, mesh.dev_elem4, mesh.count4);
	calcLinks8<<<elem8MemLen / BS, BS>>>(dev_linkCounts, mesh.dev_elem8, mesh.count8);
	cudaDeviceSynchronize();

	//std::clog << "log01\n";
	//std::clog << mesh.count8 << "\n";
	int dataWidth = 0;
	int* linkCounts = new int[mesh.nodeCount];
	cudaMemcpy(linkCounts, dev_linkCounts, mesh.nodeCount * sizeof(int), cudaMemcpyDeviceToHost);
	//clearLinks<<<nodeMemLen / BS, BS>>>(dev_linkCounts);
	cudaMemset(dev_linkCounts, 0, nodeMemLen * sizeof(int));
	//...8?
	for (int i = 0; i < mesh.nodeCount; ++i) {
		if (linkCounts[i] > dataWidth)
			dataWidth = linkCounts[i];   //вычисление требуемой памяти
		//if (mesh.nodeCount == 3000) std::cout << i << " " << linkCounts[i] << "\n";
		linkCounts[i] = 0;
	}
	//dataWidth = 12;
	std::clog << dataWidth << " " << nodeMemLen << "\n";
	int dataSize = dataWidth * nodeMemLen;

	int* allLinks = nullptr;
	int* uniqueLinks = nullptr;
	cudaMalloc((void**)&allLinks, dataSize * sizeof(int));
	cudaMalloc((void**)&uniqueLinks, dataSize * sizeof(int));
	cudaDeviceSynchronize();
	//std::clog << "log02\n";
	//DEBUG
	/*for (int i = 0; i < mesh.nodeCount; ++i)
		std::cout << i << " " << linkCounts[i] << "\n";*/
	//DEBUG

	//DEBUG
	/*int* locElem = new int[mesh.count4 * 4];
	cudaMemcpy(locElem, mesh.dev_elem, mesh.count4 * 4 * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < mesh.count4 * 4; ++i)
		std::cout << locElem[i] << " ";
	std::cout << "\n";
	delete[] locElem;*/
	//DEBUG

	//fillAllLinks<<<elem4MemLen / BS, BS>>>(allLinks, dev_linkCounts, mesh.dev_elem, mesh.count4, dataWidth);
	int* elem4 = nullptr;
	int* elem8 = nullptr;
	if (mesh.ramSaved) {
		elem4 = mesh.elem4;
		elem8 = mesh.elem8;
	}
	else {
		//std::clog << "log1\n";
		elem4 = new int[mesh.count4 * 4];
		elem8 = new int[mesh.count8 * 8];
		cudaMemcpy(elem4, mesh.dev_elem4, mesh.count4 * 4 * sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(elem8, mesh.dev_elem8, mesh.count8 * 8 * sizeof(int), cudaMemcpyDeviceToHost);
		//std::clog << "log1\n";
	}
	//std::clog << dataSize << " log03\n";
	int* locAllLinks = new int[dataSize];
	//std::clog << "log2\n";
	for (int e = 0; e < 4 * mesh.count4; e += 4)
		for (int i = e; i < e + 4; ++i) {
			int node = elem4[i];
			for (int j = e; j < e + 4; ++j) {   //заполнение связей с повторениями
				if (i != j) {
					int link = elem4[j];
					locAllLinks[node * dataWidth + linkCounts[node]] = link;
					++linkCounts[node];
				}
			}
		}
	for (int e = 0; e < 8 * mesh.count8; e += 8)
		for (int i = e; i < e + 8; ++i) {
			int node = elem8[i];
			for (int j = e; j < e + 8; ++j) {   //заполнение связей с повторениями
				if (i != j) {
					int link = elem8[j];
					locAllLinks[node * dataWidth + linkCounts[node]] = link;
					++linkCounts[node];
				}
			}
		}
	//std::clog << "log04\n";
	//std::clog << "log3\n";
	//std::clog << dataSize << "\n";
	cudaMemcpy(allLinks, locAllLinks, dataSize * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_linkCounts, linkCounts, mesh.nodeCount * sizeof(int), cudaMemcpyHostToDevice);
	//std::clog << "log04.5\n";
	delete[] locAllLinks;
	if (!mesh.ramSaved) {
		delete[] elem4;
		delete[] elem8;
	}
	int* dev_unLinkCount = nullptr;
	cudaMalloc((void**)&dev_unLinkCount, nodeMemLen * sizeof(int));
	cudaDeviceSynchronize();
	//std::clog << "log05\n";
	//DEBUG
	/*int* locAllLinks = new int[dataSize];
	cudaMemcpy(locAllLinks, allLinks, dataSize * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < mesh.nodeCount * 2; ++i) {
		std::cout << i << ", " << linkCounts[i] << ":  ";
		for (int j = 0; j < dataWidth; ++j)
			std::cout << locAllLinks[i * dataWidth + j] << " ";
		std::cout << "\n";
	}
	delete[] locAllLinks;*/
	//DEBUG

	fillUniqueLinks<<<nodeMemLen / BS, BS>>>(allLinks, uniqueLinks, dev_linkCounts, dev_unLinkCount, dataWidth);
	cudaDeviceSynchronize();
	//cudaDeviceSynchronize();
	//std::clog << "log06\n";
	//DEBUG
	//int* locUnLinks = new int[dataSize];
	//cudaMemcpy(locUnLinks, uniqueLinks, dataSize * sizeof(int), cudaMemcpyDeviceToHost);
	//for (int i = 0; i < mesh.nodeCount; ++i) {
	//	//std::cout << i << ", " << linkCounts[i] << ":  ";
	//	std::cout << i << ":  ";
	//	for (int j = 0; j < dataWidth; ++j)
	//		std::cout << locUnLinks[i * dataWidth + j] << " ";
	//	std::cout << "\n";
	//}
	//delete[] locUnLinks;
	//DEBUG

	cudaMalloc((void**)&rp, memLen * sizeof(fp));
	cudaMalloc((void**)&rows, (memLen + 1) * sizeof(int));
	cudaMemset(rows, 0, (memLen + 1) * sizeof(int));
	cudaMalloc((void**)&adjPos, (memLen / dim + 1) * sizeof(int));
	
	//std::clog << "log07\n";

	//unlinks cpu
	int* unLinkCount = new int[mesh.nodeCount];
	cudaMemcpy(unLinkCount, dev_unLinkCount, mesh.nodeCount * sizeof(int), cudaMemcpyDeviceToHost);
	int* locAdjPos = new int[mesh.nodeCount];
	int adjDataSize = 0;
	for (int i = 0; i < mesh.nodeCount; ++i) {  //последовательный цикл
		adjDataSize += unLinkCount[i];
		locAdjPos[i] = adjDataSize;
	}
	cudaMemcpy(adjPos + 1, locAdjPos, mesh.nodeCount * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&adj, adjDataSize * sizeof(int));
	dataSize = (adjDataSize + mesh.nodeCount) * dim * dim;
	cudaMalloc((void**)&data, dataSize * sizeof(fp));
	cudaMalloc((void**)&cols, dataSize * sizeof(int));

	fillAdjRows<<<nodeMemLen / BS, BS>>>(adj, adjPos, rows, dev_unLinkCount, uniqueLinks, dataWidth, dim, mesh.nodeCount);
	cudaDeviceSynchronize();
	fillCols<<<nodeMemLen / BS, BS>>>(rows, cols, adj, adjPos, dev_unLinkCount, dim, mesh.nodeCount);
	cudaDeviceSynchronize();

	//DEBUG
	/*int* locRows = new int[N + 1];
	cudaMemcpy(locRows, rows, (N + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i <= N; ++i)
		std::cout << locRows[i] << " ";
	std::cout << "\n";
	delete[] locRows;*/
	//DEBUG

	delete[] linkCounts;
	delete[] unLinkCount;
	cudaFree(dev_linkCounts);
	cudaFree(dev_unLinkCount);
	cudaFree(allLinks);
	cudaFree(uniqueLinks);
}

template<typename fp>
void CudaSparseSLAE<fp>::constructorByCPU(Mesh& mesh, int dim) {
	CudaSparseSLAE::dim = dim;
	N = mesh.nodeCount * dim;
	memLen = (N + BS - 1) / BS * BS;

	if (!mesh.ramSaved)
		mesh.meshToRAM();
	int* linkCounts = new int[mesh.nodeCount];
#pragma omp parallel for
	for (int i = 0; i < mesh.nodeCount; ++i)
		linkCounts[i] = 0;

#pragma omp parallel for
	for (int i = 0; i < 3 * mesh.count3; ++i)
		linkCounts[mesh.elem3[i]] += 2;    //подсчёт связей с повторениями
#pragma omp parallel for
	for (int i = 0; i < 4 * mesh.count4; ++i)
		linkCounts[mesh.elem4[i]] += 3;
#pragma omp parallel for
	for (int i = 0; i < 8 * mesh.count8; ++i)
		linkCounts[mesh.elem8[i]] += 7;
	int dataWidth = 0;

	for (int i = 0; i < mesh.nodeCount; ++i)
		if (linkCounts[i] > dataWidth)
			dataWidth = linkCounts[i];   //вычисление требуемой памяти
	int dataSize = dataWidth * mesh.nodeCount;
	//std::clog << dataWidth << " " << mesh.nodeCount << "\n";

	int* allLinks = new int[dataSize];
	int* uniqueLinks = new int[dataSize];
#pragma omp parallel for
	for (int i = 0; i < mesh.nodeCount; ++i)
		linkCounts[i] = 0;

	for (int e = 0; e < 3 * mesh.count3; e += 3)
		for (int i = e; i < e + 3; ++i) {
			int node = mesh.elem3[i];
			for (int j = e; j < e + 3; ++j) {   //заполнение связей с повторениями
				if (i != j) {
					int link = mesh.elem3[j];
					allLinks[node * dataWidth + linkCounts[node]] = link;
					++linkCounts[node];
				}
			}
		}
	for (int e = 0; e < 4 * mesh.count4; e += 4)
		for (int i = e; i < e + 4; ++i) {
			int node = mesh.elem4[i];
			for (int j = e; j < e + 4; ++j) {   //заполнение связей с повторениями
				if (i != j) {
					int link = mesh.elem4[j];
					allLinks[node * dataWidth + linkCounts[node]] = link;
					++linkCounts[node];
				}
			}
		}
	for (int e = 0; e < 8 * mesh.count8; e += 8)
		for (int i = e; i < e + 8; ++i) {
			int node = mesh.elem8[i];
			for (int j = e; j < e + 8; ++j) {   //заполнение связей с повторениями
				if (i != j) {
					int link = mesh.elem8[j];
					allLinks[node * dataWidth + linkCounts[node]] = link;
					++linkCounts[node];
				}
			}
		}

	int* unLinkCount = new int[mesh.nodeCount];
#pragma omp parallel for
	for (int node = 0; node < mesh.nodeCount; ++node) {
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
	rp = new double[mesh.nodeCount * dim];//to gpu
	cudaMalloc((void**)&rp, memLen * sizeof(fp));
	int* loc_rows = new int[N + 1];
	cudaMalloc((void**)&rows, (memLen + 1) * sizeof(int));
	cudaMemset(rows, 0, (memLen + 1) * sizeof(int));
	int* loc_adjPos = new int[mesh.nodeCount + 1];
	int adjDataSize = 0;
	loc_adjPos[0] = 0;
	for (int i = 0; i < mesh.nodeCount; ++i) {  //последовательный цикл
		adjDataSize += unLinkCount[i];
		loc_adjPos[i + 1] = adjDataSize;
	}

	int* loc_adj = new int[adjDataSize];
	/*ins = new int[adjDataSize];
	insSym = new int[adjDataSize];*/

	dataSize = (adjDataSize + mesh.nodeCount) * dim * dim;
	cudaMalloc((void**)&data, dataSize * sizeof(fp));
	int* loc_cols = new int[dataSize];
	cudaMalloc((void**)&cols, dataSize * sizeof(int));

#pragma omp parallel for
	for (int node = 0; node < mesh.nodeCount; ++node) {
		int begin = loc_adjPos[node], size = loc_adjPos[node + 1] - begin;
		for (int link = 0; link < unLinkCount[node]; ++link) {
			loc_adj[begin + link] = uniqueLinks[node * dataWidth + link];
		}
		for (int i = 0; i < dim; ++i)
			loc_rows[dim * node + i + 1] = begin * dim * dim + (i + 1) * size * dim + (dim * node + i + 1) * dim;
	}

	//std::clog << "log3\n";
	loc_rows[0] = 0;
#pragma omp parallel for
	for (int node = 0; node < mesh.nodeCount; ++node) {
		int begin = loc_adjPos[node], size = loc_adjPos[node + 1] - begin;

		for (int i = 0; i < dim; ++i) {
			for (int j = 0; j < dim; ++j) //диагональные столбцы
				loc_cols[loc_rows[dim * node + i] + j] = dim * node + j;
			//остальные столбцы
			for (int link = 0; link < unLinkCount[node]; ++link) {
				int insBegin = loc_rows[dim * node + i] + (link + 1) * dim;
				for (int j = 0; j < dim; ++j)
					loc_cols[insBegin + j] = loc_adj[begin + link] * dim + j;
			}
		}
	}
	//transfer to gpu
	cudaMemcpy(rows, loc_rows, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(cols, loc_cols, dataSize * sizeof(int), cudaMemcpyHostToDevice);
	/*for (int i = N - 10; i <= N; ++i)
		std::clog << loc_rows[i] << " ";
	std::clog << "\n";*/

	//std::clog << "formed\n";
	delete[] linkCounts;
	delete[] allLinks;
	delete[] uniqueLinks;
	delete[] unLinkCount;

	delete[] loc_adj;
	delete[] loc_adjPos;
	delete[] loc_cols;
	delete[] loc_rows;
}

template<typename fp>
void CudaSparseSLAE<fp>::printData() const {
	//DEBUG
	int* locRows = new int[N + 1];
	cudaMemcpy(locRows, rows, (N + 1) * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i <= N; ++i)
		std::cout << locRows[i] << " ";
	std::cout << "\n";
	
	//DEBUG
	int dataLen = 0;
	cudaMemcpy(&dataLen, rows + N, sizeof(int), cudaMemcpyDeviceToHost);
	std::cout << "datalen " << dataLen << "\n";
	int* locCols = new int[dataLen];
	cudaMemcpy(locCols, cols, dataLen * sizeof(int), cudaMemcpyDeviceToHost);
	std::cout << "N = " << N << "\n";
	/*for (int i = 0; i < N; ++i) {
		std::cout << i << "::  ";
		for (int j = locRows[i]; j < locRows[i + 1]; ++j)
			std::cout << locCols[j] << " ";
		std::cout << "\n";
	}*/

	/*for (int i = 0; i < dataLen; ++i)
		std::cout << locCols[i] << " ";
	std::cout << "\n";*/
	
	fp* locData = new fp[dataLen];
	cudaMemcpy(locData, data, dataLen * sizeof(fp), cudaMemcpyDeviceToHost);
	for (int i = 0; i < dataLen; ++i)
		std::cout << locData[i] << " ";
	std::cout << "\n";
	

	delete[] locRows;
	delete[] locCols;
	delete[] locData;
}

template<typename fp>
void CudaSparseSLAE<fp>::clearStrip(CudaSLAE<fp>& slae) {
	clearRow<<<memLen / BS, BS>>>(rows, cols, slae.matrix, slae.W, slae.N);
	cudaDeviceSynchronize();
}


template<typename fp>
fp CudaSparseSLAE<fp>::matrixHash() const {
	int dataSize = 0;
	cudaMemcpy(&dataSize, rows + N, sizeof(int), cudaMemcpyDeviceToHost);
	fp* locData = new fp[dataSize];
	cudaMemcpy(locData, data, dataSize * sizeof(fp), cudaMemcpyDeviceToHost);
	fp sum = 0.;
	for (int i = 0; i < dataSize; ++i)
		sum += fabs(locData[i]);
	//return dataSize;
	return sum;
}

template<typename fp>
fp CudaSparseSLAE<fp>::rpHash() const {
	fp* locRp = new fp[memLen];
	cudaMemcpy(locRp, rp, memLen * sizeof(fp), cudaMemcpyDeviceToHost);
	fp sum = 0.;
	for (int i = 0; i < memLen; ++i)
		sum += fabs(locRp[i]);
	return sum;
}