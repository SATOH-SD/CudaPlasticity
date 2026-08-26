#pragma once

#include <iostream>

#include "StripSLAE.h"
#include "Mesh.h"

#include "cuda_runtime.h" //TEMP

#include <iostream>
#include <fstream>
#include <algorithm>
#include <omp.h>


#if _OPENMP >= 200805
typedef unsigned omp_for_t;
#else
typedef int omp_for_t;
#endif


//Разреженная СЛАУ (CSR)
class SparseSLAE {

public:

	int N = 0;   //размер СЛАУ

	int dim = 1; //размерность узловых величин

	double* rp = nullptr;

	double* data = nullptr;
	int* rows = nullptr;
	int* cols = nullptr;

	int* adj = nullptr;
	int* adjPos = nullptr;
	//int* ins = nullptr;     //место вставки в строке
	//int* insSym = nullptr;  //место вставки в строке (симметрично)

	SparseSLAE(StripSLAE& slae) {
		N = slae.factN;
		rp = new double[N];
		rows = new int[N + 1];
		int dataK = 0;
		int memW = 2 * slae.W + 1;
		rows[0] = 0;
		for (int i = 0; i < N; ++i) {
			rp[i] = slae.rp[i];
			for (int j = 0; j < memW; ++j)
				if (slae.matrix[i * memW + j] != 0.)
					++dataK;
			rows[i + 1] = dataK;
		}
		//std::cout << "dataK " << dataK << "\n";
		data = new double[dataK];
		cols = new int[dataK];
		dataK = 0;
		for (int i = 0; i < N; ++i) {
			for (int j = 0; j < memW; ++j) {
				double value = slae.matrix[i * memW + j];
				if (value != 0.) {
					//std::cout << value << " ";
					data[dataK] = value;
					cols[dataK] = i - (int)slae.W + j;
					++dataK;
				}
			}
			//std::cout << "\n";
		}
	}

	SparseSLAE(Mesh& mesh, int dim = 2) {
		SparseSLAE::dim = dim;
		N = mesh.nodeCount * dim;
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

		int* allLinks = new int[dataSize];
		memset(linkCounts, 0, mesh.nodeCount * sizeof(int));

		omp_lock_t* locks = new omp_lock_t[mesh.nodeCount];
		for (unsigned i = 0; i < mesh.nodeCount; ++i)
			omp_init_lock(locks + i);

#pragma omp parallel
		{
#pragma omp for
			for (int e = 0; e < 3 * mesh.count3; e += 3)
				for (int i = e; i < e + 3; ++i) {
					int node = mesh.elem3[i];
					omp_set_lock(locks + node);
					int dataPos = node * dataWidth + linkCounts[node];
					linkCounts[node] += 2;
					omp_unset_lock(locks + node);
					for (int j = e; j < e + 3; ++j)    //заполнение связей с повторениями
						if (i != j) {
							int link = mesh.elem3[j];
							allLinks[dataPos++] = link;
						}
				}
#pragma omp for
			for (int e = 0; e < 4 * mesh.count4; e += 4)
				for (int i = e; i < e + 4; ++i) {
					int node = mesh.elem4[i];
					omp_set_lock(locks + node);
					int dataPos = node * dataWidth + linkCounts[node];
					linkCounts[node] += 3;
					omp_unset_lock(locks + node);
					for (int j = e; j < e + 4; ++j)    //заполнение связей с повторениями
						if (i != j) {
							int link = mesh.elem4[j];
							allLinks[dataPos++] = link;
						}
				}
#pragma omp for
			for (int e = 0; e < 8 * mesh.count8; e += 8)
				for (int i = e; i < e + 8; ++i) {
					int node = mesh.elem8[i];
					omp_set_lock(locks + node);
					int dataPos = node * dataWidth + linkCounts[node];
					linkCounts[node] += 7;
					omp_unset_lock(locks + node);
					for (int j = e; j < e + 8; ++j)    //заполнение связей с повторениями
						if (i != j) {
							int link = mesh.elem8[j];
							allLinks[dataPos++] = link;
						}
				}
		}

		for (unsigned i = 0; i < mesh.nodeCount; ++i)
			omp_destroy_lock(locks + i);
		delete[] locks;
		/*for (int i = 0; i < mesh.nodeCount; ++i)
			std::cout << linkCounts[i] << "\n";*/

		int* unLinkCount = new int[mesh.nodeCount];
#pragma omp parallel for
		for (int node = 0; node < mesh.nodeCount; ++node) {
			unLinkCount[node] = 0;
			for (int i = 0; i < linkCounts[node]; ++i) {
				int link = allLinks[node * dataWidth + i];
				bool no = true;
				for (int j = 0; j < unLinkCount[node]; ++j) {
					if (allLinks[dataWidth * node + j] == link) {
						no = false;
						break;
					}
				}
				if (no) {  //заполнение связей без повторений с избытком по памяти
					allLinks[dataWidth * node + unLinkCount[node]] = link;
					++unLinkCount[node];
				}
			}
		}
		//for (int i = 0; i < mesh.nodeCount; ++i)
		//	//if (unLinkCount[i] != 8 && unLinkCount[i] != 5)
		//	if (unLinkCount[i] != 20 && unLinkCount[i] != 12 && unLinkCount[i] != 7)
		//		std::cout << unLinkCount[i] << "\n";
		//std::cin.get();

		rp = new double[mesh.nodeCount * dim];
		rows = new int[mesh.nodeCount * dim + 1];
		adjPos = new int[mesh.nodeCount + 1];
		int adjDataSize = 0;
		adjPos[0] = 0;
		for (int i = 0; i < mesh.nodeCount; ++i) {  //последовательный цикл
			adjDataSize += unLinkCount[i];
			adjPos[i + 1] = adjDataSize;
		}

		adj = new int[adjDataSize];
		/*ins = new int[adjDataSize];
		insSym = new int[adjDataSize];*/

		dataSize = (adjDataSize + mesh.nodeCount) * dim * dim;
		//cudaMallocHost((void**)&data, dataSize * sizeof(double)); //TEMP
		data = new double[dataSize];
		cols = new int[dataSize];

#pragma omp parallel for
		for (int node = 0; node < mesh.nodeCount; ++node) {
			int begin = adjPos[node], size = adjPos[node + 1] - begin;
			for (int link = 0; link < unLinkCount[node]; ++link) {
				//adj[begin + link] = uniqueLinks[node * dataWidth + link];
				adj[begin + link] = allLinks[node * dataWidth + link];
			}
			std::sort(adj + begin, adj + begin + unLinkCount[node]);
			for (int i = 0; i < dim; ++i)
				rows[dim * node + i + 1] = begin * dim * dim + (i + 1) * size * dim + (dim * node + i + 1) * dim;
		}

		//std::clog << "log3\n";
		rows[0] = 0;
#pragma omp parallel for
		for (int node = 0; node < mesh.nodeCount; ++node) {
			int begin = adjPos[node], size = adjPos[node + 1] - begin;

			for (int i = 0; i < dim; ++i) {
				for (int j = 0; j < dim; ++j) //диагональные столбцы
					cols[rows[dim * node + i] + j] = dim * node + j;
				//остальные столбцы
				for (int link = 0; link < unLinkCount[node]; ++link) {
					int insBegin = rows[dim * node + i] + (link + 1) * dim;
					for (int j = 0; j < dim; ++j)
						cols[insBegin + j] = adj[begin + link] * dim + j;
				}
			}

		}
		//std::clog << "formed\n";
		delete[] linkCounts;
		delete[] allLinks;
		delete[] unLinkCount;
	}

	~SparseSLAE() {
		//cudaFreeHost(data); //TEMP
		delete[] data;
		delete[] rp;
		delete[] rows;
		delete[] cols;
		delete[] adj;
		delete[] adjPos;
		//delete[] ins;
		//delete[] insSym;
	}

	void insert();

	void insertSym();

	void insertDiag();

	double hashMatrix() const {
		double sum = 0.;
		for (int i = 0; i < rows[N]; ++i)
			sum += data[i];
		return sum;
	}

	void copy(const StripSLAE& slae) {
		for (int i = 0; i < N; ++i)
			for (int j = rows[i]; j < rows[i + 1]; ++j)
				data[j] = slae(i, cols[j]);
	}

	void clearStrip(StripSLAE& slae) const {
		for (int i = 0; i < N; ++i)
			for (int j = rows[i]; j < rows[i + 1]; ++j)
				slae(i, cols[j]) = 0.;
	}

	void print() const {
		std::cout << "\nrows\n";
		for (int i = 0; i <= N; ++i)
			std::cout << rows[i] << " ";
		std::cout << "\ncols\n";
		for (int i = 0; i < rows[N]; ++i)
			std::cout << cols[i] << " ";
		std::cout << "\ndata\n";
		for (int i = 0; i < rows[N]; ++i)
			std::cout << data[i] << " ";
	}

	void printRp() const {
		for (int i = 0; i < N; ++i)
			std::cout << rp[i] << "\n";
	}

	void printMatrixStruct() const {
		std::cout << "\nrows\n";
		for (int i = 0; i <= N; ++i)
			std::cout << rows[i] << " ";
		std::cout << "\ncols\n";
		for (int i = 0; i < N; ++i) {
			std::cout << i << ":  ";
			for (int j = rows[i]; j < rows[i + 1]; ++j)
				std::cout << cols[j] << " ";
			std::cout << "\n";
		}
	}

	void printStruct() const {

	}

	void saveToFile(const std::string& fileName) {
		std::ofstream file(fileName, std::ios_base::out);
		file << N << "\n";
		for (int i = 0; i <= N; ++i)
			file << rows[i] << " ";
		file << "\n";
		for (int i = 0; i < rows[N]; ++i)
			file << cols[i] << " ";
		file << "\n";
		for (int i = 0; i < rows[N]; ++i)
			file << data[i] << " ";
		file << "\n";
		for (int i = 0; i < N; ++i)
			file << rp[i] << " ";
		file.close();
		std::cout << "file saved\n";
	}

};