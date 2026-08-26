#pragma once

#include <cmath>
#include <string>
#include <functional>

#include "FiniteElement.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ptrs.h"
#include "vec2.cuh"

#include "NodeAdjStruct.h"
#include "ColorMap.h"

#include "CudaChecker.h" // TEMP


const int BS = 1024; //Стандартный размер блока CUDA



//Класс сетки
class Mesh {

private:
	
	void fillPos() {
		elemPos[0] = count3;
		elemPos[1] = elemPos[0] + count4;
		elemPos[2] = elemPos[1] + count8;
	}

	void fillPosDev() {
		dev_elemPos[0] = (count3 + BS - 1) / BS * BS;
		dev_elemPos[1] = dev_elemPos[0] + (count4 + BS - 1) / BS * BS;
		dev_elemPos[2] = dev_elemPos[1] + (count8 + BS - 1) / BS * BS;
	}

	void remapOrder(int width);

	void translateFromLegacy();  // TEMP

	void setGeomForElem(GeomType geomType, unsigned elemType);

	void mallocThickness(unsigned elemType);

	void setThickness(unsigned elemType, double h);

	void setThickness(unsigned elemType, std::function<double(const double*)> h);

public:

	unsigned totalElems = 0;

	unsigned elemTypes = 0;

	hptr<FiniteElement> elemInfo;

	hptr<unsigned> elem;

	NodeAdjStruct nodeAdjStruct;

	ColorMap colorMap;

	bool useCuda = false;  // LEGACY

	bool ramSaved = false; // LEGACY

	bool analysed = false;

	int dim = 2;

	int count3 = 0;  // LEGACY
	int count4 = 0;  // LEGACY
	int count8 = 0;  // LEGACY

	int* elem3 = nullptr;  // LEGACY
	int* elem4 = nullptr;  // LEGACY
	int* elem8 = nullptr;  // LEGACY

	int* dev_elem3 = nullptr;
	int* dev_elem4 = nullptr;
	int* dev_elem8 = nullptr;

	int elemPos[3] = {};
	int dev_elemPos[3] = {};

	unsigned nodeCount = 0;

	hptr<double> node;

	vec2* node2 = nullptr;     // LEGACY
	vec2* dev_node = nullptr;

	bool* secOrdNodes = nullptr;
	bool* dev_secOrdNodes = nullptr;

	int bordersCount = 0;
	//int dev_bordersCount = 0;
	int* borderLength = nullptr;

	int** borders = nullptr;
	//int** dev_borders = nullptr;

	hptr<unsigned> border;
	hptr<unsigned> borderIdx;

	hptr<double> borderH;  // TEMP?

	double* spaces = nullptr;
	double* aspects = nullptr;
	double* skewAngles = nullptr;

	Mesh() {
		CudaChecker checker;
		useCuda = checker.isGpuOn();  // TEMP
	}

	Mesh(const Mesh& mesh)
		: nodeAdjStruct(mesh.nodeAdjStruct) {
		//std::cout << "log1\n";
		useCuda = mesh.useCuda;
		ramSaved = mesh.ramSaved;
		analysed = mesh.analysed;
		count3 = mesh.count3;
		count4 = mesh.count4;
		count8 = mesh.count8;
		nodeCount = mesh.nodeCount;

		/*node.malloc(nodeCount * dim);
		memcpy(node, mesh.node, nodeCount * dim * sizeof(unsigned));*/

		/*totalElems = mesh.totalElems;
		elemInfo.malloc(elemTypes);*/

		fillPos();
		//std::cout << "log2\n";
		if (ramSaved) {
			//std::cout << "log5\n";
			node2 = new vec2[nodeCount];
			secOrdNodes = new bool[nodeCount];
			memcpy(node2, mesh.node2, nodeCount * sizeof(vec2));
			memcpy(secOrdNodes, mesh.secOrdNodes, nodeCount * sizeof(bool));
			elem3 = new int[3 * count3];
			elem4 = new int[4 * count4];
			elem8 = new int[8 * count8];
			memcpy(elem3, mesh.elem3, 3 * count3 * sizeof(int));
			memcpy(elem4, mesh.elem4, 4 * count4 * sizeof(int));
			memcpy(elem8, mesh.elem8, 8 * count8 * sizeof(int));
			bordersCount = mesh.bordersCount;
			borderLength = new int[bordersCount];
			memcpy(borderLength, mesh.borderLength, bordersCount * sizeof(int));
			borders = new int* [bordersCount];
			for (int i = 0; i < bordersCount; ++i) {
				borders[i] = new int[borderLength[i]];
				memcpy(borders[i], mesh.borders[i], borderLength[i] * sizeof(int));
			}
			if (analysed) {
				spaces = new double[count4];
				aspects = new double[count4];
				skewAngles = new double[count4];
				memcpy(spaces, mesh.spaces, count4 * sizeof(double));
				memcpy(aspects, mesh.aspects, count4 * sizeof(double));
				memcpy(skewAngles, mesh.skewAngles, count4 * sizeof(double));
			}
			translateFromLegacy();
		}
		if (useCuda) {
			cudaMalloc((void**)&dev_node, (nodeCount + BS - 1) / BS * BS * sizeof(vec2));
			cudaMemset(dev_node, 0, ((nodeCount + BS - 1) / BS * BS - nodeCount) * sizeof(vec2));
			cudaMemcpy(dev_node, mesh.dev_node, nodeCount * sizeof(vec2), cudaMemcpyDeviceToDevice);
			if (count3) {
				cudaMalloc((void**)&dev_elem3, 3 * count3 * sizeof(int));
				cudaMemcpy(dev_elem3, mesh.dev_elem3, 3 * count3 * sizeof(int), cudaMemcpyDeviceToDevice);
			}
			if (count4) {
				cudaMalloc((void**)&dev_elem4, 4 * count4 * sizeof(int));
				cudaMemcpy(dev_elem4, mesh.dev_elem4, 4 * count4 * sizeof(int), cudaMemcpyDeviceToDevice);
			}
			if (count8) {
				cudaMalloc((void**)&dev_elem8, 8 * count8 * sizeof(int));
				cudaMemcpy(dev_elem8, mesh.dev_elem8, 8 * count8 * sizeof(int), cudaMemcpyDeviceToDevice);
			}
			//dev_bordersCount = mesh.dev_bordersCount;
			if (!ramSaved) {
				//borderLength = new int[dev_bordersCount];
				//memcpy(borderLength, mesh.borderLength, dev_bordersCount * sizeof(int));
			}
			fillPosDev();
			//dev_borders = new int*[dev_bordersCount];
			//std::cout << "log3.5\n";
			//for (int i = 0; i < dev_bordersCount; ++i) {
			//	cudaMalloc((void**)&(dev_borders[i]), borderLength[i] * sizeof(int));
			//	cudaMemcpy(dev_borders[i], mesh.dev_borders[i], borderLength[i] * sizeof(int), cudaMemcpyDeviceToDevice);
			//}
			//std::cout << "log4\n";
		}
		
	}

	~Mesh() {
		delete[] node2;
		delete[] elem3;
		delete[] elem4;
		delete[] elem8;
		delete[] borderLength;
		for (int i = 0; i < bordersCount; ++i)
			delete[] borders[i];
		delete[] borders;
		delete[] spaces;
		delete[] aspects;
		delete[] skewAngles;
		if (useCuda) {
			cudaFree(dev_node);
			cudaFree(dev_elem3);
			cudaFree(dev_elem4);
			cudaFree(dev_elem8);
			//for (int i = 0; i < dev_bordersCount; ++i)
			//	cudaFree(dev_borders[i]);
		}

		//TODO: null sizes and pointers
		//TODO: delete secOrdNodes

		/*for (unsigned i = 0; i < elemTypes; ++i)
			elemInfo[i].~FiniteElement();*/
		//std::vector<int>::~vector()
	}

	int elemCount() const {
		return elemPos[2];
	}

	void printAnalysis();

	void genRectangle(double x1, double x2, double y1, double y2, size_t N1, size_t N2, int order = 1);

	void genRectWithHole(double x1, double x2, double y1, double y2, double holeRad, size_t N1, size_t N2, int order = 1, int enhance = 2);

	void genRing(double a, double b, size_t N_phi, size_t N_r, int order = 1, int enhance = 2);

	void genArc(double a, double b, double phi1, double phi2, size_t N_phi, size_t N_r, int order = 1, int enhance = 2);

	void renumerateRing(int borderN);

	void renumByDirection(vec2 direction = {1., 1.});

	void renumRCM();

	void setGeomType(GeomType geomType);

	void setGeomType(GeomType geomType, unsigned block);

	void setPlaneWithThickness(double h = 1.);

	void setPlaneWithThickness(std::function<double(const double*)> h);

	void setPlaneWithThickness(unsigned blockId, double h = 1.);

	void setPlaneWithThickness(unsigned blockId, std::function<double(const double*)> h);

	void fillBorderH();  // TEMP?

	void checkNodeAdjStruct() {
		if (nodeAdjStruct.empty())
			nodeAdjStruct.init(nodeCount, elemTypes, elemInfo, elem);
	}

	void checkColorMap() {
		if (colorMap.empty())
			colorMap.init(nodeCount, elemTypes, elemInfo, elem);
	}

	void smoothRing(int borderN);

	int findMaxIndexDiff() const;

	//Загрузить из файла
	bool loadFromFile(const std::string& fileName);

	//Сохранить в файл формата vtk
	void saveAsVtk(const std::string& fileName);

	//Сохранить в файл формата vtu
	void saveAsVtu(const std::string& fileName);

	void meshToRAM();

	void meshToGPU();

	//Вывод на экран
	void print();

	void printElemInfo() const;

	void printBorder() const;

	double elemSpace3(int e) const {
		double sum1 = 0., sum2 = 0.;
		for (int i = 3 * e + 1; i < 3 * (e + 1); ++i) {
			sum1 += node2[elem3[i - 1]].x * node2[elem3[i]].y;
			sum2 += node2[elem3[i - 1]].y * node2[elem3[i]].x;
		}
		sum1 += node2[elem3[3 * (e + 1) - 1]].x * node2[elem3[3 * e]].y;
		sum2 += node2[elem3[3 * (e + 1) - 1]].y * node2[elem3[3 * e]].x;
		return fabs(sum1 - sum2) * 0.5;
	}

	//Площадь элемента
	double elemSpace4(int e) const {
		double sum1 = 0., sum2 = 0.;
		for (int i = 4 * e + 1; i < 4 * (e + 1); ++i) {
			sum1 += node2[elem4[i - 1]].x * node2[elem4[i]].y;
			sum2 += node2[elem4[i - 1]].y * node2[elem4[i]].x;
		}
		sum1 += node2[elem4[4 * (e + 1) - 1]].x * node2[elem4[4 * e]].y;
		sum2 += node2[elem4[4 * (e + 1) - 1]].y * node2[elem4[4 * e]].x;
		return fabs(sum1 - sum2) * 0.5;
	}

	double elemSpace8(int e) const {
		double sum1 = 0., sum2 = 0.;
		for (int i = 8 * e; i < 8 * e + 3; ++i) {
			sum1 += node2[elem8[i]].x * node2[elem8[i + 4]].y + node2[elem8[i + 4]].x * node2[elem8[i + 1]].y;
			sum2 += node2[elem8[i]].y * node2[elem8[i + 4]].x + node2[elem8[i + 4]].y * node2[elem8[i + 1]].x;
		}
		sum1 += node2[elem8[8 * e + 3]].x * node2[elem8[8 * e + 7]].y + node2[elem8[8 * e + 7]].x * node2[elem8[8 * e]].y;
		sum2 += node2[elem8[8 * e + 3]].y * node2[elem8[8 * e + 7]].x + node2[elem8[8 * e + 7]].y * node2[elem8[8 * e]].x;
		return fabs(sum1 - sum2) * 0.5;
	}

	//Соотношение сторон
	double aspectRatio(int e) const {
		int begin = 4 * e;
		double minSide = (node2[elem4[begin]] - node2[elem4[begin + 1]]).norm();
		double maxSide = minSide;
		for (int i = 0; i < 4; ++i) {
			double length = (node2[elem4[begin + (i + 1) % 4]] - node2[elem4[begin + i]]).norm();
			if (length < minSide) minSide = length;
			else if (length > maxSide) maxSide = length;
		}
		return maxSide / minSide;
	}

	//Синус угла наклона
	double skewAngleSin(int e) const {
		int begin = 4 * e;
		if (4 == 4) {
			vec2 v1 = (node2[elem4[begin]] + node2[elem4[begin + 1]] - node2[elem4[begin + 2]] - node2[elem4[begin + 3]]) * 0.5;
			vec2 v2 = (node2[elem4[begin + 1]] + node2[elem4[begin + 2]] - node2[elem4[begin + 3]] - node2[elem4[begin]]) * 0.5;
			double scalMult = v1 * v2;
			return 1. - scalMult * scalMult / ((v1.x * v1.x + v1.y * v1.y) * (v2.x * v2.x + v2.y * v2.y));
		}
		else return 0.;
	}

	//Средний характерный размер элемента
	double avrElemSize() {
		if (!ramSaved)
			meshToRAM();
		double avrSize = 0.;
		for (size_t e = 0; e < count3; ++e)
			avrSize += elemSpace3(e);
		for (size_t e = 0; e < count4; ++e)
			avrSize += elemSpace4(e);
		for (size_t e = 0; e < count8; ++e)
			avrSize += elemSpace8(e);
		return avrSize / elemCount();
	}

};