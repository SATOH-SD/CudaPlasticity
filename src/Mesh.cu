#include "Mesh.h"

#include <filesystem>
#include <list>
#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <sstream>
#include <algorithm>
#include <omp.h>

#include "vec2.cuh"

#include "FiniteElement.h"


void Mesh::genRectangle(double x1, double x2, double y1, double y2, size_t N1, size_t N2, int order) {
	std::cout << "Mesh generation...";
	if (order == 2) {
		N1 *= 2; N2 *= 2;
	}
	nodeCount = (N1 + 1) * (N2 + 1);
	count4 = (N1 * N2);
	count3 = 0;
	count8 = 0;
	fillPos();
	
	//node.resize((N1 + 1) * (N2 + 1));
		
	delete[] node2;
	delete[] secOrdNodes;
	node2 = new vec2[nodeCount];
	secOrdNodes = new bool[nodeCount];
	for (int i = 0; i < nodeCount; ++i)
		secOrdNodes[i] = false;
	double h1 = (x2 - x1) / N1, h2 = (y2 - y1) / N2;
	for (int i = 0; i <= N1; ++i)
		for (int j = 0; j <= N2; ++j) {
			int k = i * (N2 + 1) + j;
			node2[k].x = x1 + h1 * i;
			node2[k].y = y1 + h2 * j;
		}
	delete[] elem3;
	delete[] elem4;
	delete[] elem8;
	elem3 = nullptr;
	elem8 = nullptr;
	elem4 = new int[count4 * 4];
	for (int i = 0; i < N1; ++i)
		for (int j = 0; j < N2; ++j) {
			int k = (i * N2) + j;

			int begin = k * 4;
			elem4[begin] = (i + 1) * (N2 + 1) + j + 1;
			elem4[begin + 1] = i * (N2 + 1) + j + 1;
			elem4[begin + 2] = i * (N2 + 1) + j;
			elem4[begin + 3] = (i + 1) * (N2 + 1) + j;
		}
	delete[] borderLength;
	for (int i = 0; i < bordersCount; ++i)
		delete[] borders[i];
	bordersCount = 0;
	delete[] borders;

	//borders.resize(4);
	//borders[0].resize(N2 + 1);
	//borders[1].resize(N1 + 1);
	//borders[2].resize(N2 + 1);
	//borders[3].resize(N1 + 1);
	bordersCount = 4;
	borders = new int*[bordersCount];
	borderLength = new int[bordersCount];
	borderLength[0] = borderLength[2] = N2 + 1;
	borderLength[1] = borderLength[3] = N1 + 1;
	for (int i = 0; i < bordersCount; ++i)
		borders[i] = new int[borderLength[i]];
	for (int i = 0; i <= N2; ++i) {
		borders[0][N2 - i] = (N1 + 1) * (N2 + 1) - i - 1;
		borders[2][N2 - i] = i;
	}
	for (int i = 0; i <= N1; ++i) {
		//borders[1][i] = (N1 - i) * (N2 + 1);
		//borders[3][i] = (i + 1) * (N2 + 1) - 1;
		borders[1][N1 - i] = (i + 1) * (N2 + 1) - 1;
		borders[3][N1 - i] = (N1 - i) * (N2 + 1);
	}
	if (order == 2) remapOrder(N2);

	ramSaved = true;
	analysed = false;
	if (useCuda) meshToGPU();

	translateFromLegacy();

	std::cout << "\rMesh generated: " << nodeCount << " nodes, " << elemCount() << " elements\n\n";
}


void Mesh::genRectWithHole(double x1, double x2, double y1, double y2, double holeRad, size_t N1, size_t N2, int order, int enhance) {
	const double pi = 3.141'592'653'589'793;
	const bool enhanced = true;
	std::cout << "Mesh generation...";
	int borderN = (N1 + N2) * 2;
	int radN = (N1 + N2) / 4;
	radN = round(double(radN) * (x2 - x1 + y2 - y1 - 4. * holeRad) * 2. / (x2 - x1 + y2 - y1));
	//std::cout << radN << "\n";

	double* steps = nullptr;
	if (enhance == 1) {
		double a = holeRad * pi, b = ((x2 - x1) + (y2 - y1));
		//double a = holeRad, b = sqrt((x2 - x1) * (y2 - y1)) / pi;
		radN = (size_t)round(borderN * (b - a) / (pi * (a + b)));
		if (order == 2) {
			N1 *= 2; N2 *= 2;
			borderN *= 2;
			radN *= 2;
		}
		//std::cout << (borderN * (b - a) / (pi * (a + b))) << "\n\n";
		steps = new double[radN + 1];
		double alpha = 2. / (double(radN) * (a + b));
		double a1 = alpha * a;
		double h = alpha * (b - a) / double(radN - 1);
		for (size_t i = 0; i <= radN; ++i)
			steps[i] = (2. * a1 + h * (double(i) - 1.)) * 0.5 * i;
	}
	else if (enhance == 0) {
		if (order == 2) {
			N1 *= 2; N2 *= 2;
			borderN *= 2;
			radN *= 2;
		}
		steps = new double[radN + 1];
		double h = 1. / double(radN);
		for (size_t i = 0; i <= radN; ++i)
			steps[i] = h * i;
	}
	else {
		double a = holeRad;
		//double b = ((x2 - x1) + (y2 - y1)) / pi;
		double b = 0.25 * (std::min((x2 - x1), (y2 - y1)) + vec2(x2 - x1, y2 - y1).norm());
		//double b = (((x2 - x1) + (y2 - y1)) / pi + 0.5 * std::min((x2 - x1), (y2 - y1))) * 0.5;
		//double b = 0.5 * std::min((x2 - x1), (y2 - y1));
		radN = (size_t)round(log(b / a) / log(1. + 2. * pi / double(borderN)));
		if (order == 2) {
			N1 *= 2; N2 *= 2;
			borderN *= 2;
			radN *= 2;
		}
		steps = new double[radN + 1];
		double q = pow(b / a, 1. / double(radN));
		double coef = 1.;
		for (size_t i = 0; i <= radN; ++i, coef *= q)
			steps[i] = (a * coef - a) / (b - a);
	}
	//for (size_t i = 0; i <= radN; ++i) \
			std::cout << steps[i] << "\n";

	delete[] borderLength;
	for (int i = 0; i < bordersCount; ++i)
		delete[] borders[i];
	bordersCount = 0;
	delete[] borders;

	nodeCount = borderN * (radN + 1);
	delete[] node2;
	delete[] secOrdNodes;
	node2 = new vec2[nodeCount];
	secOrdNodes = new bool[nodeCount];
	for (int i = 0; i < nodeCount; ++i)
		secOrdNodes[i] = false;
	bordersCount = 5;
	borders = new int*[bordersCount];
	borderLength = new int[bordersCount];
	borderLength[0] = borderLength[2] = N1 + 1;
	borderLength[1] = borderLength[3] = N2 + 1;
	borderLength[4] = borderN + 1;
	for (int i = 0; i < bordersCount; ++i)
		borders[i] = new int[borderLength[i]];

	int nodeId = 0;
	double hx = (x2 - x1) / N1, hy = (y2 - y1) / N2;
	for (int i = 0; i < N1; ++i, ++nodeId) {
		node2[nodeId] = { x1 + i * hx, y1 };
		borders[0][i] = nodeId;
	}
	borders[0][borderLength[0] - 1] = nodeId;
	for (int j = 0; j < N2; ++j, ++nodeId) {
		node2[nodeId] = { x2, y1 + j * hy };
		borders[1][j] = nodeId;
	}
	borders[1][borderLength[1] - 1] = nodeId;
	for (int i = 0; i < N1; ++i, ++nodeId) {
		node2[nodeId] = { x2 - i * hx, y2 };
		borders[2][i] = nodeId;
	}
	borders[2][borderLength[2] - 1] = nodeId;
	for (int j = 0; j < N2; ++j, ++nodeId) {
		node2[nodeId] = { x1, y2 - j * hy };
		borders[3][j] = nodeId;
	}
	borders[3][borderLength[3] - 1] = 0;

	//const double pi = 3.141'592'653'589'793;
	double phi = 2. * pi / borderN;
	nodeId = radN * borderN;
	vec2 center(0.5 * (x1 + x2), 0.5 * (y1 + y2));
	double start = phi * (N1 + N2 * 1.5);
	for (size_t i = 0; i < borderN; ++i, ++nodeId) {
		node2[nodeId] = center + vec2(cos(start + i * phi), sin(start + i * phi)) * holeRad;
		borders[4][borderN - i - 1] = nodeId;
	}
	borders[4][borderLength[4] - 1] = nodeCount - 1;

	//vec2 rh = (node[radN * borderN] - node[0]) / radN;
	//for (int i = 1; i < radN; ++i) //rad - 1?
	//	node[i * borderN] = node[0] + rh * i;
	vec2 len = (node2[radN * borderN] - node2[0]);
	for (size_t i = 1; i < radN; ++i)
		node2[i * borderN] = node2[0] + len * (1. - steps[radN - i]);

	/*rh = (node[nodeCount - 1] - node[borderN - 1]) / radN;
	for (int i = 1; i < radN; ++i)
		node[i * borderN + borderN - 1] = node[borderN - 1] + rh * i;*/
	for (size_t j = 1; j < borderN; ++j) {
		/*vec2 rh = (node[radN * borderN + j] - node[j]) / radN;
		for (size_t i = 1; i < radN; ++i)
			node[i * borderN + j] = node[j] + rh * i;*/
		vec2 len = (node2[radN * borderN + j] - node2[j]);
		for (size_t i = 1; i < radN; ++i)
			node2[i * borderN + j] = node2[j] + len * (1. - steps[radN - i]);
	}

	count4 = borderN * radN;
	count3 = 0;
	count8 = 0;
	delete[] elem3;
	delete[] elem4;
	delete[] elem8;
	elem3 = nullptr;
	elem8 = nullptr;
	elem4 = new int[count4 * 4];
	fillPos();
	for (int i = 0; i < radN; ++i)
		for (int j = 0; j < borderN - 1; ++j) {
			//int k = i * borderN + j;
			int k = j * radN + i;
			int begin = k * 4;
			elem4[begin + 1] = (i + 1) * borderN + j;
			elem4[begin + 2] = i * borderN + j;
			elem4[begin + 3] = i * borderN + j + 1;
			elem4[begin + 0] = (i + 1) * borderN + j + 1;
		}
	for (int i = 0; i < radN; ++i) {
		//int k = i * borderN + borderN - 1;
		int k = (borderN - 1) * radN + i;
		int begin = k * 4;
		elem4[begin + 1] = (i + 1) * borderN + borderN - 1;
		elem4[begin + 2] = i * borderN + borderN - 1;
		elem4[begin + 3] = i * borderN;
		elem4[begin + 0] = (i + 1) * borderN;
	}
	delete[] steps;

	ramSaved = true;
	analysed = false;
	//useCuda = false;
	//smoothRing(borderN);
	renumerateRing(borderN);
	if (order == 2) remapOrder(radN);

	if (useCuda) {
		meshToGPU();
		//meshToGPU();
		
		/*cudaFree(dev_elem3);
		cudaFree(dev_elem4);
		cudaFree(dev_elem8);
		cudaMalloc((void**)&dev_elem3, 3 * count3 * sizeof(int));
		cudaMalloc((void**)&dev_elem4, 4 * count4 * sizeof(int));
		cudaMalloc((void**)&dev_elem8, 8 * count8 * sizeof(int));
		cudaMemcpy(dev_elem3, elem3, 3 * count3 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_elem4, elem4, 4 * count4 * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_elem8, elem8, 8 * count8 * sizeof(int), cudaMemcpyHostToDevice);*/
	}

	translateFromLegacy();

	std::cout << "\rMesh generated: " << nodeCount << " nodes, " << elemCount() << " elements\n\n";
	//printAnalysis();
}


//Сгенерировать сетку для кольца с внутренним радиусом а и внешним радиусом b
void Mesh::genRing(double a, double b, size_t N_phi, size_t N_r, int order, int enhance) {
	const double pi = 3.141'592'653'589'793;
	const bool enhanced = true;
	std::cout << "Mesh generation...";

	double* steps = nullptr;
	if (enhance == 1) {
		//std::cout << N_r << "\n";
		N_r = (size_t)round(N_phi * (b - a) / (pi * (a + b)));
		//N_r = (size_t)round(N_phi * (b - a) / (2 * pi * (a + b))) + 20;
		//std::cout << (N_phi * (b - a) / (pi * (a + b))) << "\n\n";
		if (order == 2) {
			N_phi *= 2; N_r *= 2;
		}
		//N_r *= 2;
		steps = new double[N_r + 1];
		double alpha = 2. / (double(N_r) * (a + b));
		//double alpha = 2. * pi / (double(N_phi) * (b - a));
		double a1 = alpha * a;
		double h = alpha * (b - a) / double(N_r - 1);
		for (size_t i = 0; i <= N_r; ++i)
			steps[i] = a + (b - a) * ((2. * a1 + h * (double(i) - 1.)) * 0.5 * i);
		//for (size_t i = 0; i <= N_r; ++i) \
			std::cout << (steps[i] - a) / (b - a) << "\n";
	}
	else if (enhance == 0) {
		if (order == 2) {
			N_phi *= 2; N_r *= 2;
		}
		steps = new double[N_r + 1];
		double h = (b - a) / double(N_r);
		for (size_t i = 0; i <= N_r; ++i)
			steps[i] = a + h * i;
	}
	else {
		//N_r = (size_t)round(log(b / a) / log(1. + 2. * (b - a) / double(N_phi)));
		N_r = (size_t)round(log(b / a) / log(1. + 2. * pi / double(N_phi)));
		//N_r = (size_t)round(log(b / a) / log((N_phi * (b - a) + pi *  (6. * b - 4. * a)) / (2. * pi * b + N_phi * (b - a))));
		//std::cout << log(b / a) / log(1. + 4. * pi / double(N_phi)) << "\n";
		//std::cout << log(b / a) / log((N_phi * (b - a) + pi * (6. * b - 4. * a)) / (2. * pi * b + N_phi * (b - a))) << "\n";
		if (order == 2) {
			N_phi *= 2; N_r *= 2;
		}
		steps = new double[N_r + 1];
		double q = pow(b / a, 1. / double(N_r));
		//std::cout << q << "\n";
		double coef = 1.;
		for (size_t i = 0; i <= N_r; ++i, coef *= q) {
			steps[i] = a * coef;
			//std::cout << steps[i] << "\n";
		}
		//for (size_t i = 0; i <= N_r; ++i) {
			//steps[i] = (steps[i] - a) / (b - a);
			//std::cout << steps[i] << "\n";
		//}
	}

	delete[] borderLength;
	for (int i = 0; i < bordersCount; ++i)
		delete[] borders[i];
	bordersCount = 0;
	delete[] borders;

	nodeCount = (N_r + 1) * N_phi;
	delete[] node2;
	delete[] secOrdNodes;
	node2 = new vec2[nodeCount];
	secOrdNodes = new bool[nodeCount];
	for (int i = 0; i < nodeCount; ++i)
		secOrdNodes[i] = false;
	bordersCount = 2;
	borders = new int* [bordersCount];
	borderLength = new int[bordersCount];
	borderLength[0] = N_phi + 1;
	borderLength[1] = N_phi + 1;
	for (int i = 0; i < bordersCount; ++i)
		borders[i] = new int[borderLength[i]];

	double h_r = (b - a) / N_r, h_phi = 2. * pi / N_phi;

	for (int j = 0; j < N_phi; ++j) {
		for (int i = 0; i <= N_r; ++i) {
			//double r = b - h_r * i;
			double r = steps[N_r - i];

			double phi = h_phi * j;
			node2[i * N_phi + j] = vec2(r * cos(phi), r * sin(phi));
		}
		borders[1][N_phi - j] = j;
		borders[0][j] = N_r * N_phi + j;
	}
	borders[1][0] = 0;
	borders[0][borderLength[1] - 1] = N_r * N_phi;
	count4 = N_r * N_phi;
	count3 = 0;
	count8 = 0;
	delete[] elem3;
	delete[] elem4;
	delete[] elem8;
	elem3 = nullptr;
	elem8 = nullptr;
	elem4 = new int[count4 * 4];
	fillPos();
	for (int i = 0; i < N_r; ++i)
		for (int j = 0; j < N_phi - 1; ++j) {
			//int k = (i * N_phi) + j;
			int k = (j * N_r) + i;
			int begin = 4 * k;
			elem4[begin] = (i + 1) * N_phi + j + 1;
			elem4[begin + 3] = i * N_phi + j + 1;
			elem4[begin + 2] = i * N_phi + j;
			elem4[begin + 1] = (i + 1) * N_phi + j;
		}
	for (int i = 0; i < N_r; ++i) {
		//int k = i * N_phi + N_phi - 1;
		int k = (N_phi - 1) * N_r + i;
		int begin = 4 * k;
		elem4[begin] = (i + 1) * N_phi;
		elem4[begin + 3] = i * N_phi;
		elem4[begin + 2] = i * N_phi + N_phi - 1;
		elem4[begin + 1] = (i + 1) * N_phi + N_phi - 1;
	}
	delete[] steps;

	ramSaved = true;
	analysed = false;
	//useCuda = false;
	//smoothRing(borderN);
	renumerateRing(N_phi);
	if (order == 2) remapOrder(N_r);

	if (useCuda) meshToGPU();

	translateFromLegacy();

	std::cout << "\rMesh generated: " << nodeCount << " nodes, " << elemCount() << " elements\n\n";
	//printAnalysis();
}


void Mesh::genArc(double a, double b, double phi1, double phi2, size_t N_phi, size_t N_r, int order, int enhance) {
	const double pi = 3.141'592'653'589'793;
	const bool enhanced = true;
	std::cout << "Mesh generation...";

	double* steps = nullptr;
	if (enhance == 1) {
		N_r = (size_t)round(N_phi * (b - a) / (0.5 * (phi2 - phi1) * (a + b)));
		if (order == 2) {
			N_phi *= 2; N_r *= 2;
		}
		steps = new double[N_r + 1];
		double alpha = 2. / (double(N_r) * (a + b));
		double a1 = alpha * a;
		double h = alpha * (b - a) / double(N_r - 1);
		for (size_t i = 0; i <= N_r; ++i)
			steps[i] = a + (b - a) * ((2. * a1 + h * (double(i) - 1.)) * 0.5 * i);
	}
	else if (enhance == 0) {
		if (order == 2) {
			N_phi *= 2; N_r *= 2;
		}
		steps = new double[N_r + 1];
		double h = (b - a) / double(N_r);
		for (size_t i = 0; i <= N_r; ++i)
			steps[i] = a + h * i;
	}
	else {
		N_r = (size_t)round(log(b / a) / log(1. + (phi2 - phi1) / double(N_phi)));
		if (order == 2) {
			N_phi *= 2; N_r *= 2;
		}
		steps = new double[N_r + 1];
		double q = pow(b / a, 1. / double(N_r));
		double coef = 1.;
		for (size_t i = 0; i <= N_r; ++i, coef *= q)
			steps[i] = a * coef;
	}

	delete[] borderLength;
	for (int i = 0; i < bordersCount; ++i)
		delete[] borders[i];
	bordersCount = 0;
	delete[] borders;

	nodeCount = (N_r + 1) * (N_phi + 1);
	delete[] node2;
	delete[] secOrdNodes;
	node2 = new vec2[nodeCount];
	secOrdNodes = new bool[nodeCount];
	for (int i = 0; i < nodeCount; ++i)
		secOrdNodes[i] = false;
	bordersCount = 4;
	borders = new int* [bordersCount];
	borderLength = new int[bordersCount];
	borderLength[0] = borderLength[2] = N_r + 1;
	borderLength[1] = borderLength[3] = N_phi + 1;
	for (int i = 0; i < bordersCount; ++i)
		borders[i] = new int[borderLength[i]];
	for (int i = 0; i <= N_r; ++i) {
		borders[0][N_r - i] = (N_phi + 1) * (N_r + 1) - i - 1;
		borders[2][N_r - i] = i;
	}
	for (int i = 0; i <= N_phi; ++i) {
		//borders[1][i] = (N_phi - i) * (N_r + 1);
		//borders[3][i] = (i + 1) * (N_r + 1) - 1;
		borders[1][N_phi - i] = (i + 1) * (N_r + 1) - 1;
		borders[3][N_phi - i] = (N_phi - i) * (N_r + 1);
	}

	double h_r = (b - a) / N_r, h_phi = (phi2 - phi1) / N_phi;

	for (int j = 0; j <= N_r; ++j) {
		for (int i = 0; i <= N_phi; ++i) {
			//double r = b - h_r * i;
			double r = steps[N_r - j];

			double phi = phi1 + h_phi * i;
			node2[i * (N_r + 1) + j] = vec2(r * cos(phi), r * sin(phi));
		}
	}
	count4 = N_r * N_phi;
	count3 = 0;
	count8 = 0;
	delete[] elem3;
	delete[] elem4;
	delete[] elem8;
	elem3 = nullptr;
	elem8 = nullptr;
	elem4 = new int[count4 * 4];
	fillPos();
	for (int i = 0; i < N_phi; ++i)
		for (int j = 0; j < N_r; ++j) {
			int k = (i * N_r) + j;
			//int k = (j * N_r) + i;

			int begin = k * 4;
			elem4[begin] = (i + 1) * (N_r + 1) + j + 1;
			elem4[begin + 1] = i * (N_r + 1) + j + 1;
			elem4[begin + 2] = i * (N_r + 1) + j;
			elem4[begin + 3] = (i + 1) * (N_r + 1) + j;
		}
	delete[] steps;

	ramSaved = true;
	analysed = false;
	//useCuda = false;
	//smoothRing(borderN);
	if (order == 2) remapOrder(N_r);

	if (useCuda) meshToGPU();

	translateFromLegacy();

	std::cout << "\rMesh generated: " << nodeCount << " nodes, " << elemCount() << " elements\n\n";
}


void Mesh::renumerateRing(int borderN) {
	std::vector<int> newNodes(nodeCount);
	int newN = 0, radN = nodeCount / borderN,
		iBack = radN - 1, iFront = 0,
		jBack = borderN - 1, jFront = 0;
	int i = 0;

	while (jBack > jFront) {
		for (int i = 0; i < radN; ++i)
			newNodes[i * borderN + jFront] = newN++;
		++jFront;
		for (int i = 0; i < radN; ++i)
			newNodes[i * borderN + jBack] = newN++;
		--jBack;
	}
	if (jBack == jFront)
		for (int i = 0; i < radN; ++i)
			newNodes[i* borderN + jFront] = newN++;

	for (int i = 0; i < count4; ++i)
		for (int j = 4 * i; j < 4 * (i + 1); ++j)
			elem4[j] = newNodes[elem4[j]];
	for (int i = 0; i < bordersCount; ++i)
		for (int j = 0; j < borderLength[i]; ++j)
			borders[i][j] = newNodes[borders[i][j]];
	std::vector<vec2> nodes(nodeCount);
	for (size_t i = 0; i < nodeCount; ++i)
		nodes[newNodes[i]] = node2[i];
	memcpy(node2, nodes.data(), nodeCount * sizeof(vec2));
}

void Mesh::smoothRing(int borderN) {
	const double pi = 3.141'592'653'589'793;
	int radN = nodeCount / borderN;
	double d_eta_2 = 1. / (borderN * borderN),
		d_xi_2 = 1. / (radN * radN);
	double d_eta = 1. / borderN,
		d_xi = 1. / radN;
	double d_diag_2 = d_xi_2 + d_eta_2;
	for (int i = 0; i < 1; ++i) {
		for (int i = 1; i < radN - 1; ++i) {
			//node[i] = 0.25 * (d_xi_2 * (node[])
		}
		//for (int i = 1; i < radN - 1; ++i)
		for (int i = radN - 2; i > 0; --i)
			for (int j = 1; j < borderN - 1; ++j) {
				//node[i * borderN + j] = 0.5 * (d_xi_2 * (node[i * borderN + j + 1] + node[i * borderN + j - 1]) + \
					d_eta_2 * (node[(i + 1) * borderN + j] + node[(i - 1) * borderN + j])) / (d_xi_2 + d_eta_2);
				//node[i * borderN + j] = 0.25 * (node[(i + 1) * borderN + j + 1] + node[(i + 1) * borderN + j - 1] + \
					node[(i - 1) * borderN + j + 1] + node[(i - 1) * borderN + j - 1]);
				//node[i * borderN + j] = (d_eta * (3. * node[(i - 1) * borderN + j] + node[(i + 1) * borderN + j]) - \
					d_xi * (3. * node[i * borderN + j - 1] + node[i * borderN + j + 1])) * 0.25 / (d_eta - d_xi);
				double ddy = 0.5 * ((node2[(i + 1) * borderN + j] - node2[i * borderN + j]).norm() + (node2[(i - 1) * borderN + j] - node2[i * borderN + j]).norm()),
					ddx = 0.5 * ((node2[i * borderN + j + 1] - node2[i * borderN + j]).norm() + (node2[i * borderN + j - 1] - node2[i * borderN + j]).norm());
				//node[i * borderN + j] = (node[(i + 1) * borderN + j] * ddx + node[(i - 1) * borderN + j] * ddy) / (ddx + ddy);
				//node[i * borderN + j] = node[(i + 1) * borderN + j] - (node[(i + 1) * borderN + j] - node[i * borderN + j]).normalize() * ddx;
				node2[i * borderN + j] = node2[(i + 1) * borderN + j] - \
					(node2[(i + 1) * borderN + j] - node2[i * borderN + j]).normalize() * 2. * pi * node2[(i + 1) * borderN + j].norm() / borderN;
			}
	}
}

int Mesh::findMaxIndexDiff() const { //TO DO: parallel
	int max = 0;
	for (int e = 0; e < count3; ++e)
		for (int i = 3 * e; i < 3 * (e + 1); ++i)
			for (int j = i + 1; j < 3 * (e + 1); ++j) {
				int diff = abs(elem3[i] - elem3[j]);
				if (diff > max)
					max = diff;
			}
	for (int e = 0; e < count4; ++e)
		for (int i = 4 * e; i < 4 * (e + 1); ++i)
			for (int j = i + 1; j < 4 * (e + 1); ++j) {
				int diff = abs(elem4[i] - elem4[j]);
				if (diff > max) {
					max = diff;
					//std::cout << max << ": " << elem4[i] << " " << elem4[j] << "\n";
				}
			}
	for (int e = 0; e < count8; ++e)
		for (int i = 8 * e; i < 8 * (e + 1); ++i)
			for (int j = i + 1; j < 8 * (e + 1); ++j) {
				int diff = abs(elem8[i] - elem8[j]);
				if (diff > max)
					max = diff;
			}
	return max;
}

void Mesh::remapOrder(int width) {
	count8 = count4 / 4;
	delete[] elem8;
	elem8 = new int[8 * count8];
	bool* remNodes = new bool[nodeCount];
	for (int i = 0; i < nodeCount; ++i)
		remNodes[i] = true;
	
	for (int i = 0; i < count4 / width; i += 2)
		for (int j = 0; j < width; j += 2) {
			int begin = 8 * (i * width / 4 + j / 2);
			elem8[begin] = elem4[4 * ((i + 1) * width + j + 1)];
			elem8[begin + 1] = elem4[4 * ((i) * width + j + 1) + 1];
			elem8[begin + 2] = elem4[4 * ((i) * width + j) + 2];
			elem8[begin + 3] = elem4[4 * ((i + 1) * width + j) + 3];
			elem8[begin + 4] = elem4[4 * ((i + 1) * width + j + 1) + 1];
			elem8[begin + 5] = elem4[4 * ((i) * width + j + 1) + 2];
			elem8[begin + 6] = elem4[4 * ((i) * width + j) + 3];
			elem8[begin + 7] = elem4[4 * ((i + 1) * width + j)];
			remNodes[elem4[4 * ((i) * width + j)]] = false;
		}
	/*for (int i = 0; i < nodeCount; ++i)
		if (!remNodes[i])
			std::cout << i << "\n";*/
	int* newNodes = new int[nodeCount];
	for (int iOld = 0, iNew = 0; iOld < nodeCount; ++iOld)
		if (remNodes[iOld]) {
			newNodes[iOld] = iNew++;
		}
	/*for (int i = 0; i < nodeCount; ++i)
		std::cout << i << " -> " << newNodes[i] << "\n";*/

	for (int i = 0; i < 8 * count8; ++i)
		elem8[i] = newNodes[elem8[i]];
	for (int i = 0; i < bordersCount; ++i)
		for (int j = 0; j < borderLength[i]; ++j)
			borders[i][j] = newNodes[borders[i][j]];
	
	vec2* nodes = new vec2[nodeCount - count8];
	for (size_t i = 0; i < nodeCount; ++i)
		if (remNodes[i])
			nodes[newNodes[i]] = node2[i];
	nodeCount -= count8;
	delete[] node2;
	node2 = nodes;

	delete[] secOrdNodes;
	secOrdNodes = new bool[nodeCount];
	for (int i = 0; i < nodeCount; ++i)
		secOrdNodes[i] = false;
	for (int e = 0; e < count8; ++e)
		for (int i = 4; i < 8; ++i)
			secOrdNodes[elem8[8 * e + i]] = true;

	delete[] remNodes;
	delete[] newNodes;
	delete[] elem4;
	elem4 = nullptr;
	count4 = 0;
	fillPos();
}

static void renumSort(double* rBegin, double* rEnd, int* nBegin, double* bufferR, int* bufferN) {
	int dist = rEnd - rBegin;
	//std::cout << dist << "\n";
	if (dist != 1) {
		double* rMiddle = rBegin + dist / 2;
		renumSort(rBegin, rMiddle, nBegin, bufferR, bufferN);
		int half = rMiddle - rBegin;
		renumSort(rMiddle, rEnd, nBegin + half, bufferR + half, bufferN + half);
		//merge
		double* ir1 = rBegin, * ir2 = rMiddle, * ibr = bufferR;
		int* in1 = nBegin, * in2 = nBegin + half, * ibn = bufferN;
		while (ir1 != rMiddle && ir2 != rEnd) {
			if (*ir1 < *ir2) {
				*ibr++ = *ir1++;
				*ibn++ = *in1++;
			}
			else {
				*ibr++ = *ir2++;
				*ibn++ = *in2++;
			}
		}
		//copy rest to buffers
		memcpy(ibr, ir1, (rMiddle - ir1) * sizeof(double));
		memcpy(ibr, ir2, (rEnd - ir2) * sizeof(double));
		memcpy(ibn, in1, (rMiddle - ir1) * sizeof(int));
		memcpy(ibn, in2, (rEnd - ir2) * sizeof(int));
		//copy from buffers
		memcpy(rBegin, bufferR, (rEnd - rBegin) * sizeof(double));
		memcpy(nBegin, bufferN, (rEnd - rBegin) * sizeof(int));
	}
}

void Mesh::renumByDirection(vec2 direction) {
	if (!ramSaved) meshToRAM();
	/*int keyNode = 0;
	vec2 r = node[keyNode];
	for (int i = 1; i < nodeCount; ++i) {
		if (r.x <= node[i].x && r.y <= node[i].y) {
			keyNode = i;
			r = node[i];
		}
	}*/
	int* newNodes = new int[nodeCount];
	int* oldNodes = new int[nodeCount];
	int* bufferN = new int[nodeCount];
	double* ranges = new double[nodeCount];
	double* bufferR = new double[nodeCount];

	for (int i = 0; i < nodeCount; ++i) {
		oldNodes[i] = i;
		//ranges[i] = (node[i] - node[keyNode]).norm();
		ranges[i] = node2[i] * direction;
	}
	
	/*int threads = omp_get_max_threads();
	int* displs = new int[threads + 1];
	displs[0] = 0;
	for (int i = 0; i < threads; ++i)
		displs[i + 1] = displs[i] + nodeCount / threads + ((i < nodeCount % threads) ? 1 : 0);*/

	renumSort(ranges, ranges + nodeCount, oldNodes, bufferR, bufferN);

//#pragma omp parallel
//	{
//		int th = omp_get_thread_num();
//		renumSort(ranges + displs[th], ranges + displs[th + 1], oldNodes + displs[th], bufferR + displs[th], bufferN + displs[th]);
//	}
//	for (int i = threads; i > 2; i /= 2) {
//#pragma omp parallel for
//		for (int j = 0; j < i; j += 2) {
//			double* rBegin = ranges + displs[j],
//				* rMiddle = ranges + displs[j + 1],
//				* rEnd = ranges + displs[j + 2];
//			double* ir1 = rBegin, * ir2 = rMiddle, * ibr = bufferR;
//			int* nBegin = oldNodes + displs[j];
//			int* in1 = nBegin, * in2 = nBegin + displs[j + 1] - displs[j], * ibn = bufferN;
//			while (ir1 != rMiddle && ir2 != rEnd) {
//				if (*ir1 < *ir2) {
//					*ibr++ = *ir1++;
//					*ibn++ = *in1++;
//				}
//				else {
//					*ibr++ = *ir2++;
//					*ibn++ = *in2++;
//				}
//			}
//			//copy rest to buffers
//			memcpy(ibr, ir1, (rMiddle - ir1) * sizeof(double));
//			memcpy(ibr, ir2, (rEnd - ir2) * sizeof(double));
//			memcpy(ibn, in1, (rMiddle - ir1) * sizeof(int));
//			memcpy(ibn, in2, (rEnd - ir2) * sizeof(int));
//			//copy from buffers
//			memcpy(rBegin, bufferR, (rEnd - rBegin) * sizeof(double));
//			memcpy(nBegin, bufferN, (rEnd - rBegin) * sizeof(int));
//		}
//		for (int j = 0; j <= i / 2; ++j)
//			displs[j] = displs[j * 2];
//	}

	//for (int i = 0; i < nodeCount; ++i) \
		std::cout << ranges[i] << "\n";
	for (int i = 0; i < nodeCount; ++i)
		newNodes[oldNodes[i]] = i;

	//DEBUG
	/*std::cout << nodeCount << " nodes\n";
	for (int i = 0; i < nodeCount; ++i) {
		if (newNodes[i] <= 0 || newNodes[i] >= nodeCount)
			std::cout << i << " - " << newNodes[i] << "\n";
	}*/

	for (int i = 0; i < 3 * count3; ++i)
		elem3[i] = newNodes[elem3[i]];
	for (int i = 0; i < 4 * count4; ++i) {
		//int old = elem4[i];
		elem4[i] = newNodes[elem4[i]];
		/*if (elem4[i] <= 0 || elem4[i] > 30000)
			std::cout << i / 4 << " - " << old << " " << elem4[i] << "\n";*/
	}
	/*for (int i = 0; i < 4 * count4; ++i) {
		
	}*/
	for (int i = 0; i < 8 * count8; ++i)
		elem8[i] = newNodes[elem8[i]];
	for (int i = 0; i < bordersCount; ++i)
		for (int j = 0; j < borderLength[i]; ++j)
			borders[i][j] = newNodes[borders[i][j]];
	vec2* nodes = new vec2[nodeCount];
	for (size_t i = 0; i < nodeCount; ++i)
		nodes[newNodes[i]] = node2[i];
	memcpy(node2, nodes, nodeCount * sizeof(vec2));
	for (int i = 0; i < nodeCount; ++i)
		secOrdNodes[i] = false;
	for (int e = 0; e < count8; ++e)
		for (int i = 4; i < 8; ++i)
			secOrdNodes[elem8[8 * e + i]] = true;

	delete[] newNodes;
	delete[] oldNodes;
	delete[] bufferN;
	delete[] ranges;
	delete[] bufferR;
	delete[] nodes;
	//delete[] displs;

	if (useCuda) meshToGPU();

	translateFromLegacy();

	std::cout << "Nodes renumbered\n";
	/*if (count4) {
		std::cout << "\nElements (4 nodes): " << count4 << "\n";
		for (size_t i = 0; i < count4; ++i) {
			for (size_t j = 4 * i; j < 4 * (i + 1); ++j)
				std::cout << elem4[j] << " ";
			std::cout << "\n";
		}
	}*/
}


using RootStruct = std::deque<std::deque<unsigned>>;  //список уровней


// Построение корневой структуры от базового узла
static void initRootStruct(RootStruct& lvl, unsigned nodeCount, unsigned* adj, unsigned* adjIdx, unsigned baseNode) {
	std::vector<bool> mask(nodeCount, true);
	lvl.push_back({ baseNode });
	//std::cout << "baseNode " << baseNode << "\n";
	mask[baseNode] = false;
	bool notAll = false;
	do {
		std::deque<unsigned> newLvl;
		for (unsigned cell : lvl.back())
			for (unsigned i = adjIdx[cell]; i < adjIdx[cell + 1]; ++i) {
				//std::cout << i << " ";
				unsigned cmi = adj[i];
				if (mask[cmi]) {
					newLvl.push_back(adj[i]);
					mask[adj[i]] = false;
				}
			}
		lvl.push_back(newLvl);
		notAll = false;
		for (bool m : mask)
			if (m) {
				notAll = true;
				break;
			}
	} while (notAll);
}


// Поиск псевдопереферийного узла и перестроение корневой структуры от него
static unsigned findRoot(RootStruct& lvl, unsigned nodeCount, unsigned* adj, unsigned* adjIdx) {
	unsigned y = lvl.front().front(), max = lvl.size();
	for (;;) {
		unsigned min = SIZE_MAX;
		unsigned imin = 0;
		for (int newY : lvl.back()) {
			unsigned locMin = adjIdx[newY + 1] - adjIdx[newY];
			if (locMin < min) {
				min = locMin;
				imin = newY;
			}
		}
		RootStruct newRS;
		initRootStruct(newRS, nodeCount, adj, adjIdx, imin);
		if (newRS.size() > lvl.size())
			newRS.swap(lvl);
		else
			break;
	}
	return lvl.front().front();
}


void Mesh::renumRCM() {
	checkNodeAdjStruct();

	RootStruct rs;
	initRootStruct(rs, nodeCount, nodeAdjStruct.adj, nodeAdjStruct.adjIdx, 0);
	findRoot(rs, nodeCount, nodeAdjStruct.adj, nodeAdjStruct.adjIdx);

	std::vector<size_t> degs(nodeCount);  // степени
	for (size_t i = 0; i < degs.size(); ++i)
		degs[i] = nodeAdjStruct.adjIdx[i + 1] - nodeAdjStruct.adjIdx[i];

	// исключение учёта связей с предыдущими уровнями (не обязательно и очень медленно)
	/*auto cur = rs.begin();
	for (auto prev = cur++; cur != rs.end(); ++cur, ++prev)
		for (auto node0 : *cur)
			for (int i = nodeAdjStruct.adjIdx[node0]; i < nodeAdjStruct.adjIdx[node0 + 1]; ++i)
				for (int node1 : *prev)
					if (nodeAdjStruct.adj[i] == node1)
						--degs[node0];*/

	const unsigned ui32max = ~unsigned(0);
	std::vector<unsigned> newNodes(nodeCount, ui32max);

	unsigned newN = 0;
	newNodes[rs.front().front()] = newN++;
	for (auto lv = rs.begin(); lv != rs.end(); ++lv) {
		std::vector<unsigned> nextNodes;
		nextNodes.reserve(20);
		std::vector<unsigned> lvl(lv->size());
		std::copy(lv->begin(), lv->end(), lvl.begin());

		// отсортировать по возрастанию уже перенумерованных
		std::sort(lvl.begin(), lvl.end(), [&](unsigned a1, unsigned a2) { return newNodes[a1] < newNodes[a2]; });

		for (unsigned cell : lvl) {
			nextNodes.clear();
			for (unsigned i = nodeAdjStruct.adjIdx[cell]; i < nodeAdjStruct.adjIdx[cell + 1]; ++i)
				if (newNodes[nodeAdjStruct.adj[i]] == ui32max)
					nextNodes.push_back(nodeAdjStruct.adj[i]);

			// отсортировать по возрастанию степеней
			std::sort(nextNodes.begin(), nextNodes.end(), [&](unsigned a1, unsigned a2) { return degs[a1] < degs[a2]; });

			//расставить номера
			for (unsigned i = 0; i < nextNodes.size(); ++i)
				newNodes[nextNodes[i]] = newN++;
		}
	}

	for (unsigned i = 0; i < nodeCount; ++i)      // обращение нумерации
		newNodes[i] = nodeCount - newNodes[i] - 1;

	FiniteElement& lastElem = elemInfo[elemTypes - 1];
	for (unsigned i = 0; i < lastElem.memIdx + lastElem.elemCount * lastElem.nodeCount; ++i)
		elem[i] = newNodes[elem[i]];

	hptr<double> nodes(dim * nodeCount);
	for (unsigned i = 0; i < nodeCount; ++i)
		memcpy(nodes + dim * newNodes[i], node + dim * i, dim * sizeof(double));
	memcpy(node, nodes, dim * nodeCount * sizeof(double));

	// TEMP
	memcpy(node2, nodes, nodeCount * sizeof(vec2));
	for (int i = 0; i < 3 * count3; ++i)
		elem3[i] = newNodes[elem3[i]];
	for (int i = 0; i < 4 * count4; ++i)
		elem4[i] = newNodes[elem4[i]];
	for (int i = 0; i < 8 * count8; ++i)
		elem8[i] = newNodes[elem8[i]];
	for (int i = 0; i < bordersCount; ++i)
		for (int j = 0; j < borderLength[i]; ++j)
			borders[i][j] = newNodes[borders[i][j]];
	for (int i = 0; i < nodeCount; ++i)
		secOrdNodes[i] = false;
	for (int e = 0; e < count8; ++e)
		for (int i = 4; i < 8; ++i)
			secOrdNodes[elem8[8 * e + i]] = true;
	if (useCuda) meshToGPU();
	// TEMP

	hptr<unsigned> oldNodes(nodeCount);
	for (size_t i = 0; i < nodeCount; ++i)
		oldNodes[newNodes[i]] = i;

	hptr<unsigned> newAdj(nodeAdjStruct.adjIdx[nodeCount]);
	hptr<unsigned> newAdjIdx(nodeCount + 1);
	newAdjIdx[0] = 0;
	for (unsigned i = 0; i < nodeCount; ++i) {
		unsigned oldNode = oldNodes[i];
		unsigned nodeDataBegin = nodeAdjStruct.adjIdx[oldNode];
		unsigned nodeDataEnd = nodeAdjStruct.adjIdx[oldNode + 1];
		unsigned nodeDataSize = nodeDataEnd - nodeDataBegin;
		unsigned newNodeDataBegin = newAdjIdx[i];
		newAdjIdx[i + 1] = newNodeDataBegin + nodeDataSize;

		for (unsigned i = 0; i < nodeDataSize; ++i)
			newAdj[newNodeDataBegin + i] = newNodes[nodeAdjStruct.adj[nodeDataBegin + i]];
		std::sort(newAdj + newNodeDataBegin, newAdj + newNodeDataBegin + nodeDataSize);
	}
	nodeAdjStruct.adj.swap(newAdj);
	nodeAdjStruct.adjIdx.swap(newAdjIdx);

	//nodeAdjStruct.init(nodeCount, elemTypes, elemInfo, elem);

	std::cout << "Nodes renumbered\n";
}


void Mesh::setGeomForElem(GeomType geomType, unsigned elemType) {
	FiniteElement& type = elemInfo[elemType];
	type.geomType = geomType;
	switch (geomType) {
	case GeomType::planeStress:
		mallocThickness(elemType);
		setThickness(elemType, 1.);
		break;
	default:
		type.data.free();
	}
}


void Mesh::mallocThickness(unsigned elemType) {
	FiniteElement& type = elemInfo[elemType];
	type.data.realloc(type.elemCount * type.nodeCount);
}


void Mesh::setThickness(unsigned elemType, double h) {
	FiniteElement& type = elemInfo[elemType];
	for (unsigned i = 0; i < type.elemCount * type.nodeCount; ++i)
		type.data[i] = h;
}

void Mesh::setThickness(unsigned elemType, std::function<double(const double*)> h) {
	FiniteElement& type = elemInfo[elemType];
	unsigned* locElem = elem + type.memIdx;
	for (unsigned i = 0; i < type.elemCount * type.nodeCount; ++i)
		type.data[i] = h(node + locElem[i] * dim);
}


void Mesh::setGeomType(GeomType geomType) {
	for (unsigned t = 0; t < elemTypes; ++t) {
		setGeomForElem(geomType, t);
	}
}


void Mesh::setGeomType(GeomType geomType, unsigned blockId) {
	for (unsigned t = 0; t < elemTypes; ++t) {
		FiniteElement& type = elemInfo[t];
		if (type.blockId == blockId) {
			setGeomForElem(geomType, t);
		}
	}
}


void Mesh::setPlaneWithThickness(double h) {
	for (unsigned t = 0; t < elemTypes; ++t) {
		FiniteElement& type = elemInfo[t];
		type.geomType = GeomType::planeStress;
		mallocThickness(t);
		setThickness(t, h);
	}
	fillBorderH(); // TEMP
}

void Mesh::setPlaneWithThickness(std::function<double(const double*)> h) {
	for (unsigned t = 0; t < elemTypes; ++t) {
		FiniteElement& type = elemInfo[t];
		type.geomType = GeomType::planeStress;
		mallocThickness(t);
		setThickness(t, h);
	}
	fillBorderH(); // TEMP
}

void Mesh::setPlaneWithThickness(unsigned blockId, double h) {
	for (unsigned t = 0; t < elemTypes; ++t) {
		FiniteElement& type = elemInfo[t];
		if (type.blockId == blockId) {
			type.geomType = GeomType::planeStress;
			mallocThickness(t);
			setThickness(t, h);
		}
	}
	fillBorderH(); // TEMP
}

void Mesh::setPlaneWithThickness(unsigned blockId, std::function<double(const double*)> h) {
	for (unsigned t = 0; t < elemTypes; ++t) {
		FiniteElement& type = elemInfo[t];
		if (type.blockId == blockId) {
			type.geomType = GeomType::planeStress;
			mallocThickness(t);
			setThickness(t, h);
		}
	}
	fillBorderH(); // TEMP
}


void Mesh::fillBorderH() {
	hptr<unsigned> nodeMap(3 * nodeCount); // { elemType, elem, node in elem }
	for (unsigned t = 0; t < elemTypes; ++t) {
		FiniteElement& type = elemInfo[t];
		unsigned* locElem = elem + type.memIdx;
		for (unsigned e = 0; e < type.elemCount; ++e)
			for (unsigned i = 0; i < type.nodeCount; ++i) {
				unsigned node = locElem[type.nodeCount * e + i];
				nodeMap[3 * node] = t;
				nodeMap[3 * node + 1] = e;
				nodeMap[3 * node + 2] = i;
			}
	}
	borderH.realloc(borderIdx[bordersCount]);
	for (unsigned j = 0; j < borderIdx[bordersCount]; ++j) {
		unsigned node = border[j];
		unsigned t = nodeMap[3 * node];
		unsigned e = nodeMap[3 * node + 1];
		unsigned i = nodeMap[3 * node + 2];
		FiniteElement& type = elemInfo[t];
		switch (type.geomType) {
		case GeomType::planeStress:
			borderH[j] = type.data[e * type.nodeCount + i];
			break;
		default:
			borderH[j] = 1.;
		}
	}
}


void Mesh::meshToRAM() {
	delete[] node2;
	node2 = new vec2[nodeCount];
	cudaMemcpy(node2, dev_node, nodeCount * sizeof(vec2), cudaMemcpyDeviceToHost);
	
	delete[] elem3;
	delete[] elem4;
	delete[] elem8;
	elem3 = new int[3 * count3];
	elem4 = new int[4 * count4];
	elem8 = new int[8 * count8];
	cudaMemcpy(elem3, dev_elem3, 3 * count3 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(elem4, dev_elem4, 4 * count4 * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(elem8, dev_elem8, 8 * count8 * sizeof(int), cudaMemcpyDeviceToHost);
	for (int i = 0; i < bordersCount; ++i)
		delete[] borders[i];
	delete[] borders;
	//bordersCount = dev_bordersCount;
	borders = new int*[bordersCount];
	//std::clog << bordersCount << "\n";
	for (int i = 0; i < bordersCount; ++i) {
		//std::clog << borderLength[i] << "\n";
		borders[i] = new int[borderLength[i]];
		//cudaMemcpy(borders[i], dev_borders[i], borderLength[i] * sizeof(int), cudaMemcpyDeviceToHost);
	}
	//std::clog << "borders copied\n";
	ramSaved = true;
}

void Mesh::meshToGPU() {
	fillPosDev();
	cudaFree(dev_node);
	cudaFree(dev_elem3);
	cudaFree(dev_elem4);
	cudaFree(dev_elem8);
	cudaMalloc((void**)&dev_node, (nodeCount + BS - 1) / BS * BS * sizeof(vec2));
	cudaMemset(dev_node, 0, ((nodeCount + BS - 1) / BS * BS - nodeCount) * sizeof(vec2));
	//cudaMalloc((void**)&dev_node, nodeCount * sizeof(vec2));
	cudaMemcpy(dev_node, node2, nodeCount * sizeof(vec2), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_elem3, 3 * count3 * sizeof(int));
	cudaMalloc((void**)&dev_elem4, 4 * count4 * sizeof(int));
	cudaMalloc((void**)&dev_elem8, 8 * count8 * sizeof(int));
	cudaMemcpy(dev_elem3, elem3, 3 * count3 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_elem4, elem4, 4 * count4 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_elem8, elem8, 8 * count8 * sizeof(int), cudaMemcpyHostToDevice);

	//for (int i = 0; i < dev_bordersCount; ++i)
	//	cudaFree(dev_borders[i]);
	//delete[] dev_borders;
	//dev_bordersCount = bordersCount;
	//dev_borders = new int* [dev_bordersCount];
	//for (int i = 0; i < dev_bordersCount; ++i) {
	//	cudaMalloc((void**)&(dev_borders[i]), borderLength[i] * sizeof(int));
	//	cudaMemcpy(dev_borders[i], borders[i], borderLength[i] * sizeof(int), cudaMemcpyHostToDevice);
	//}
	
	useCuda = true;
}

void deleteCommas(std::string& str) {
	for (int i = 0; i < str.size(); ++i)
		if (str[i] == ',') str[i] = ' ';
}

//Загрузить из файла
bool Mesh::loadFromFile(const std::string& fileName) {
	std::cout << "Mesh loading...";
	if (fileName.substr(fileName.size() - 4) != ".inp") {
		throw "Unknown mesh file format";
	}
	else if (!std::filesystem::exists(std::filesystem::path("./" + fileName))) {
		throw "There is no such file";
	}
	//std::clog << "eee\n";

	count3 = count4 = count8 = 0;
	nodeCount = 0;

	delete[] node2;
	delete[] secOrdNodes;
	delete[] elem3;
	delete[] elem4;
	delete[] elem8;
	delete[] borderLength;
	for (int i = 0; i < bordersCount; ++i)
		delete[] borders[i];
	bordersCount = 0;
	delete[] borders;
	delete[] spaces; spaces = nullptr;
	delete[] aspects; aspects = nullptr;
	delete[] skewAngles; skewAngles = nullptr;
	if (useCuda) {
		cudaFree(dev_node); dev_node = nullptr;
		cudaFree(dev_elem3); dev_elem3 = nullptr;
		cudaFree(dev_elem4); dev_elem4 = nullptr;
		cudaFree(dev_elem8); dev_elem8 = nullptr;
	}

	unsigned maxIndex = 0, input = 0;

	std::ifstream file(fileName);
	std::string line;
	while (std::getline(file, line)) {
		if (line == "*NODE")
			while (std::getline(file, line)) {
				if (line.size() != 0 && line[0] != '*') {
					++nodeCount;
					std::istringstream ss(line);
					ss >> input;
					if (input > maxIndex)
						maxIndex = input;
				}
				else break;
			}
		else if (line.substr(0, 8) == "*ELEMENT") {
			std::string type;
			for (int i = 0; i < line.length(); ++i) {
				if (line.substr(i, 5) == "TYPE=") {
					i += 5;
					//std::clog << line.substr(i, 3) << "\n";
					if (line.substr(i, 3) == "S3R") {
						//std::clog << "TYPE3!\n";
						while (std::getline(file, line)) {
							if (line.size() != 0 && line[0] != '*')
								++count4;
							else break;
						}
					}
					else if (line.substr(i, 2) == "S4") {
						while (std::getline(file, line)) {
							if (line.size() != 0 && line[0] != '*')
								++count4;
							else break;
						}
					}
					else if (line.substr(i, 3) == "S8R") {
						//std::clog << "TYPE8!\n";
						while (std::getline(file, line)) {
							if (line.size() != 0 && line[0] != '*')
								++count8;
							else break;
						}
					}
					//std::cout << line << "\n";
					break;
				}
			}
			continue;
		}
		else if (line == "*MPC") 
			for (;;) {
				int* tempBorderLength = new int[bordersCount + 1];
				memcpy(tempBorderLength, borderLength, bordersCount * sizeof(int));
				std::swap(tempBorderLength, borderLength);
				delete[] tempBorderLength;
				borderLength[bordersCount] = 0;
				while (std::getline(file, line)) {
					if (line.size() != 0 && line[0] != '*')
						++borderLength[bordersCount];
					else break;
				}
				++bordersCount;
				if (line != "*MPC") break;
			}
	}
	file.clear();
	file.seekg(0);

	//std::cout << "maxIndex = " << maxIndex << "\n";

	//std::clog << "nodes: " << nodeCount \
		<< "\nelem3: " << count3 \
		<< "\nelem4: " << count4 \
		<< "\nelem8: " << count8 << "\n\n";
	//nodeCount -= bordersCount;
	elem3 = new int[3 * count3];
	elem4 = new int[4 * count4];
	elem8 = new int[8 * count8];
	node2 = new vec2[nodeCount];
	secOrdNodes = new bool[nodeCount];
	for (int i = 0; i < nodeCount; ++i)
		secOrdNodes[i] = false;
	borders = new int*[bordersCount];
	fillPos();
	int curNode = 0, curElem3 = 0, curElem4 = 0, curElem8 = 0, curBorder = 0;
	std::list<unsigned> superNodes;
	hptr<unsigned> indexes(maxIndex + 1);

	while (std::getline(file, line)) {
		if (line == "*NODE")
			while (std::getline(file, line)) {
				if (line.size() != 0 && line[0] != '*') {
					deleteCommas(line);
					//std::clog << line << "\n";
					std::istringstream ss(line);
					ss >> input;
					indexes[input] = curNode;
					ss >> node2[curNode].x >> node2[curNode].y;
					++curNode;

					/*--input;
					if (input < nodeCount)
						ss >> node[input].x >> node[input].y;*/
					//std::clog << input << " " << node[input].x << " " << node[input].y << "\n";
				}
				else break;
			}
		else if (line.substr(0, 8) == "*ELEMENT") {
			std::string type;
			for (int i = 0; i < line.length(); ++i) {
				if (line.substr(i, 5) == "TYPE=") {
					i += 5;
					if (line.substr(i, 3) == "S3R") {
						//std::cout << "read S3R\n";
						while (std::getline(file, line)) {
							if (line.size() != 0 && line[0] != '*') {
								deleteCommas(line);
								std::istringstream ss(line);
								ss >> input;
								int begin = 4 * curElem4;
								for (int i = 0; i < 3; ++i) {
									ss >> elem4[begin + i];
									//--elem4[begin + i];
								}
								elem4[begin + 3] = elem4[begin + 2];
								++curElem4;
							}
							else break;
						}
					}
					else if (line.substr(i, 2) == "S4") {
						//std::cout << "read S4\n";
						while (std::getline(file, line)) {
							if (line.size() != 0 && line[0] != '*') {
								deleteCommas(line);
								std::istringstream ss(line);
								ss >> input;
								int begin = 4 * curElem4;
								for (int i = 0; i < 4; ++i) {
									ss >> elem4[begin + i];
									//--elem4[begin + i];
								}
								++curElem4;
							}
							else break;
						}
					}
					else if (line.substr(i, 3) == "S8R") {
						while (std::getline(file, line)) {
							if (line.size() != 0 && line[0] != '*') {
								deleteCommas(line);
								std::istringstream ss(line);
								ss >> input;
								int begin = 8 * curElem8;
								for (int i = 0; i < 8; ++i) {
									ss >> elem8[begin + i];
									//--elem8[begin + i];
								}
								for (int i = 4; i < 8; ++i)
									secOrdNodes[elem8[begin + i]] = true;
								++curElem8;
							}
							else break;
						}
					}
				}
			}
		}
		else if (line == "*MPC")
			for (;;) {
				int curPoint = 0;
				//std::clog << curBorder << " " << borderLength[curBorder] << "\n";
				borders[curBorder] = new int[borderLength[curBorder]];
				while (std::getline(file, line)) {
					if (line.size() != 0 && line[0] != '*') {
						deleteCommas(line);
						line[3] = line[2] = line[1] = line[0] = ' ';
						std::istringstream ss(line);
						ss >> borders[curBorder][curPoint];
						if (curPoint == 0) {
							int super;
							ss >> super;
							//superNodes.push_back(super - 1);
							superNodes.push_back(super);
						}
						//--borders[curBorder][curPoint];
						++curPoint;
					}
					else break;
				}
				++curBorder;
				if (line != "*MPC") break;
			}
	}
	file.close();

	/*for (unsigned i = 0; i < maxIndex + 1; ++i)
		std::cout << indexes[i] << " ";*/

	/*for (int s : superNodes)
		std::cout << "super " << s << "\n";*/

	for (unsigned i = 0; i < 3 * count3; ++i)
		elem3[i] = indexes[elem3[i]];
	for (unsigned i = 0; i < 4 * count4; ++i)
		elem4[i] = indexes[elem4[i]];
	for (unsigned i = 0; i < 8 * count8; ++i)
		elem8[i] = indexes[elem8[i]];
	for (unsigned i = 0; i < bordersCount; ++i)
		for (unsigned j = 0; j < borderLength[i]; ++j)
			borders[i][j] = indexes[borders[i][j]];
	for (unsigned& super : superNodes)
		super = indexes[super];
	for (unsigned super : superNodes) {
		for (unsigned i = super + 1; i < nodeCount; ++i)
			node2[i - 1] = node2[i];
		--nodeCount;
		for (unsigned i = 0; i < 3 * count3; ++i)
			if (elem3[i] > super)
				--elem3[i];
		for (unsigned i = 0; i < 4 * count4; ++i)
			if (elem4[i] > super)
				--elem4[i];
		for (unsigned i = 0; i < 8 * count8; ++i)
			if (elem8[i] > super)
				--elem8[i];
		for (unsigned i = 0; i < bordersCount; ++i)
			for (unsigned j = 0; j < borderLength[i]; ++j)
				if (borders[i][j] > super)
					--borders[i][j];
		for (unsigned& rest : superNodes)
			if (rest > super)
				--rest;
	}

	/*for (unsigned i = 0; i < bordersCount; ++i) {
		for (unsigned j = 0; j < borderLength[i]; ++j)
			std::cout << borders[i][j] << " ";
		std::cout << "\n";
	}*/
	//for (unsigned i = 0; i < 4 * count4; ++i)
		//std::cout << elem4[i] << (elem4[i] > 100000 ? "------" : " ");

	/*for (int s : superNodes) {
		--nodeCount;
		for (int i = s; i < nodeCount - 1; ++i)
			node[i] = node[i + 1];
		for (int i = 0; i < 3 * count3; ++i)
			if (elem3[i] > s) --elem3[i];
		for (int i = 0; i < 4 * count4; ++i)
			if (elem4[i] > s) --elem4[i];
		for (int i = 0; i < 8 * count8; ++i)
			if (elem8[i] > s) --elem8[i];
	}*/

	//for (int i = 0; i < nodeCount; ++i) \
		std::cout << i << " " << (bool)secOrdNodes[i] << "\n";
	ramSaved = true;
	analysed = false;
	if (useCuda) meshToGPU();

	translateFromLegacy();

	std::cout << "\rMesh loaded: " << nodeCount << " nodes, " << elemCount() << " elements\n\n";

	//DEBUG
	/*int maxNodeId = 0;
	for (int i = 0; i < nodeCount; ++i) {
		for (int i = 0; i < count4 * 4; ++i) {
			if (elem4[i] > maxNodeId)
				maxNodeId = elem4[i];
		}
	}
	std::cout << "Max Node ID: " << maxNodeId << "\n";*/
	/*std::cout << "\nBorders\n" << bordersCount;
	for (size_t i = 0; i < bordersCount; ++i) {
		std::cout << "\n" << borderLength[i] << ": ";
		for (size_t j = 0; j < borderLength[i]; ++j)
			std::cout << borders[i][j] << " ";
	}*/
	/*if (count4) {
		std::cout << "\nElements (4 nodes): " << count4 << "\n";
		for (size_t i = 0; i < count4; ++i) {
			for (size_t j = 4 * i; j < 4 * (i + 1); ++j)
				std::cout << elem4[j] << " ";
			std::cout << "\n";
		}
	}*/
}

void Mesh::saveAsVtk(const std::string& fileName) {
	if (!ramSaved)
		meshToRAM();
	std::ofstream file(fileName, std::ios_base::out);
	file << "# vtk DataFile Version 2.0\n";
	file << "Mesh\n";
	file << "ASCII\n";
	file << "DATASET POLYDATA\n";
	file << "POINTS " << nodeCount << " float\n";
	for (int i = 0; i < nodeCount; ++i)
		file << node2[i].x << " " << node2[i].y << " " << 0 << "\n";
	file << "POLYGONS " << elemCount() << " " << 4 * count3 + 5 * count4 + 9 * count8;
	for (size_t i = 0; i < count3; ++i) {
		file << "\n3 ";
		for (size_t j = 3 * i; j < 3 * (i + 1); ++j)
			file << elem3[j] << " ";
	}
	for (size_t i = 0; i < count4; ++i) {
		file << "\n4 ";
		for (size_t j = 4 * i; j < 4 * (i + 1); ++j)
			file << elem4[j] << " ";
	}
	for (size_t i = 0; i < count8; ++i) {
		file << "\n8 ";
		for (size_t j = 8 * i; j < 8 * i + 4; ++j)
			file << elem8[j] << " " << elem8[j + 4] << " ";
	}
	file << "\nCELL_DATA " << elemCount() << "\n";
	file << "SCALARS ElementID int 1\n";
	file << "LOOKUP_TABLE default\n";
	for (size_t i = 0; i < elemCount(); ++i) {
		file << i << "\n";
	}
	if (analysed) {
		file << "FIELD FieldData 3\n";
		file << "Element_space 1 " << elemCount() << " float\n";
		for (size_t i = 0; i < elemCount(); ++i)
			file << spaces[i] << " ";
		file << "\nElement_aspect_ratio 1 " << elemCount() << " float\n";
		for (size_t i = 0; i < elemCount(); ++i)
			file << aspects[i] << " ";
		file << "\nElement_skew_angle 1 " << elemCount() << " float\n";
		for (size_t i = 0; i < elemCount(); ++i)
			file << skewAngles[i] << " ";
	}
	file << "\nPOINT_DATA " << nodeCount << \
		"\nSCALARS NodeID int 1\nLOOKUP_TABLE my_table";
	for (size_t i = 0; i < nodeCount; ++i)
		file << "\n" << i;
	file.close();
}


// TODO: change to new data format
void Mesh::saveAsVtu(const std::string& fileName) {
	std::cout << "\nSaving...\n";
	double t = -omp_get_wtime();

	std::ofstream file(fileName, std::ios_base::out);

	file << "<?xml version=\"1.0\"?>" \
		<< "\n<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt64\" >" \
		<< "\n\t<UnstructuredGrid>" \
		<< "\n\t\t<Piece NumberOfPoints=\"" << nodeCount << "\" NumberOfCells=\"" << totalElems << "\">";

	file << "\n\t\t\t<PointData>";  // Данные в узлах
	// Номера узлов
	file << "\n\t\t\t\t<DataArray type=\"UInt32\" Name=\"NodeID\" NumberOfComponents=\"1\" format=\"ascii\" >\n";
	for (unsigned i = 0; i < nodeCount; ++i)
		file << i << " ";
	file << "\n\t\t\t\t</DataArray>";
	// Толщина плоскостей (и оболочек)
	bool hasThickness = false;
	for (unsigned t = 0; t < elemTypes; ++t) {
		GeomType type = elemInfo[t].geomType;
		if (type == GeomType::planeStress) {
			hasThickness = true;
			break;
		}
	}
	if (hasThickness) {
		file << "\n\t\t\t\t<DataArray type=\"Float64\" Name=\"Thickness\" NumberOfComponents=\"1\" format=\"ascii\" >\n";
		hptr<double> nodeH(nodeCount);
		nodeH.setZero(nodeCount);
		hptr<unsigned> inNodeCount(nodeCount);
		inNodeCount.setZero(nodeCount);
		for (unsigned t = 0; t < elemTypes; ++t) {
			FiniteElement& type = elemInfo[t];
			if (type.geomType == GeomType::planeStress) {
				unsigned* locElem = elem + type.memIdx;
				for (unsigned i = 0; i < type.nodeCount * type.elemCount; ++i) {
					unsigned ni = locElem[i];
					nodeH[ni] += type.data[i];
					++inNodeCount[ni];
				}
			}
		}
		for (unsigned i = 0; i < nodeCount; ++i)
			file << (inNodeCount[i] ? nodeH[i] / inNodeCount[i] : 0.) << " ";
		file << "\n\t\t\t\t</DataArray>";
	}

	file << "\n\t\t\t</PointData>";

	file << "\n\t\t\t<CellData>";  // Данные в элементах
	// Номера элементов
	file << "\n\t\t\t\t<DataArray type=\"UInt32\" Name=\"ElementID\" NumberOfComponents=\"1\" format=\"ascii\" >\n";
	for (unsigned i = 0; i < totalElems; ++i)
		file << i << " ";
	file << "\n\t\t\t\t</DataArray>";
	// Цвета
	if (!colorMap.empty()) {
		file << "\n\t\t\t\t<DataArray type=\"UInt8\" Name=\"Color\" NumberOfComponents=\"1\" format=\"ascii\" >\n";
		for (unsigned i = 0; i < totalElems; ++i)
			file << (unsigned)colorMap.elemColor[i] << " ";
		file << "\n\t\t\t\t</DataArray>";
	}
	file << "\n\t\t\t</CellData>";

	file << "\n\t\t\t<Points>";  // Узлы
	file << "\n\t\t\t\t<DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\" >\n";
	for (unsigned i = 0; i < nodeCount; ++i)
		file << node2[i].x << " " << node2[i].y << " 0 ";
	file << "\n\t\t\t\t</DataArray>";
	file << "\n\t\t\t</Points>";

	file << "\n\t\t\t<Cells>";  // Элементы
	file << "\n\t\t\t\t<DataArray type=\"UInt32\" Name=\"connectivity\" format=\"ascii\" >\n";
	for (int i = 0; i < count3 * 3; ++i)
		file << elem3[i] << " ";
	for (int i = 0; i < count4 * 4; ++i)
		file << elem4[i] << " ";
	for (int i = 0; i < count8 * 8; ++i)
		file << elem8[i] << " ";
	file << "\n\t\t\t\t</DataArray>";
	file << "\n\t\t\t\t<DataArray type=\"UInt32\" Name=\"offsets\" format=\"ascii\" >\n";
	unsigned offset = 0;
	for (int i = 0; i < count3; ++i)
		file << (offset += 3) << " ";
	for (int i = 0; i < count4; ++i)
		file << (offset += 4) << " ";
	for (int i = 0; i < count8; ++i)
		file << (offset += 8) << " ";
	file << "\n\t\t\t\t</DataArray>";
	file << "\n\t\t\t\t<DataArray type=\"UInt32\" Name=\"types\" format=\"ascii\" >\n";
	for (int i = 0; i < count3; ++i)
		file << "5 ";
	for (int i = 0; i < count4; ++i)
		file << "9 ";
	for (int i = 0; i < count8; ++i)
		file << "23 ";
	file << "\n\t\t\t\t</DataArray>";
	file << "\n\t\t\t</Cells>";

	file << "\n\t\t</Piece>";
	file << "\n\t</UnstructuredGrid>";
	file << "\n</VTKFile>";

	file.close();
	t += omp_get_wtime();
	std::cout << "Mesh saved in file " << fileName << " [" << t << " sec]\n";
}


//Вывод на экран
void Mesh::print() {
	if (!ramSaved)
		meshToRAM();
	std::cout << "Nodes\n" << nodeCount << "\n";
	for (size_t i = 0; i < nodeCount; ++i)
		std::cout << i + 1 << " " << node2[i].x << " " << node2[i].y << "\n";
	if (count3) {
		std::cout << "\nElements (3 nodes): " << count3 << "\n";
		for (size_t i = 0; i < count3; ++i) {
			for (size_t j = 3 * i; j < 3 * (i + 1); ++j)
				std::cout << elem3[j] + 1 << " ";
			std::cout << "\n";
		}
	}
	if (count4) {
		std::cout << "\nElements (4 nodes): " << count4 << "\n";
		for (size_t i = 0; i < count4; ++i) {
			for (size_t j = 4 * i; j < 4 * (i + 1); ++j)
				std::cout << elem4[j] + 1 << " ";
			std::cout << "\n";
		}
	}
	if (count8) {
		std::cout << "\nElements (8 nodes): " << count8 << "\n";
		for (size_t i = 0; i < count8; ++i) {
			for (size_t j = 8 * i; j < 8 * (i + 1); ++j)
				std::cout << elem8[j] + 1 << " ";
			std::cout << "\n";
		}
	}
	std::cout << "\nBorders\n" << bordersCount;
	for (size_t i = 0; i < bordersCount; ++i) {
		std::cout << "\n" << borderLength[i] << ": ";
		for (size_t j = 0; j < borderLength[i]; ++j)
			std::cout << borders[i][j] << " ";
	}
	std::cout << std::endl;
}


void Mesh::printElemInfo() const {
	std::cout << elemTypes << " finite element type" << (elemTypes == 1 ? "" : "s") << "\n";
	for (unsigned i = 0; i < elemTypes; ++i) {
		std::cout << "Type " << i << ":\n";
		elemInfo[i].print();
	}
}


void Mesh::printBorder() const {
	for (unsigned i = 0; i < bordersCount; ++i) {
		std::cout << "Border " << i << ": " << borderIdx[i + 1] - borderIdx[i] << "\n";
		for (unsigned j = borderIdx[i]; j < borderIdx[i + 1]; ++j)
			std::cout << border[j] << " ";
		std::cout << "\n";
	}
}


//TO DO: fix for elem8
void Mesh::printAnalysis() {//TO DO: all element types
	delete[] spaces;
	delete[] aspects;
	delete[] skewAngles;
	spaces = new double[count4];
	aspects = new double[count4];
	skewAngles = new double[count4];
	
	double avrSpace = 0., minSpace = 1e300, maxSpace = 0., \
		avrAspectRatio = 0., minAspectRatio = 1e300, maxAspectRatio = 0., \
		avrSkewAngleSin = 0., minSkewAngleSin = 1e300, maxSkewAngleSin = 0.;
	for (size_t e = 0; e < count4; ++e) {
		double space = elemSpace4(e);
		spaces[e] = space;
		avrSpace += space;
		if (space > maxSpace) maxSpace = space;
		if (space < minSpace) minSpace = space;
		double aspectRatio = Mesh::aspectRatio(e);
		aspects[e] = aspectRatio;
		avrAspectRatio += aspectRatio;
		if (aspectRatio > maxAspectRatio) maxAspectRatio = aspectRatio;
		if (aspectRatio < minAspectRatio) minAspectRatio = aspectRatio;
		double skewAngleSin = Mesh::skewAngleSin(e);
		skewAngles[e] = acos(skewAngleSin);
		avrSkewAngleSin += skewAngleSin;
		if (skewAngleSin > maxSkewAngleSin) maxSkewAngleSin = skewAngleSin;
		if (skewAngleSin < minSkewAngleSin) minSkewAngleSin = skewAngleSin;
	}
	avrAspectRatio /= count4;
	avrSpace /= count4;
	avrSkewAngleSin /= count4;
	std::cout << "Mesh analysis:"
		<< "\nNodes max index difference in element: " << findMaxIndexDiff() \
		<< "\nSpace: avr = " << avrSpace << ", min = " << minSpace << ", max = " << maxSpace \
		<< "\nAspect ratio " << 100. / avrAspectRatio << "%: avr = " \
		<< avrAspectRatio << ", min = " << minAspectRatio << ", max = " << maxAspectRatio \
		<< "\nSkew angle sin " << 100 * avrSkewAngleSin << "%: avr = " \
		<< avrSkewAngleSin << ", min = " << minSkewAngleSin << ", max = " << maxSkewAngleSin \
		<< "\n" << std::endl;
	analysed = true;
}


void Mesh::translateFromLegacy() {
	node.realloc(nodeCount * dim);
	memcpy(node, node2, nodeCount * 2 * sizeof(double));

	elemTypes = unsigned(count3 > 0) + unsigned(count4 > 0) + unsigned(count8 > 0);
	elemInfo.realloc(elemTypes);
	//elemInfo.resize(elemTypes);
	totalElems = count3 + count4 + count8;
	elem.realloc(count3 * 3 + count4 * 4 + count8 * 8);

	unsigned poolIdx = 0;
	unsigned memIdx = 0;
	unsigned typeIdx = 0;
	if (count3) {
		FiniteElement& type = elemInfo[typeIdx++];
		type.elemCount = count3;
		type.elemDim = 2;
		type.setType(ElemType::tria1, 1);
		//type.geomType = GeomType::planeStress;
		type.nodeCount = 3;
		type.blockId = 0;
		type.poolIdx = poolIdx;
		type.memIdx = memIdx;
		poolIdx += count3;
		memIdx += count3 * 3;
		memcpy(elem + type.memIdx, elem3, count3 * 3 * sizeof(unsigned));
	}
	if (count4) {
		FiniteElement& type = elemInfo[typeIdx++];
		type.elemCount = count4;
		type.elemDim = 2;
		type.setType(ElemType::quad1, 2);
		//type.geomType = GeomType::planeStress;
		type.nodeCount = 4;
		type.blockId = 0;
		type.poolIdx = poolIdx;
		type.memIdx = memIdx;
		poolIdx += count4;
		memIdx += count4 * 4;
		memcpy(elem + type.memIdx, elem4, count4 * 4 * sizeof(unsigned));
	}
	if (count8) {
		FiniteElement& type = elemInfo[typeIdx++];
		type.elemCount = count8;
		type.elemDim = 2;
		type.setType(ElemType::quad2, 3);
		//type.geomType = GeomType::planeStress;
		type.nodeCount = 8;
		type.blockId = 0;
		type.poolIdx = poolIdx;
		type.memIdx = memIdx;
		poolIdx += count8;
		memIdx += count8 * 8;
		memcpy(elem + type.memIdx, elem8, count8 * 8 * sizeof(unsigned));
	}

	borderIdx.realloc(bordersCount + 1);
	if (bordersCount) borderIdx[0] = 0;
	for (unsigned i = 0; i < bordersCount; ++i)
		borderIdx[i + 1] = borderIdx[i] + borderLength[i];
	border.realloc(borderIdx[bordersCount]);
	for (unsigned i = 0; i < bordersCount; ++i)
		memcpy(border + borderIdx[i], borders[i], borderLength[i] * sizeof(unsigned));
}