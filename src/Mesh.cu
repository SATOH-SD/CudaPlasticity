#include <filesystem>

#include "Mesh.cuh"

#include "vec2.cuh"

__global__ void rectangleNodes(vec2* node, int N2_1, double x1, double h1, double y1, double h2) {
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int i = k / N2_1, j = k % N2_1;
	node[k].x = x1 + h1 * i;
	node[k].y = y1 + h2 * j;
	if (k < 25)
		printf("%d - %d, %d - %e %e\n", k, i, j, node[k].x, node[k].y);
}


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
	if (0 * useCuda) {
		int grid = (nodeCount + BS - 1) / BS;
		int dev_nodeCount = grid * BS;
		cudaMalloc((void**)&dev_node, dev_nodeCount * sizeof(vec2));
		double h1 = (x2 - x1) / N1, h2 = (y2 - y1) / N2;
		std::cout << x1 << " " << h1 << " " << y1 << " " << h2 << "\n";
		rectangleNodes<<<grid, BS>>>(dev_node, N2 + 1, x1, h1, y1, h2);
		cudaDeviceSynchronize();
		
		grid = (count4 + BS - 1) / BS;
		int dev_count4 = grid * BS;
		cudaMalloc((void**)&dev_elem4, dev_count4 * 4 * sizeof(int));

		//meshToRAM();
	}
	else {
		//node.resize((N1 + 1) * (N2 + 1));
		
		delete[] node;
		delete[] secOrdNodes;
		node = new vec2[nodeCount];
		secOrdNodes = new bool[nodeCount];
		for (int i = 0; i < nodeCount; ++i)
			secOrdNodes[i] = false;
		double h1 = (x2 - x1) / N1, h2 = (y2 - y1) / N2;
		for (int i = 0; i <= N1; ++i)
			for (int j = 0; j <= N2; ++j) {
				int k = i * (N2 + 1) + j;
				node[k].x = x1 + h1 * i;
				node[k].y = y1 + h2 * j;
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
	}
	std::cout << "\rMesh generated: " << nodeCount << " nodes, " << elemCount() << " elements\n\n";
}


void Mesh::genRectWithHole(double x1, double x2, double y1, double y2, double holeRad, size_t N1, size_t N2, int order) {
	std::cout << "Mesh generation...";
	int borderN = (N1 + N2) * 2;
	int radN = (N1 + N2) / 4;
	radN = round(double(radN) * (x2 - x1 + y2 - y1 - 4. * holeRad) * 2. / (x2 - x1 + y2 - y1));
	//std::cout << radN << "\n";
	if (order == 2) {
		N1 *= 2; N2 *= 2;
		borderN *= 2;
		radN *= 2;
	}

	delete[] borderLength;
	for (int i = 0; i < bordersCount; ++i)
		delete[] borders[i];
	bordersCount = 0;
	delete[] borders;

	nodeCount = borderN * (radN + 1);
	delete[] node;
	delete[] secOrdNodes;
	node = new vec2[nodeCount];
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
		node[nodeId] = { x1 + i * hx, y1 };
		borders[0][i] = nodeId;
	}
	borders[0][borderLength[0] - 1] = nodeId;
	for (int j = 0; j < N2; ++j, ++nodeId) {
		node[nodeId] = { x2, y1 + j * hy };
		borders[1][j] = nodeId;
	}
	borders[1][borderLength[1] - 1] = nodeId;
	for (int i = 0; i < N1; ++i, ++nodeId) {
		node[nodeId] = { x2 - i * hx, y2 };
		borders[2][i] = nodeId;
	}
	borders[2][borderLength[2] - 1] = nodeId;
	for (int j = 0; j < N2; ++j, ++nodeId) {
		node[nodeId] = { x1, y2 - j * hy };
		borders[3][j] = nodeId;
	}
	borders[3][borderLength[3] - 1] = 0;

	const double pi = 3.141'592'653'589'793;
	double phi = 2. * pi / borderN;
	nodeId = radN * borderN;
	vec2 center(0.5 * (x1 + x2), 0.5 * (y1 + y2));
	double start = phi * (N1 + N2 * 1.5);
	for (size_t i = 0; i < borderN; ++i, ++nodeId) {
		node[nodeId] = center + vec2(cos(start + i * phi), sin(start + i * phi)) * holeRad;
		borders[4][borderN - i - 1] = nodeId;
	}
	borders[4][borderLength[4] - 1] = nodeCount - 1;

	vec2 rh = (node[radN * borderN] - node[0]) / radN;
	for (int i = 1; i < radN; ++i) //rad - 1?
		node[i * borderN] = node[0] + rh * i;
	rh = (node[nodeCount - 1] - node[borderN - 1]) / radN;
	for (int i = 1; i < radN; ++i)
		node[i * borderN + borderN - 1] = node[borderN - 1] + rh * i;
	for (size_t j = 1; j < borderN; ++j) {
		vec2 rh = (node[radN * borderN + j] - node[j]) / radN;
		for (size_t i = 1; i < radN; ++i)
			node[i * borderN + j] = node[j] + rh * i;
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

	std::cout << "\rMesh generated: " << nodeCount << " nodes, " << elemCount() << " elements\n\n";
	//printAnalysis();
}


//Сгенерировать сетку для кольца с внутренним радиусом а и внешним радиусом b
void Mesh::genRing(double a, double b, size_t N_phi, size_t N_r, int order) {
	std::cout << "Mesh generation...";
	if (order == 2) {
		N_phi *= 2; N_r *= 2;
	}
	delete[] borderLength;
	for (int i = 0; i < bordersCount; ++i)
		delete[] borders[i];
	bordersCount = 0;
	delete[] borders;

	nodeCount = (N_r + 1) * N_phi;
	delete[] node;
	delete[] secOrdNodes;
	node = new vec2[nodeCount];
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

	const double pi = 3.141'592'653'589'793;
	double h_r = (b - a) / N_r, h_phi = 2. * pi / N_phi;

	for (int j = 0; j < N_phi; ++j) {
		for (int i = 0; i <= N_r; ++i) {
			double r = b - h_r * i;
			double phi = h_phi * j;
			node[i * N_phi + j] = vec2(r * cos(phi), r * sin(phi));
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
	ramSaved = true;
	analysed = false;
	//useCuda = false;
	//smoothRing(borderN);
	renumerateRing(N_phi);
	if (order == 2) remapOrder(N_r);

	if (useCuda) meshToGPU();

	std::cout << "\rMesh generated: " << nodeCount << " nodes, " << elemCount() << " elements\n\n";
	//printAnalysis();
}


void Mesh::renumerateRing(int borderN) {
	std::vector<int> newNodes(nodeCount);
	int newN = 0, radN = nodeCount / borderN,
		iBack = radN - 1, iFront = 0,
		jBack = borderN - 1, jFront = 0;
	int i = 0;

	while (jBack > jFront) {
		for (int i = 0; i < radN; ++i)
			newNodes[(i)*borderN + (jFront)] = newN++;
		for (int i = 0; i < radN; ++i)
			newNodes[(i)*borderN + (jBack)] = newN++;
		++jFront;
		--jBack;
	}
	if (jBack == jFront)
		for (int i = 0; i < radN; ++i)
			newNodes[newN++] = (i)*borderN + (jFront);

	/*while (jBack > jFront) {
		for (int i = 0; i < radN; ++i) {
			if (i % 2 == 0)
				newNodes[(i / 2)*borderN + (jFront)] = newN++;
			else
				newNodes[(radN - 1 - i / 2) * borderN + (jFront)] = newN++;
		}
		for (int i = 0; i < radN; ++i) {
			if (i % 2 == 0)
				newNodes[(i / 2) * borderN + (jBack)] = newN++;
			else
				newNodes[(radN - 1 - i / 2) * borderN + (jBack)] = newN++;
		}
		++jFront;
		--jBack;
	}
	if (jBack == jFront)
		for (int i = 0; i < radN; ++i)
			newNodes[newN++] = (i)*borderN + (jFront);*/

	//for (int i = 0; i < nodeCount; ++i) \
		std::cout << i << " " << newNodes[i] << "\n";

	for (int i = 0; i < count4; ++i)
		for (int j = 4 * i; j < 4 * (i + 1); ++j)
			elem4[j] = newNodes[elem4[j]];
	for (int i = 0; i < bordersCount; ++i)
		for (int j = 0; j < borderLength[i]; ++j)
			borders[i][j] = newNodes[borders[i][j]];
	std::vector<vec2> nodes(nodeCount);
	for (size_t i = 0; i < nodeCount; ++i)
		nodes[newNodes[i]] = node[i];
	memcpy(node, nodes.data(), nodeCount * sizeof(vec2));
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
				double ddy = 0.5 * ((node[(i + 1) * borderN + j] - node[i * borderN + j]).norm() + (node[(i - 1) * borderN + j] - node[i * borderN + j]).norm()),
					ddx = 0.5 * ((node[i * borderN + j + 1] - node[i * borderN + j]).norm() + (node[i * borderN + j - 1] - node[i * borderN + j]).norm());
				//node[i * borderN + j] = (node[(i + 1) * borderN + j] * ddx + node[(i - 1) * borderN + j] * ddy) / (ddx + ddy);
				//node[i * borderN + j] = node[(i + 1) * borderN + j] - (node[(i + 1) * borderN + j] - node[i * borderN + j]).normalize() * ddx;
				node[i * borderN + j] = node[(i + 1) * borderN + j] - \
					(node[(i + 1) * borderN + j] - node[i * borderN + j]).normalize() * 2. * pi * node[(i + 1) * borderN + j].norm() / borderN;
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
				if (diff > max)
					max = diff;
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
			nodes[newNodes[i]] = node[i];
	nodeCount -= count8;
	delete[] node;
	node = nodes;

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

void renumSort(double* rBegin, double* rEnd, int* nBegin, double* bufferR, int* bufferN) {
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
		ranges[i] = node[i] * direction;
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

	for (int i = 0; i < 4 * count4; ++i)
		elem4[i] = newNodes[elem4[i]];
	for (int i = 0; i < 8 * count8; ++i)
		elem8[i] = newNodes[elem8[i]];
	for (int i = 0; i < bordersCount; ++i)
		for (int j = 0; j < borderLength[i]; ++j)
			borders[i][j] = newNodes[borders[i][j]];
	vec2* nodes = new vec2[nodeCount];
	for (size_t i = 0; i < nodeCount; ++i)
		nodes[newNodes[i]] = node[i];
	memcpy(node, nodes, nodeCount * sizeof(vec2));
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
}


void Mesh::meshToRAM() {
	delete[] node;
	node = new vec2[nodeCount];
	cudaMemcpy(node, dev_node, nodeCount * sizeof(vec2), cudaMemcpyDeviceToHost);
	
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
	bordersCount = dev_bordersCount;
	borders = new int*[bordersCount];
	//std::clog << bordersCount << "\n";
	for (int i = 0; i < bordersCount; ++i) {
		//std::clog << borderLength[i] << "\n";
		borders[i] = new int[borderLength[i]];
		cudaMemcpy(borders[i], dev_borders[i], borderLength[i] * sizeof(int), cudaMemcpyDeviceToHost);
	}
	//std::clog << "borders copied\n";
	if (analysed) {
		delete[] spaces;
		spaces = new double[count4];
		cudaMemcpy(spaces, dev_spaces, count4 * sizeof(double), cudaMemcpyDeviceToHost);
		delete[] aspects;
		aspects = new double[count4];
		cudaMemcpy(aspects, dev_aspects, count4 * sizeof(double), cudaMemcpyDeviceToHost);
		delete[] skewAngles;
		skewAngles = new double[count4];
		cudaMemcpy(skewAngles, dev_skewAngles, count4 * sizeof(double), cudaMemcpyDeviceToHost);
	}
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
	cudaMemcpy(dev_node, node, nodeCount * sizeof(vec2), cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_elem3, 3 * count3 * sizeof(int));
	cudaMalloc((void**)&dev_elem4, 4 * count4 * sizeof(int));
	cudaMalloc((void**)&dev_elem8, 8 * count8 * sizeof(int));
	cudaMemcpy(dev_elem3, elem3, 3 * count3 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_elem4, elem4, 4 * count4 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_elem8, elem8, 8 * count8 * sizeof(int), cudaMemcpyHostToDevice);

	for (int i = 0; i < dev_bordersCount; ++i)
		cudaFree(dev_borders[i]);
	delete[] dev_borders;
	dev_bordersCount = bordersCount;
	dev_borders = new int* [dev_bordersCount];
	for (int i = 0; i < dev_bordersCount; ++i) {
		cudaMalloc((void**)&(dev_borders[i]), borderLength[i] * sizeof(int));
		cudaMemcpy(dev_borders[i], borders[i], borderLength[i] * sizeof(int), cudaMemcpyHostToDevice);
	}
	if (analysed) {
		cudaFree(dev_spaces);
		cudaMalloc((void**)&dev_spaces, elemCount() * sizeof(double));
		cudaMemcpy(dev_spaces, spaces, elemCount() * sizeof(double), cudaMemcpyHostToDevice);
		cudaFree(dev_aspects);
		cudaMalloc((void**)&dev_aspects, elemCount() * sizeof(double));
		cudaMemcpy(dev_aspects, aspects, elemCount() * sizeof(double), cudaMemcpyHostToDevice);
		cudaFree(dev_skewAngles);
		cudaMalloc((void**)&dev_skewAngles, elemCount() * sizeof(double));
		cudaMemcpy(dev_skewAngles, skewAngles, elemCount() * sizeof(double), cudaMemcpyHostToDevice);
	}
	useCuda = true;
}

void deleteCommas(std::string& str) {
	for (int i = 0; i < str.size(); ++i)
		if (str[i] == ',') str[i] = ' ';
}

//Загрузить из файла
bool Mesh::loadFromFile(const std::string& fileName) {
	if (fileName.substr(fileName.size() - 4) != ".inp") {
		throw "Unknown mesh file format";
	}
	else if (!std::filesystem::exists(std::filesystem::path("./" + fileName))) {
		throw "There is no such file";
	}
	//std::clog << "eee\n";

	count3 = count4 = count8 = 0;
	nodeCount = 0;

	delete[] node;
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
		for (int i = 0; i < dev_bordersCount; ++i)
			cudaFree(dev_borders[i]);
		dev_bordersCount = 0;
		cudaFree(dev_spaces); dev_spaces = nullptr;
		cudaFree(dev_aspects); dev_aspects = nullptr;
		cudaFree(dev_skewAngles); dev_skewAngles = nullptr;
	}
	delete[] dev_borders; dev_borders = nullptr;

	std::ifstream file(fileName);
	std::string line;
	while (std::getline(file, line)) {
		if (line == "*NODE")
			while (std::getline(file, line)) {
				if (line.size() != 0 && line[0] != '*')
					++nodeCount;
				else break;
			}
		else if (line.substr(0, 8) == "*ELEMENT") {
			std::string type;
			for (int i = 0; i < line.length(); ++i) {
				if (line.substr(i, 5) == "TYPE=") {
					i += 5;
					std::clog << line.substr(i, 3) << "\n";
					if (line.substr(i, 2) == "S3") {
						while (std::getline(file, line)) {
							if (line.size() != 0 && line[0] != '*')
								++count3;
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
						std::clog << "TYPE8!\n";
						while (std::getline(file, line)) {
							if (line.size() != 0 && line[0] != '*')
								++count8;
							else break;
						}
					}
				}
			}
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
	file.close();
	//std::clog << "nodes: " << nodeCount \
		<< "\nelem3: " << count3 \
		<< "\nelem4: " << count4 \
		<< "\nelem8: " << count8 << "\n\n";
	nodeCount -= bordersCount;
	elem3 = new int[3 * count3];
	elem4 = new int[4 * count4];
	elem8 = new int[8 * count8];
	node = new vec2[nodeCount];
	secOrdNodes = new bool[nodeCount];
	for (int i = 0; i < nodeCount; ++i)
		secOrdNodes[i] = false;
	borders = new int*[bordersCount];
	fillPos();
	int curElem3 = 0, curElem4 = 0, curElem8 = 0, curBorder = 0, input = 0;

	file.open(fileName);
	while (std::getline(file, line)) {
		if (line == "*NODE")
			while (std::getline(file, line)) {
				if (line.size() != 0 && line[0] != '*') {
					deleteCommas(line);
					//std::clog << line << "\n";
					std::istringstream ss(line);
					ss >> input;
					--input;
					if (input < nodeCount)
						ss >> node[input].x >> node[input].y;
					//std::clog << input << " " << node[input].x << " " << node[input].y << "\n";
				}
				else break;
			}
		else if (line.substr(0, 8) == "*ELEMENT") {
			std::string type;
			for (int i = 0; i < line.length(); ++i) {
				if (line.substr(i, 5) == "TYPE=") {
					i += 5;
					if (line.substr(i, 2) == "S3") {
						while (std::getline(file, line)) {
							if (line.size() != 0 && line[0] != '*') {
								deleteCommas(line);
								std::istringstream ss(line);
								ss >> input;
								int begin = 3 * curElem3;
								for (int i = 0; i < 3; ++i) {
									ss >> elem3[begin + i];
									--elem3[begin + i];
								}
								++curElem3;
							}
							else break;
						}
					}
					else if (line.substr(i, 2) == "S4") {
						while (std::getline(file, line)) {
							if (line.size() != 0 && line[0] != '*') {
								deleteCommas(line);
								std::istringstream ss(line);
								ss >> input;
								int begin = 4 * curElem4;
								for (int i = 0; i < 4; ++i) {
									ss >> elem4[begin + i];
									--elem4[begin + i];
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
									--elem8[begin + i];
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
						--borders[curBorder][curPoint];
						++curPoint;
					}
					else break;
				}
				++curBorder;
				if (line != "*MPC") break;
			}
	}
	file.close();
	//for (int i = 0; i < nodeCount; ++i) \
		std::cout << i << " " << (bool)secOrdNodes[i] << "\n";
	ramSaved = true;
	analysed = false;
	if (useCuda) meshToGPU();
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
		file << node[i].x << " " << node[i].y << " " << 0 << "\n";
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

//Вывод на экран
void Mesh::print() {
	if (!ramSaved)
		meshToRAM();
	std::cout << "Nodes\n" << nodeCount << "\n";
	for (size_t i = 0; i < nodeCount; ++i)
		std::cout << i + 1 << " " << node[i].x << " " << node[i].y << "\n";
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
	cudaFree(dev_spaces);
	cudaMalloc((void**)&dev_spaces, count4 * sizeof(double));
	cudaMemcpy(dev_spaces, spaces, count4 * sizeof(double), cudaMemcpyHostToDevice);
	cudaFree(dev_aspects);
	cudaMalloc((void**)&dev_aspects, count4 * sizeof(double));
	cudaMemcpy(dev_aspects, aspects, count4 * sizeof(double), cudaMemcpyHostToDevice);
	cudaFree(dev_skewAngles);
	cudaMalloc((void**)&dev_skewAngles, count4 * sizeof(double));
	cudaMemcpy(dev_skewAngles, skewAngles, count4 * sizeof(double), cudaMemcpyHostToDevice);
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