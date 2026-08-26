#include "PlasticitySolver.h"

#include "CG.h"
#include "CSR.h"

#include "FiniteElement.h"

//#include "cuda.h"

#include <omp.h>
#include <fstream>



#if _OPENMP >= 200805
typedef unsigned omp_for_t;
#else
typedef int omp_for_t;
#endif


static void formK_planeStress_sym(double* Ke, unsigned nodeCount, const double* C_, const double* B, const double* detJ, const double* intCoefs, const double* h, unsigned intPointsCount) {
	for (unsigned p = 0; p < intPointsCount; ++p) {
		double BC[2][3] = {};
		double coef = intCoefs[p] * detJ[p] * h[p];
		double CC[3][3];
		for (unsigned i = 0; i < 9; ++i)
			CC[0][i] = C_[i] * coef;

		/*for (unsigned i = 0; i < 9; ++i)
			printf("%e ", B[i]);
		printf("\n");*/

		for (unsigned i = 0; i < nodeCount; ++i) {
			unsigned i2 = i * 2;
			for (int j = 0; j < 3; ++j) {
				BC[0][j] = B[i] * CC[0][j] + B[nodeCount + i] * CC[2][j]; // BC(0, j) = B(0, i) * C(0, j) + B(1, i) * C(2, j)
				BC[1][j] = B[nodeCount + i] * CC[1][j] + B[i] * CC[2][j]; // BC(1, j) = B(1, i) * C(1, j) + B(0, i) * C(2, j)
			}

			Ke[i2 * (i2 + 1) / 2 + i2] += BC[0][0] * B[i] + BC[0][2] * B[nodeCount + i];         // Ke(i2, i2)
			Ke[(i2 + 1) * (i2 + 2) / 2 + i2] += BC[1][0] * B[i] + BC[1][2] * B[nodeCount + i];     // Ke(i2 + 1, i2)
			Ke[(i2 + 1) * (i2 + 2) / 2 + i2 + 1] += BC[1][1] * B[nodeCount + i] + BC[1][2] * B[i]; // Ke(i2 + 1, i2 + 1)

			for (int j = 0; j < i; ++j) {
				int j2 = j * 2;
				Ke[i2 * (i2 + 1) / 2 + j2] += BC[0][0] * B[j] + BC[0][2] * B[nodeCount + j];    // Ke(i2, j2)
				Ke[i2 * (i2 + 1) / 2 + j2 + 1] += BC[0][1] * B[nodeCount + j] + BC[0][2] * B[j]; // Ke(i2, j2 + 1)
				Ke[(i2 + 1) * (i2 + 2) / 2 + j2] += BC[1][0] * B[j] + BC[1][2] * B[nodeCount + j]; // Ke(i2 + 1, j2)
				Ke[(i2 + 1) * (i2 + 2) / 2 + j2 + 1] += BC[1][1] * B[nodeCount + j] + BC[1][2] * B[j]; // Ke(i2 + 1, j2 + 1)
			}
		}
		B += nodeCount * 2;
	}
}


static void formK_planeStrain_sym(double* Ke, unsigned nodeCount, const double* C_, const double* B, \
	const double* detJ, const double* intCoefs, unsigned intPointsCount) {
	for (unsigned p = 0; p < intPointsCount; ++p) {
		double BC[2][3] = {};
		double coef = intCoefs[p] * detJ[p];
		double CC[3][3];
		for (unsigned i = 0; i < 9; ++i)
			CC[0][i] = C_[i] * coef;

		for (unsigned i = 0; i < nodeCount; ++i) {
			unsigned i2 = i * 2;
			for (int j = 0; j < 3; ++j) {
				BC[0][j] = B[i] * CC[0][j] + B[nodeCount + i] * CC[2][j]; // BC(0, j) = B(0, i) * C(0, j) + B(1, i) * C(2, j)
				BC[1][j] = B[nodeCount + i] * CC[1][j] + B[i] * CC[2][j]; // BC(1, j) = B(1, i) * C(1, j) + B(0, i) * C(2, j)
			}

			Ke[i2 * (i2 + 1) / 2 + i2] += BC[0][0] * B[i] + BC[0][2] * B[nodeCount + i];         // Ke(i2, i2)
			Ke[(i2 + 1) * (i2 + 2) / 2 + i2] += BC[1][0] * B[i] + BC[1][2] * B[nodeCount + i];     // Ke(i2 + 1, i2)
			Ke[(i2 + 1) * (i2 + 2) / 2 + i2 + 1] += BC[1][1] * B[nodeCount + i] + BC[1][2] * B[i]; // Ke(i2 + 1, i2 + 1)

			for (int j = 0; j < i; ++j) {
				int j2 = j * 2;
				Ke[i2 * (i2 + 1) / 2 + j2] += BC[0][0] * B[j] + BC[0][2] * B[nodeCount + j];    // Ke(i2, j2)
				Ke[i2 * (i2 + 1) / 2 + j2 + 1] += BC[0][1] * B[nodeCount + j] + BC[0][2] * B[j]; // Ke(i2, j2 + 1)
				Ke[(i2 + 1) * (i2 + 2) / 2 + j2] += BC[1][0] * B[j] + BC[1][2] * B[nodeCount + j]; // Ke(i2 + 1, j2)
				Ke[(i2 + 1) * (i2 + 2) / 2 + j2 + 1] += BC[1][1] * B[nodeCount + j] + BC[1][2] * B[j]; // Ke(i2 + 1, j2 + 1)
			}
		}
		B += nodeCount * 2;
	}
}


static void formC_plainStressIsotropic(double* C, double E, double nu) {
	//Плоское напряжённое состояние
	double coef = E / (1. - nu * nu);
	C[0] = C[4] = coef;
	C[1] = C[3] = nu * coef;
	C[8] = (1. - nu) * 0.5 * coef;
	C[2] = C[5] = C[6] = C[7] = 0.;
}


static void formC_plainStrainIsotropic(double* C, double E, double nu) {
	//Плоское деформированное состояние
	double coef = E * (1. - nu) / ((1. + nu) * (1 - 2. * nu));
	C[0] = C[4] = 1. * coef;
	C[1] = C[3] = nu / (1. - nu) * coef;
	C[8] = (1. - 2. * nu) * 0.5 / (1. - nu) * coef;
	C[2] = C[5] = C[6] = C[7] = 0.;
}


// sym Ke
static void assemble_planeStress(unsigned elemType, CSR& K, \
	unsigned* colorMap, unsigned* colorMapIdx, unsigned colorCount, \
	double* B, double* detJ, double* C, double* E, double* nu, \
	unsigned* elem, unsigned nodeCount,
	unsigned intPointsCount, double* intCoefs, double* h) {

	//std::cout << "assemble plane stress\n";

	unsigned offset = std::hardware_destructive_interference_size / sizeof(double);

	unsigned size = nodeCount * 2;
	hptr<double> Ke(((1 + size) * size / 2 + offset) * omp_get_max_threads());

	//hptr<double> H_TEMP(9);
	//for (unsigned i = 0; i < 9; ++i)  // TEMP
	//	H_TEMP[i] = 1.;

//#pragma omp parallel for
//	for (omp_for_t e = 0; e < colorMapIdx[colorCount]; ++e) {
//		formC_plainStressIsotropic(C + e * 9, E[e], nu[e]);
//		double* locKe = Ke + ((1 + size) * size / 2 + offset) * omp_get_thread_num();
//		//Ke.setZero((1 + size) * size / 2);
//		memset(locKe, 0, ((1 + size) * size / 2 + offset) * sizeof(double));
//		formK_planeStress_sym(locKe, nodeCount, C + e * 9,
//			B + e * 2 * nodeCount * intPointsCount,
//			detJ + e * intPointsCount,
//			intCoefs,
//			h + e * intPointsCount,
//			intPointsCount
//		);
//		K.insertKeSym(elemType, e, locKe, elem + nodeCount * e, nodeCount, 2);
//		//for (int i = 0; i < nodeCount; ++i)
//		//	for (int j = 0; j < nodeCount; ++j) {
//		//		int gi = 2 * elem[nodeCount * e + i];  //глобальне индексы
//		//		int gj = 2 * elem[nodeCount * e + j];
//		//		K.getItem(gi, gj) += KK[(2 * i) * size + 2 * j];
//		//		K.getItem(gi + 1, gj) += KK[(2 * i + 1) * size + 2 * j];
//		//		K.getItem(gi, gj + 1) += KK[(2 * i) * size + 2 * j + 1];
//		//		K.getItem(gi + 1, gj + 1) += KK[(2 * i + 1) * size + 2 * j + 1];
//		//	}
//	}
	for (unsigned c = 0; c < colorCount; ++c) {
#pragma omp parallel for
		for (omp_for_t i = colorMapIdx[c]; i < colorMapIdx[c + 1]; ++i) {
			unsigned e = colorMap[i];
			formC_plainStressIsotropic(C + e * 9, E[e], nu[e]);
			double* locKe = Ke + ((1 + size) * size / 2 + offset) * omp_get_thread_num();
			//Ke.setZero((1 + size) * size / 2);
			memset(locKe, 0, ((1 + size) * size / 2 + offset) * sizeof(double));
			formK_planeStress_sym(locKe, nodeCount, C + e * 9,
				B + e * 2 * nodeCount * intPointsCount,
				detJ + e * intPointsCount,
				intCoefs,
				h + e * intPointsCount,
				intPointsCount
			);
			K.insertKeSym(elemType, e, locKe, elem + nodeCount * e, nodeCount, 2);
		}
	}
}


void PlasticitySolver::assemble(CSR& K) {
	//std::cout << "AVENGERS ASSEMBLE!\n";
	K.data.setZero(K.dataSize());
	for (unsigned t = 0; t < mesh.elemTypes; ++t) {
		const FiniteElement& type = mesh.elemInfo[t];
		switch (type.geomType) {
		case GeomType::planeStress:
			assemble_planeStress(t, K,
				mesh.colorMap.colorMap + type.poolIdx,
				mesh.colorMap.colorMapIdx + t * (mesh.colorMap.colorCount + 1),
				mesh.colorMap.colorCount,
				B + Bidx[t],
				detJ + detJidx[t],
				(double*)C_,  // + offset
				E_c, nu_c,
				mesh.elem + type.memIdx,
				type.nodeCount,
				type.intPointsCount, type.intCoefs,
				hsj + hsjIdx[t]
			);
			break;
		case GeomType::planeStrain:

			break;
		}
	}
}


double PlasticitySolver::solveCPU() {
	std::cout << "Solving..." << (iterOutput ? "\n" : "") << std::endl;
	//std::cout << "K size: " << stripK.size() << "\n";

	double solvingTime = -omp_get_wtime();

	calcThickness();
	calcBs();

	mesh.checkNodeAdjStruct();
	//mesh.nodeAdjStruct.print();
	//CSR K(mesh.nodeAdjStruct, 2);
	CSR K(mesh.nodeAdjStruct, mesh.elemTypes, mesh.elemInfo, mesh.elem);
	//K.printStruct();

	mesh.checkColorMap();

	uv.resize(K.N);
	hptr<double> R(K.N);
	initConditions(uv.data(), kinMask, R, lineCount, lines, lineRows);

	JCGV cjs(K.N, K.data, K.rows, K.cols, R, \
		uv.data(), kinMask, preconditioning, true, false, 2, lines, lineRows, lineCount);
	//cjs.setLinesDim(2);

	size_t iterNum = 0;
	double relErr = 0.;
	do {
		assemble(K);

		unsigned insideIter = 0;
		insideIter = cjs.solve(1e-7);

		updateParameters();
		relErr = exitCondition();
		printIter(iterNum, insideIter, relErr);

		++iterNum;
	} while (relErr > 1e-5);
	calcPlastDeform();
	solvingTime += omp_get_wtime();

	std::cout << (iterOutput ? "\n" : "") << "Solved [" << solvingTime << " sec]\n";
	if (!iterOutput)
		std::cout << "Iterations: " << iterNum << "                                             " \
		<< "\nExit error: " << exitCondition() << "\n";

	plastSolved = true;

	delete[] h;
	h = nullptr;

	return solvingTime;
}