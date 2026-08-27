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


void PlasticitySolver::initConditions(double* uvw, bool* mask, double* rhs, unsigned& lineCount, hptr<double>& lines, hptr<unsigned>& lineRows) {
	static const double GS2_point[2] = { -0.577'350'269'189'626, 0.577'350'269'189'626 };
	for (int i = 0; i < 2 * mesh.nodeCount; ++i) {
		rhs[i] = 0.;
		mask[i] = true;
		uvw[i] = 0.;
	}
	for (const auto& [id, force] : cond.forceCond) {  //граничные силовые условия
		int* border = mesh.borders[id];
		for (int i = 1; i < mesh.borderLength[id]; ++i) {
			if (mesh.secOrdNodes[border[i]]) {
				++i;
				double R[6] = {};
				vec2 node1 = mesh.node2[border[i - 2]], node2 = mesh.node2[border[i - 1]], node3 = mesh.node2[border[i]];
				vec2 F1 = force.forceFunc(node1), F2 = force.forceFunc(node2), F3 = force.forceFunc(node3);
				if (force.normOrient) {
					//F1 = func(node1);
					vec2 tang1 = (node2 - node1).normalize();
					vec2 norm1 = vec2(tang1.y, -tang1.x).normalize();
					vec2 tang2 = (node3 - node2).normalize();
					vec2 norm2 = vec2(tang2.y, -tang2.x).normalize();
					//tang *= force.value.y;
					F1 = norm1 * F1.x + tang1 * F1.y;
					F2 = 0.5 * ((norm1 + norm2) * F2.x + (tang1 + tang2) * F2.y);
					F3 = norm2 * F3.x + tang2 * F3.y;
				}
				for (unsigned p = 0; p < 2; ++p) {
					double xi = GS2_point[p];
					double N[3] = { 0.5 * xi * (xi - 1.), (1. - xi) * (1. + xi), 0.5 * xi * (xi + 1.) };
					// detJ = sqrt(det(J^T J)) ???
					double Jx = (-0.5 + xi) * node1.x - 2. * xi * node2.x + (0.5 + xi) * node3.x;
					double Jy = (-0.5 + xi) * node1.y - 2. * xi * node2.y + (0.5 + xi) * node3.y;
					double detJ = sqrt(Jx * Jx + Jy * Jy);

					double h = N[0] * mesh.borderH[mesh.borderIdx[id] + i - 2] \
						+ N[1] * mesh.borderH[mesh.borderIdx[id] + i - 1] \
						+ N[2] * mesh.borderH[mesh.borderIdx[id] + i];

					vec2 F = N[0] * F1 + N[1] * F2 + N[2] * F3;

					double coefX = h * detJ * F.x;
					double coefY = h * detJ * F.y;

					R[0] += N[0] * coefX;
					R[1] += N[0] * coefY;
					R[2] += N[1] * coefX;
					R[3] += N[1] * coefY;
					R[4] += N[2] * coefX;
					R[5] += N[2] * coefY;
				}
				unsigned gi = 2 * border[i - 2];
				unsigned gj = 2 * border[i - 1];
				unsigned gk = 2 * border[i];

				rhs[gi] += R[0];
				rhs[gi + 1] += R[1];
				rhs[gj] += R[2];
				rhs[gj + 1] += R[3];
				rhs[gk] += R[4];
				rhs[gk + 1] += R[5];
			}
			else {
				double R[2];
				vec2 node1 = mesh.node2[border[i - 1]], node2 = mesh.node2[border[i]];
				double halfDiff1 = 0.5 * (node1.x - node2.x), halfDiff2 = 0.5 * (node1.y - node2.y);
				double dl = sqrt(halfDiff1 * halfDiff1 + halfDiff2 * halfDiff2);
				vec2 F1 = force.forceFunc(node1), F2 = force.forceFunc(node2);
				if (force.normOrient) {
					//F1 = func(node1);
					vec2 tang = (node2 - node1).normalize();
					vec2 norm = vec2(tang.y, -tang.x).normalize();
					//tang *= force.value.y;
					F1 = norm * F1.x + tang * F1.y;
					F2 = norm * F2.x + tang * F2.y;
				}
				R[0] = 0.5 * (F1.x + F2.x);
				R[1] = 0.5 * (F1.y + F2.y);

				double coef = dl * 0.5 * (mesh.borderH[mesh.borderIdx[id] + i - 1] + mesh.borderH[mesh.borderIdx[id] + i]);

				R[0] *= coef;
				R[1] *= coef;

				int gi = 2 * border[i - 1];
				int gj = 2 * border[i];

				rhs[gi] += R[0];
				rhs[gi + 1] += R[1];
				rhs[gj] += R[0];
				rhs[gj + 1] += R[1];
			}
		}
	}
	if (cond.Rset) {
		for (unsigned t = 0; t < mesh.elemTypes; ++t) {
			FiniteElement& type = mesh.elemInfo[t];
			hptr<double> N(type.nodeCount);
			hptr<double> F(type.nodeCount * mesh.dim);
			hptr<double> R(type.nodeCount * mesh.dim);
			hptr<double> Fp(mesh.dim);
			unsigned* locElem = mesh.elem + type.memIdx;
			double* locDetJ = detJ + detJidx[t];

			double* locHsj = hsj + hsjIdx[t]; // TEMP, switch case for geom type

			for (unsigned e = 0; e < type.elemCount; ++e) {
				unsigned* curElem = locElem + e * type.nodeCount;
				double* curDetJ = locDetJ + e * type.intPointsCount;

				for (unsigned i = 0; i < type.nodeCount; ++i)
					((vec2*)F.raw())[i] = cond.R(*(vec2*)(mesh.node + mesh.dim * curElem[i]));
				/*for (unsigned i = 0; i < type.nodeCount; ++i)
					((vec2*)F.raw())[i] = *(vec2*)(mesh.node + mesh.dim * curElem[i]);*/
				R.setZero(type.nodeCount * mesh.dim);

				for (unsigned p = 0; p < type.intPointsCount; ++p) {
					type.calcN(N, type.intPoints + p * type.elemDim);
					for (unsigned j = 0; j < mesh.dim; ++j)
						Fp[j] = N[0] * F[j];
					for (unsigned i = 1; i < type.nodeCount; ++i)
						for (unsigned j = 0; j < mesh.dim; ++j)
							Fp[j] += N[i] * F[i * mesh.dim + j];
					double coef = m.rho * locHsj[e * type.intPointsCount + p] * curDetJ[p] * type.intCoefs[p];
					//double coef = type.intCoefs[p];
					for (unsigned i = 0; i < type.nodeCount; ++i)
						for (unsigned j = 0; j < mesh.dim; ++j)
							R[i * mesh.dim + j] += N[i] * coef * Fp[j];
				}
				for (unsigned i = 0; i < type.nodeCount; ++i) {
					unsigned line = mesh.dim * curElem[i];   // TEMP, add nodeLines array
					for (unsigned j = 0; j < mesh.dim; ++j)
						rhs[line + j] += R[i * mesh.dim + j];
					//rhs[line + j] += F[i * mesh.dim + j];
				}
			}
		}
	}
	for (const auto& [id, displ] : cond.displCond) { //общие кинематические условие
		int* border = mesh.borders[id];
		size_t end = mesh.borderLength[id] - (border[0] == border[mesh.borderLength[id] - 1] ? 1 : 0);
		if (displ.fixMidX) {
			int gi = 2 * border[mesh.borderLength[id] / 2];
			mask[gi] = false;
			uvw[gi] = displ.displFunc(mesh.node2[border[mesh.borderLength[id] / 2]]).x;
		}
		else
			for (size_t i = 0; i < end; ++i) {
				int gi = 2 * border[i];
				mask[gi] = false;
				uvw[gi] = displ.displFunc(mesh.node2[border[i]]).x;
			}
		if (displ.fixMidY) {
			int gi = 2 * border[mesh.borderLength[id] / 2] + 1;
			mask[gi] = false;
			uvw[gi] = displ.displFunc(mesh.node2[border[mesh.borderLength[id] / 2]]).y;

		}
		else
			for (size_t i = 0; i < end; ++i) {
				int gi = 2 * border[i] + 1;
				mask[gi] = false;
				uvw[gi] = displ.displFunc(mesh.node2[border[i]]).y;
			}
	}
	for (const auto& fixedAxis : cond.fixedAxis) {  //зафиксированные оси
		if (fixedAxis.vertical) {
			for (int i = 0; i < mesh.nodeCount; ++i) {
				//if (fabs(mesh.node[i].x - fixedAxis.coord) < (fabs(mesh.node[i].x) + fabs(fixedAxis.coord)) * 1e-10) {
				if (fabs(mesh.node2[i].x - fixedAxis.coord) < 1e-14) {
					mask[2 * i] = false;
				}
			}
		}
		else {
			for (int i = 0; i < mesh.nodeCount; ++i)
				//if (fabs(mesh.node[i].y - fixedAxis.coord) < (fabs(mesh.node[i].y) + fabs(fixedAxis.coord)) * 1e-14)
				if (fabs(mesh.node2[i].y - fixedAxis.coord) < 1e-14) {
					mask[2 * i + 1] = false;
				}
		}
	}
	for (const auto& [id, fixedBorder] : cond.fixedBorder) {  //зафиксированные по одной оси границы
		int* border = mesh.borders[id];
		int end = mesh.borderLength[id] - (border[0] == border[mesh.borderLength[id] - 1] ? 1 : 0);
		if (fixedBorder.vertical) {
			for (int i = 0; i < end; ++i)
				mask[2 * border[i]] = false;
		}
		else {
			for (int i = 0; i < end; ++i)
				mask[2 * border[i] + 1] = false;
		}
	}

	unsigned lineMem = 0;
	for (const auto& [id, line] : cond.alongLine) // подсчёт закреплений границ вдоль прямых
		lineMem += mesh.borderLength[id];
	lineMem += cond.pointOnLine.size();
	if (lineMem) {
		lines.malloc(2 * lineMem);
		lineRows.malloc(lineMem);
	}
	else {
		lines.free();
		lineRows.free();
	}
	lineCount = 0;
	for (const auto& [id, line] : cond.alongLine)  // закрепления границ вдоль прямых
		for (unsigned i = 0; i < mesh.borderLength[id]; ++i) {
			lineRows[lineCount] = 2 * mesh.borders[id][i];
			reinterpret_cast<vec2*>(lines.raw())[lineCount] = line;
			++lineCount;
		}
	for (const auto& line : cond.pointOnLine)  // закрепления точек вдоль прямых
		for (int i = 0; i < mesh.nodeCount; ++i)
			if ((mesh.node2[i] - line.point).norm() < 1e-14) {
				lineRows[lineCount] = 2 * i;
				reinterpret_cast<vec2*>(lines.raw())[lineCount] = line.value;
				++lineCount;
				break;
			}

	for (const auto& forcePoint : cond.forcePoint) {  //сила в точке
		for (int i = 0; i < mesh.nodeCount; ++i)
			if ((mesh.node2[i] - forcePoint.point).norm() < 1e-14) {
				rhs[2 * i] += forcePoint.value.x;
				rhs[2 * i + 1] += forcePoint.value.y;
				break;
			}
	}
	for (const auto& displPoint : cond.displPoint) {  //перемещение в точке
		for (int i = 0; i < mesh.nodeCount; ++i)
			if ((mesh.node2[i] - displPoint.point).norm() < 1e-14) {
				uvw[2 * i] = displPoint.value.x;
				uvw[2 * i + 1] = displPoint.value.y;
				mask[2 * i] = false;
				mask[2 * i + 1] = false;
				break;
			}
	}
}