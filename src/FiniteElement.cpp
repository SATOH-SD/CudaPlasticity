#include "FiniteElement.h"

#include <cmath>
#include <iostream>


unsigned vtuElemType(ElemType elemType) {
	switch (elemType) {
	//case ElemType::beam1: return 3;
	//case ElemType::beam2: return 21;
	case ElemType::tria1: return 5;
	//case ElemType::tria2: return 22;
	case ElemType::quad1: return 9;
	case ElemType::quad2: return 23;
	//case ElemType::tetra1: return 10;
	//case ElemType::tetra2: return 24;
	//case ElemType::hexa1: return 12;
	//case ElemType::hexa2: return 25;
	//case ElemType::wedge1: return 13;
	//case ElemType::wedge2: return 26;
	//case ElemType::pyra1: return 14;
	//case ElemType::pyra2: return 27;
	default: return 0;
	}
}


static constexpr double gaussPoints1 = 0.;
static constexpr double gaussCoef1 = 2.;

static const double gaussPoints2[2] = { -1. / sqrt(3.), 1. / sqrt(3.) };
static const double gaussCoef2[2] = { 1., 1. };

static const double gaussPoints3[3] = { -sqrt(0.6), 0., sqrt(0.6) };
static const double gaussCoef3[3] = { 5. / 9., 8. / 9., 5. / 9. };

static const double gaussPoints4[4] = {
	-sqrt(3. / 7. + 2. / 7. * sqrt(1.2)),
	-sqrt(3. / 7. - 2. / 7. * sqrt(1.2)),
	sqrt(3. / 7. - 2. / 7. * sqrt(1.2)),
	sqrt(3. / 7. + 2. / 7. * sqrt(1.2))
};
static const double gaussCoef4[4] = {
	(18. - sqrt(30.)) / 36.,
	(18. + sqrt(30.)) / 36.,
	(18. + sqrt(30.)) / 36.,
	(18. - sqrt(30.)) / 36.
};

static const double gaussPoints5[5] = {
	-sqrt(5. + 2. * sqrt(10. / 7.)) / 3.,
	-sqrt(5. - 2. * sqrt(10. / 7.)) / 3.,
	0.,
	sqrt(5. - 2. * sqrt(10. / 7.)) / 3.,
	sqrt(5. + 2. * sqrt(10. / 7.)) / 3.
};
static const double gaussCoef5[5] = {
	(322. - 13. * sqrt(70.)) / 900.,
	(322. + 13. * sqrt(70.)) / 900.,
	128. / 225.,
	(322. + 13. * sqrt(70.)) / 900.,
	(322. - 13. * sqrt(70.)) / 900.
};


static void fillGaussScheme(double* points, double* coefs, unsigned n, unsigned dim, unsigned& totalPoints) {
	const double* points1D = nullptr;
	const double* coef1D = nullptr;
	totalPoints = n;
	for (unsigned i = 1; i < dim; ++i, totalPoints *= n);
	switch (n) {
	case 1: points1D = &gaussPoints1; coef1D = &gaussCoef1; break;
	case 2: points1D = gaussPoints2; coef1D = gaussCoef2; break;
	case 3: points1D = gaussPoints3; coef1D = gaussCoef3; break;
	case 4: points1D = gaussPoints4; coef1D = gaussCoef4;  break;
	case 5: points1D = gaussPoints5; coef1D = gaussCoef5;  break;
	}
	switch (dim) {
	case 1:
		for (unsigned i = 0; i < n; ++i) {
			points[i] = points1D[i];
			coefs[i] = coef1D[i];
		}
		break;
	case 2:
		for (unsigned i = 0; i < n; ++i)
			for (unsigned j = 0; j < n; ++j) {
				unsigned pi = n * i + j;
				points[2 * pi] = points1D[i];
				points[2 * pi + 1] = points1D[j];
				coefs[pi] = coef1D[i] * coef1D[j];
			}
		break;
	case 3:
		for (unsigned i = 0; i < n; ++i)
			for (unsigned j = 0; j < n; ++j)
				for (unsigned k = 0; k < n; ++j) {
					unsigned pi = n * (n * i + j) + k;
					points[3 * pi] = points1D[i];
					points[3 * pi + 1] = points1D[j];
					points[3 * pi + 2] = points1D[k];
					coefs[pi] = coef1D[i] * coef1D[j] * coef1D[k];
				}
		break;
	}
}


// TODO: функция для заполнения схемы интегрирования в треугольниках и тетраэдрах



static void N_quad1(double* N, const double* xi) {
	double r = xi[0], s = xi[1];
	N[0] = 0.25 * (1. + r) * (1 + s);
	N[1] = 0.25 * (1. - r) * (1 + s);
	N[2] = 0.25 * (1. - r) * (1 - s);
	N[3] = 0.25 * (1. + r) * (1 - s);
}


static void B_quad1(double* B, double& detJ, const double* x, const double* xi) {
	double
		x1 = x[0], y1 = x[1],
		x2 = x[2], y2 = x[3],
		x3 = x[4], y3 = x[5],
		x4 = x[6], y4 = x[7];
	double r = xi[0], s = xi[1];
	double J[4];  // TODO: оптимизировать через сохранения значений производных функций форм
	J[0] = 0.25 * ((1. + s) * x1 - (1. + s) * x2 - (1. - s) * x3 + (1. - s) * x4); //x_r
	J[2] = 0.25 * ((1. + r) * x1 + (1. - r) * x2 - (1. - r) * x3 - (1. + r) * x4); //x_s
	J[1] = 0.25 * ((1. + s) * y1 - (1. + s) * y2 - (1. - s) * y3 + (1. - s) * y4); //y_r
	J[3] = 0.25 * ((1. + r) * y1 + (1. - r) * y2 - (1. - r) * y3 - (1. + r) * y4); //y_s

	detJ = J[0] * J[3] - J[1] * J[2];

	double invJ[4];
	double invDet = 1. / detJ;
	invJ[0] = J[3] * invDet;
	invJ[1] = -J[1] * invDet;
	invJ[2] = -J[2] * invDet;
	invJ[3] = J[0] * invDet;

	B[0] = 0.25 * ((1. + s) * invJ[0] + (1. + r) * invJ[1]);   // [0, 0]
	B[4] = 0.25 * ((1. + s) * invJ[2] + (1. + r) * invJ[3]);   // [1, 0]
	B[1] = 0.25 * ((-1. - s) * invJ[0] + (1. - r) * invJ[1]);  // [0, 1]
	B[5] = 0.25 * ((-1. - s) * invJ[2] + (1. - r) * invJ[3]);  // [1, 1]
	B[2] = 0.25 * ((-1. + s) * invJ[0] - (1. - r) * invJ[1]);  // [0, 2]
	B[6] = 0.25 * ((-1. + s) * invJ[2] - (1. - r) * invJ[3]);  // [1, 2]
	B[3] = 0.25 * ((1. - s) * invJ[0] - (1. + r) * invJ[1]);   // [0, 3]
	B[7] = 0.25 * ((1. - s) * invJ[2] - (1. + r) * invJ[3]);   // [1, 3]
}

static void N_quad2(double* N, const double* xi) {
	double r = xi[0], s = xi[1];
	N[0] = 0.25 * (1. + r) * (1. + s) * (r + s - 1.);
	N[1] = -0.25 * (1. - r) * (1. + s) * (r - s + 1.);
	N[2] = -0.25 * (1. - r) * (1. - s) * (r + s + 1.);
	N[3] = 0.25 * (1. + r) * (1. - s) * (r - s - 1.);
	N[4] = 0.5 * (1. - r * r) * (1. + s);
	N[5] = 0.5 * (1. - r) * (1. - s * s);
	N[6] = 0.5 * (1. - r * r) * (1. - s);
	N[7] = 0.5 * (1. + r) * (1. - s * s);
}


static void B_quad2(double* B, double& detJ, const double* x, const double* xi) {
	double
		x1 = x[0], y1 = x[1],
		x2 = x[2], y2 = x[3],
		x3 = x[4], y3 = x[5],
		x4 = x[6], y4 = x[7],
		x5 = x[8], y5 = x[9],
		x6 = x[10], y6 = x[11],
		x7 = x[12], y7 = x[13],
		x8 = x[14], y8 = x[15];
	double r = xi[0], s = xi[1];
	double J[4];  // TODO: оптимизировать через сохранения значений производных функций форм
	J[0] = 0.25 * (s * (x1 - x2 + x3 - x4) - 2. * x6 +
		2. * r * (x1 + s * x1 + x2 + s * x2 + x3 - s * x3 + x4 - s * x4 - 2 * x5 - 2. * s * x5 -
			2. * x7 + 2. * s * x7) + s * s * (x1 - x2 - x3 + x4 + 2. * x6 - 2. * x8) + 2. * x8);
	//x_s
	J[2] = 0.25 * (r * r * (x1 + x2 - x3 - x4 - 2. * x5 + 2. * x7) +
		2. * (x5 - x7 + s * (x1 + x2 + x3 + x4 - 2. * x6 - 2. * x8)) +
		r * (x1 + 2. * s * x1 - x2 - 2. * s * x2 + x3 - 2. * s * x3 - x4 + 2. * s * x4 + 4 * s * x6 -
			4. * s * x8));
	//y_r
	J[1] = 0.25 * (s * (y1 - y2 + y3 - y4) - 2. * y6 +
		2. * r * (y1 + s * y1 + y2 + s * y2 + y3 - s * y3 + y4 - s * y4 - 2 * y5 -
			2. * s * y5 - 2. * y7 + 2. * s * y7) +
		s * s * (y1 - y2 - y3 + y4 + 2. * y6 - 2. * y8) + 2. * y8);
	//y_s
	J[3] = 0.25 * (r * r * (y1 + y2 - y3 - y4 - 2. * y5 + 2. * y7) +
		2. * (y5 - y7 + s * (y1 + y2 + y3 + y4 - 2. * y6 - 2. * y8)) +
		r * (y1 + 2. * s * y1 - y2 - 2. * s * y2 + y3 - 2. * s * y3 - y4 + 2. * s * y4 + 4. * s * y6 -
			4. * s * y8));

	detJ = J[0] * J[3] - J[1] * J[2];

	double invJ[4];
	double invDet = 1. / detJ;
	invJ[0] = J[3] * invDet;  //r_x
	invJ[1] = -J[1] * invDet; //s_x
	invJ[2] = -J[2] * invDet; //r_y
	invJ[3] = J[0] * invDet;  //s_y

	B[0] = 0.25 * ((1. + s) * (-1. + r + s) * invJ[0] + (1. + r) * (-1. + r + s) * invJ[1] + (1. + r) * (1. + s) * (invJ[0] + invJ[1]));   // [0, 0]
	B[8] = 0.25 * ((1. + s) * (-1. + r + s) * invJ[2] + (1. + r) * (-1. + r + s) * invJ[3] + (1. + r) * (1. + s) * (invJ[2] + invJ[3]));   // [1, 0]
	B[1] = 0.25 * (-s * ((1 + s) * invJ[0] - 2. * invJ[1]) + r * r * invJ[1] + r * (2. * (1. + s) * invJ[0] - (1. + 2. * s) * invJ[1]));   // [0, 1]
	B[9] = 0.25 * (-s * ((1 + s) * invJ[2] - 2. * invJ[3]) + r * r * invJ[3] + r * (2. * (1. + s) * invJ[2] - (1. + 2. * s) * invJ[3]));   // [1, 1]
	B[2] = 0.25 * (-r * r * invJ[1] + s * (-((-1. + s) * invJ[0]) + 2. * invJ[1]) + r * (-2. * (-1. + s) * invJ[0] + (1. - 2. * s) * invJ[1]));  // [0, 2]
	B[10] = 0.25 * (-r * r * invJ[3] + s * (-((-1. + s) * invJ[2]) + 2. * invJ[3]) + r * (-2. * (-1. + s) * invJ[2] + (1. - 2. * s) * invJ[3])); // [1, 2]
	B[3] = 0.25 * (-r * r * invJ[1] + s * ((-1. + s) * invJ[0] + 2. * invJ[1]) - r * (2. * (-1. + s) * invJ[0] + (1. - 2. * s) * invJ[1]));      // [0, 3]
	B[11] = 0.25 * (-r * r * invJ[3] + s * ((-1. + s) * invJ[2] + 2. * invJ[3]) - r * (2. * (-1. + s) * invJ[2] + (1. - 2. * s) * invJ[3]));     // [1, 3]
	B[4] = -r * (1. + s) * invJ[0] - 0.5 * (-1. + r * r) * invJ[1];   // [0, 4]
	B[12] = -r * (1. + s) * invJ[2] - 0.5 * (-1. + r * r) * invJ[3];  // [1, 4]
	B[5] = 0.5 * (-1. + s * s) * invJ[0] + (-1. + r) * s * invJ[1];   // [0, 5]
	B[13] = 0.5 * (-1. + s * s) * invJ[2] + (-1. + r) * s * invJ[3];  // [1, 5]
	B[6] = r * (-1. + s) * invJ[0] + 0.5 * (-1. + r * r) * invJ[1];   // [0, 6]
	B[14] = r * (-1. + s) * invJ[2] + 0.5 * (-1. + r * r) * invJ[3];  // [1, 6]
	B[7] = -0.5 * (-1. + s * s) * invJ[0] - (1. + r) * s * invJ[1];   // [0, 7]
	B[15] = -0.5 * (-1. + s * s) * invJ[2] - (1. + r) * s * invJ[3];  // [1, 7]
}



FiniteElement::FiniteElement(ElemType elemType, unsigned gaussPoints) {
	setType(elemType, gaussPoints);
}


void FiniteElement::setType(ElemType elemType, unsigned gaussPoints) {
	FiniteElement::elemType = elemType;
	switch (elemType) {
	case ElemType::quad1:
		calcN = N_quad1;
		calcB = B_quad1;
		break;
	case ElemType::quad2:
		calcN = N_quad2;
		calcB = B_quad2;
		break;
	}
	setIntPointsCount(gaussPoints);
}


void FiniteElement::setIntPointsCount(unsigned gaussPoints) {
	switch (elemType) {
	case ElemType::none:
		intPoints.free();
		intCoefs.free();
		break;
	case ElemType::tria1:
		// realloc
		// fill
		break;
	case ElemType::quad1: case ElemType::quad2:
		intPoints.realloc(2 * gaussPoints * gaussPoints);
		intCoefs.realloc(gaussPoints * gaussPoints);
		fillGaussScheme(intPoints, intCoefs, gaussPoints, 2, intPointsCount);
		break;
	}
}


void FiniteElement::print() const {
	std::cout << "Element type: ";
	switch (elemType) {
	//case ElemType::beam1: std::cout << "beam1"; break;
	//case ElemType::beam2: std::cout << "beam2"; break;
	case ElemType::tria1: std::cout << "tria1"; break;
	//case ElemType::tria2: std::cout << "tria2"; break;
	case ElemType::quad1: std::cout << "quad1"; break;
	case ElemType::quad2: std::cout << "quad2"; break;
	//case ElemType::tetra1: std::cout << "tetra1"; break;
	//case ElemType::tetra2: std::cout << "tetra2"; break;
	//case ElemType::hexa1: std::cout << "hexa1"; break;
	//case ElemType::hexa2: std::cout << "hexa2"; break;
	//case ElemType::wedge1: std::cout << "wedge1"; break;
	//case ElemType::wedge2: std::cout << "wedge2"; break;
	//case ElemType::pyra1: std::cout << "pyra1"; break;
	//case ElemType::pyra2: std::cout << "pyra2"; break;
	default: std::cout << "none";
	}
	// TODO: geomType
	std::cout << "\nTotal integration points: " << intPointsCount \
		<< "\nBlock ID: " << blockId \
		<< "\nCount: " << elemCount \
		<< "\nPool begin index: " << poolIdx \
		<< "\nMemory begin index: " << memIdx
		<< "\nN: " << calcN
		<< "\nB: " << calcB
		<< "\n";
}