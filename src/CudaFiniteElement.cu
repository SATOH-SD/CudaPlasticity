#include "FiniteElement.h"


template<typename fp>
static __device__ void N_quad1(fp* N, const fp* xi) {
	fp r = xi[0], s = xi[1];
	N[0] = 0.25 * (1. + r) * (1 + s);
	N[1] = 0.25 * (1. - r) * (1 + s);
	N[2] = 0.25 * (1. - r) * (1 - s);
	N[3] = 0.25 * (1. + r) * (1 - s);
}

template<typename fp>
static __device__ void (*ptr_N_quad1)(fp* N, const fp* xi) = N_quad1;


template<typename fp>
static __device__ void B_quad1(fp* B, fp* detJ, const fp* x, const fp* xi) {
	fp x1 = x[0], y1 = x[1],
		x2 = x[2], y2 = x[3],
		x3 = x[4], y3 = x[5],
		x4 = x[6], y4 = x[7];
	fp r = xi[0], s = xi[1];
	fp J[4];  // TODO: оптимизировать через сохранения значений производных функций форм
	J[0] = 0.25 * ((1. + s) * x1 - (1. + s) * x2 - (1. - s) * x3 + (1. - s) * x4); //x_r
	J[2] = 0.25 * ((1. + r) * x1 + (1. - r) * x2 - (1. - r) * x3 - (1. + r) * x4); //x_s
	J[1] = 0.25 * ((1. + s) * y1 - (1. + s) * y2 - (1. - s) * y3 + (1. - s) * y4); //y_r
	J[3] = 0.25 * ((1. + r) * y1 + (1. - r) * y2 - (1. - r) * y3 - (1. + r) * y4); //y_s

	*detJ = J[0] * J[3] - J[1] * J[2];

	fp invJ[4];
	fp invDet = 1. / *detJ;
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

template<typename fp>
static __device__ void (*ptr_B_quad1)(fp* B, fp* detJ, const fp* x, const fp* xi) = B_quad1;


template<typename fp>
static __device__ void N_quad2(fp* N, const fp* xi) {
	fp r = xi[0], s = xi[1];
	N[0] = 0.25 * (1. + r) * (1. + s) * (r + s - 1.);
	N[1] = -0.25 * (1. - r) * (1. + s) * (r - s + 1.);
	N[2] = -0.25 * (1. - r) * (1. - s) * (r + s + 1.);
	N[3] = 0.25 * (1. + r) * (1. - s) * (r - s - 1.);
	N[4] = 0.5 * (1. - r * r) * (1. + s);
	N[5] = 0.5 * (1. - r) * (1. - s * s);
	N[6] = 0.5 * (1. - r * r) * (1. - s);
	N[7] = 0.5 * (1. + r) * (1. - s * s);
}

template<typename fp>
static __device__ void (*ptr_N_quad2)(fp* N, const fp* xi) = N_quad2;


template<typename fp>
static __device__ void B_quad2(fp* B, fp* detJ, const fp* x, const fp* xi) {
	fp x1 = x[0], y1 = x[1],
		x2 = x[2], y2 = x[3],
		x3 = x[4], y3 = x[5],
		x4 = x[6], y4 = x[7],
		x5 = x[8], y5 = x[9],
		x6 = x[10], y6 = x[11],
		x7 = x[12], y7 = x[13],
		x8 = x[14], y8 = x[15];
	fp r = xi[0], s = xi[1];
	fp J[4];  // TODO: оптимизировать через сохранения значений производных функций форм
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

	*detJ = J[0] * J[3] - J[1] * J[2];

	fp invJ[4];
	fp invDet = 1. / *detJ;
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

template<typename fp>
static __device__ void (*ptr_B_quad2)(fp* B, fp* detJ, const fp* x, const fp* xi) = B_quad2;


CudaFiniteElement::CudaFiniteElement() = default;


CudaFiniteElement::CudaFiniteElement(const FiniteElement& fe) {
	intPointsCount = fe.intPointsCount;
	intCoefs.malloc(intPointsCount);
	hostToDevice(intCoefs, fe.intCoefs, intPointsCount);
	unsigned intMem = intPointsCount * fe.elemDim;
	intPoints.malloc(intMem);
	hostToDevice(intPoints, fe.intPoints, intMem);
	setType(fe.elemType);
}


void CudaFiniteElement::addFloat(const FiniteElement& fe) {
	hptr<float> coefs_f(intPointsCount);
	for (unsigned i = 0; i < intPointsCount; ++i)
		coefs_f[i] = float(fe.intCoefs[i]);
	intCoefs_f.realloc(intPointsCount);
	hostToDevice(intCoefs_f, coefs_f, intPointsCount);

	unsigned intMem = intPointsCount * fe.elemDim;
	hptr<float> points_f(intMem);
	for (unsigned i = 0; i < intMem; ++i)
		points_f[i] = float(fe.intPoints[i]);
	intPoints_f.realloc(intMem);
	hostToDevice(intPoints_f, points_f, intMem);
}


void CudaFiniteElement::setType(ElemType elemType) {
	switch (elemType) {
	case ElemType::quad1:
		cudaMemcpyFromSymbol(&calcN, ptr_N_quad1<double>, sizeof(calcN));
		cudaMemcpyFromSymbol(&calcN_f, ptr_N_quad1<float>, sizeof(calcN));
		cudaMemcpyFromSymbol(&calcB, ptr_B_quad1<double>, sizeof(calcB));
		cudaMemcpyFromSymbol(&calcB_f, ptr_B_quad1<float>, sizeof(calcB));
		break;
	case ElemType::quad2:
		cudaMemcpyFromSymbol(&calcN, ptr_N_quad2<double>, sizeof(calcN));
		cudaMemcpyFromSymbol(&calcN_f, ptr_N_quad2<float>, sizeof(calcN));
		cudaMemcpyFromSymbol(&calcB, ptr_B_quad2<double>, sizeof(calcB));
		cudaMemcpyFromSymbol(&calcB_f, ptr_B_quad2<float>, sizeof(calcB));
		break;
	}
}