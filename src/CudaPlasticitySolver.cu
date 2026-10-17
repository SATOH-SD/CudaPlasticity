#include "PlasticitySolver.h"

#include "DeviceCommon.cuh"
#include "StaticMatrix.cuh"
#include "SymMatrix.cuh"

//#include "ConjGradSolver.h"
//#include "ConjGradCuda.cu";

#include "CG.h"
#include "CudaCG.cu"
#include "CSR.h"
#include "CudaCSR.cuh"

#include "FiniteElement.h"

//#include "cuda.h"

#include <omp.h>
#include <fstream>


__constant__ double GS2_d_point[2] = { -0.577'350'269'189'626, 0.577'350'269'189'626 };
__constant__ float GS2_f_point[2] = { -0.577'350'269, 0.577'350'269 };
//__constant__ double GS2c_coef[4] = { 1., 1. };

static const double GS2_point[2] = { -0.577'350'269'189'626, 0.577'350'269'189'626 };


template<typename fp>
__host__ __device__
static void calcB3(int e, StaticMatrix<2, 3, fp>* Bs, vec2* node, int* elem);

template<typename fp>
__host__ __device__
static void calcB4(const int e, StaticMatrix<2, 4, fp>* Bs, fp* detJ, const vec2* node, const int* elem) {
	const fp GS_point[2] = { -0.577'350'269'189'626, 0.577'350'269'189'626 };
	int begin = 4 * e;
	fp x1 = node[elem[begin]].x, y1 = node[elem[begin]].y,
		x2 = node[elem[begin + 1]].x, y2 = node[elem[begin + 1]].y,
		x3 = node[elem[begin + 2]].x, y3 = node[elem[begin + 2]].y,
		x4 = node[elem[begin + 3]].x, y4 = node[elem[begin + 3]].y;
	for (int i = 0; i < 2; ++i)
		for (int j = 0; j < 2; ++j) {
			int k = i * 2 + j;
			fp r = GS_point[i], s = GS_point[j];
			StaticMatrix<2, 2, fp> J;
			J.data[0] = 0.25 * ((1. + s) * x1 - (1. + s) * x2 - (1. - s) * x3 + (1. - s) * x4); //x_r
			J.data[2] = 0.25 * ((1. + r) * x1 + (1. - r) * x2 - (1. - r) * x3 - (1. + r) * x4); //x_s
			J.data[1] = 0.25 * ((1. + s) * y1 - (1. + s) * y2 - (1. - s) * y3 + (1. - s) * y4); //y_r
			J.data[3] = 0.25 * ((1. + r) * y1 + (1. - r) * y2 - (1. - r) * y3 - (1. + r) * y4); //y_s

			detJ[begin + k] = J.data[0] * J.data[3] - J.data[1] * J.data[2];

			StaticMatrix<2, 2, fp> invJ;
			fp invDet = 1. / detJ[begin + k];
			invJ.data[0] = J.data[3] * invDet;
			invJ.data[1] = -J.data[1] * invDet;
			invJ.data[2] = -J.data[2] * invDet;
			invJ.data[3] = J.data[0] * invDet;

			StaticMatrix<2, 4, fp>& B = Bs[begin + k];
			
			B(0, 0) = 0.25 * ((1. + s) * invJ.data[0] + (1. + r) * invJ.data[1]);
			B(1, 0) = 0.25 * ((1. + s) * invJ.data[2] + (1. + r) * invJ.data[3]);
			B(0, 1) = 0.25 * ((-1. - s) * invJ.data[0] + (1. - r) * invJ.data[1]);
			B(1, 1) = 0.25 * ((-1. - s) * invJ.data[2] + (1. - r) * invJ.data[3]);
			B(0, 2) = 0.25 * ((-1. + s) * invJ.data[0] - (1. - r) * invJ.data[1]);
			B(1, 2) = 0.25 * ((-1. + s) * invJ.data[2] - (1. - r) * invJ.data[3]);
			B(0, 3) = 0.25 * ((1. - s) * invJ.data[0] - (1. + r) * invJ.data[1]);
			B(1, 3) = 0.25 * ((1. - s) * invJ.data[2] - (1. + r) * invJ.data[3]);
		}
}

template<typename fp>
__host__ __device__
static void calcB8(const int e, StaticMatrix<2, 8, fp>* Bs, fp* detJ, const vec2* node, const int* elem) {
	int begin = 8 * e;
	const fp GS_point[3] = {-0.774'596'669'241'483, 0., 0.774'596'669'241'483 };
	//const fp GS_point[4] = { -0.861136311594053, -0.3399810435848563, 0.3399810435848563, 0.861136311594053 };
	fp x1 = node[elem[begin]].x, y1 = node[elem[begin]].y,
		x2 = node[elem[begin + 1]].x, y2 = node[elem[begin + 1]].y,
		x3 = node[elem[begin + 2]].x, y3 = node[elem[begin + 2]].y,
		x4 = node[elem[begin + 3]].x, y4 = node[elem[begin + 3]].y,
		x5 = node[elem[begin + 4]].x, y5 = node[elem[begin + 4]].y,
		x6 = node[elem[begin + 5]].x, y6 = node[elem[begin + 5]].y,
		x7 = node[elem[begin + 6]].x, y7 = node[elem[begin + 6]].y,
		x8 = node[elem[begin + 7]].x, y8 = node[elem[begin + 7]].y;
	begin = e * secIntPs * secIntPs;
	for (int i = 0; i < secIntPs; ++i)
		for (int j = 0; j < secIntPs; ++j) {
			int k = i * secIntPs + j;
			fp r = GS_point[i], s = GS_point[j];
			StaticMatrix<2, 2, fp> J;
			//x_r
			J.data[0] = 0.25 * (s * (x1 - x2 + x3 - x4) - 2. * x6 +
				2. * r * (x1 + s * x1 + x2 + s * x2 + x3 - s * x3 + x4 - s * x4 - 2 * x5 - 2. * s * x5 -
					2. * x7 + 2. * s * x7) + s * s * (x1 - x2 - x3 + x4 + 2. * x6 - 2. * x8) + 2. * x8);
			//x_s
			J.data[2] = 0.25 * (r * r * (x1 + x2 - x3 - x4 - 2. * x5 + 2. * x7) +
				2. * (x5 - x7 + s * (x1 + x2 + x3 + x4 - 2. * x6 - 2. * x8)) +
				r * (x1 + 2. * s * x1 - x2 - 2. * s * x2 + x3 - 2. * s * x3 - x4 + 2. * s * x4 + 4 * s * x6 -
					4. * s * x8));
			//y_r
			J.data[1] = 0.25 * (s * (y1 - y2 + y3 - y4) - 2. * y6 +
				2. * r * (y1 + s * y1 + y2 + s * y2 + y3 - s * y3 + y4 - s * y4 - 2 * y5 -
					2. * s * y5 - 2. * y7 + 2. * s * y7) +
				s * s * (y1 - y2 - y3 + y4 + 2. * y6 - 2. * y8) + 2. * y8);
			//y_s
			J.data[3] = 0.25 * (r * r * (y1 + y2 - y3 - y4 - 2. * y5 + 2. * y7) +
				2. * (y5 - y7 + s * (y1 + y2 + y3 + y4 - 2. * y6 - 2. * y8)) +
				r * (y1 + 2. * s * y1 - y2 - 2. * s * y2 + y3 - 2. * s * y3 - y4 + 2. * s * y4 + 4. * s * y6 -
					4. * s * y8));

			detJ[begin + k] = J.data[0] * J.data[3] - J.data[1] * J.data[2];

			StaticMatrix<2, 2, fp> invJ;
			fp invDet = 1. / detJ[begin + k];
			invJ.data[0] = J.data[3] * invDet;  //r_x
			invJ.data[1] = -J.data[1] * invDet; //s_x
			invJ.data[2] = -J.data[2] * invDet; //r_y
			invJ.data[3] = J.data[0] * invDet;  //s_y

			StaticMatrix<2, 8, fp>& B = Bs[begin + k];
			
			B(0, 0) = 0.25 * ((1. + s) * (-1. + r + s) * invJ.data[0] + (1. + r) * (-1. + r + s) * invJ.data[1] + (1. + r) * (1. + s) * (invJ.data[0] + invJ.data[1]));
			B(1, 0) = 0.25 * ((1. + s) * (-1. + r + s) * invJ.data[2] + (1. + r) * (-1. + r + s) * invJ.data[3] + (1. + r) * (1. + s) * (invJ.data[2] + invJ.data[3]));
			B(0, 1) = 0.25 * (-s * ((1 + s) * invJ.data[0] - 2. * invJ.data[1]) + r * r * invJ.data[1] + r * (2. * (1. + s) * invJ.data[0] - (1. + 2. * s) * invJ.data[1]));
			B(1, 1) = 0.25 * (-s * ((1 + s) * invJ.data[2] - 2. * invJ.data[3]) + r * r * invJ.data[3] + r * (2. * (1. + s) * invJ.data[2] - (1. + 2. * s) * invJ.data[3]));
			B(0, 2) = 0.25 * (-r * r * invJ.data[1] + s * (-((-1. + s) * invJ.data[0]) + 2. * invJ.data[1]) + r * (-2. * (-1. + s) * invJ.data[0] + (1. - 2. * s) * invJ.data[1]));
			B(1, 2) = 0.25 * (-r * r * invJ.data[3] + s * (-((-1. + s) * invJ.data[2]) + 2. * invJ.data[3]) + r * (-2. * (-1. + s) * invJ.data[2] + (1. - 2. * s) * invJ.data[3]));
			B(0, 3) = 0.25 * (-r * r * invJ.data[1] + s * ((-1. + s) * invJ.data[0] + 2. * invJ.data[1]) - r * (2. * (-1. + s) * invJ.data[0] + (1. - 2. * s) * invJ.data[1]));
			B(1, 3) = 0.25 * (-r * r * invJ.data[3] + s * ((-1. + s) * invJ.data[2] + 2. * invJ.data[3]) - r * (2. * (-1. + s) * invJ.data[2] + (1. - 2. * s) * invJ.data[3]));
			B(0, 4) = -r * (1. + s) * invJ.data[0] - 0.5 * (-1. + r * r) * invJ.data[1];
			B(1, 4) = -r * (1. + s) * invJ.data[2] - 0.5 * (-1. + r * r) * invJ.data[3];
			B(0, 5) = 0.5 * (-1. + s * s) * invJ.data[0] + (-1. + r) * s * invJ.data[1];
			B(1, 5) = 0.5 * (-1. + s * s) * invJ.data[2] + (-1. + r) * s * invJ.data[3];
			B(0, 6) = r * (-1. + s) * invJ.data[0] + 0.5 * (-1. + r * r) * invJ.data[1];
			B(1, 6) = r * (-1. + s) * invJ.data[2] + 0.5 * (-1. + r * r) * invJ.data[3];
			B(0, 7) = -0.5 * (-1. + s * s) * invJ.data[0] - (1. + r) * s * invJ.data[1];
			B(1, 7) = -0.5 * (-1. + s * s) * invJ.data[2] - (1. + r) * s * invJ.data[3];
		}
}

template<typename fp>
__global__ static void calcBs3(StaticMatrix<2, 3, fp>* Bs, const vec2* node, const int* elem, const int count);

template<typename fp>
__global__ static void calcBs4(StaticMatrix<2, 4, fp>* Bs, fp* detJ, const vec2* node, const int* elem, const int count) {
	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e < count)
		calcB4<fp>(e, Bs, detJ, node, elem);
}

template<typename fp>
__global__ static void calcBs8(StaticMatrix<2, 8, fp>* Bs, fp* detJ, const vec2* node, const int* elem, const int count) {
	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e < count)
		calcB8<fp>(e, Bs, detJ, node, elem);
}

__host__ __device__
static inline void formKe3(SymMatrix<6, double>& Ke, const int e, const StaticMatrix<2, 3, double>* Bs, const StaticMatrix<3, 3, double>& C_, const double h);

template<typename fp>
__host__ __device__
static inline void formKe4(SymMatrix<8, fp>& Ke, const int e, const StaticMatrix<2, 4, fp>* Bs, const fp* detJ, const StaticMatrix<3, 3, fp>& C_, const fp h) {
	for (int k = 0; k < 4; ++k) {
		const StaticMatrix<2, 4, fp>& B = Bs[4 * e + k];
		StaticMatrix<2, 3, fp> BC;
		const fp dJ = detJ[4 * e + k];

		/*for (unsigned i = 0; i < 8; ++i)
			printf("%e ", B.data[i]);
		printf("\n");*/

		for (int i = 0; i < 4; ++i) {
			int i2 = i * 2;
			for (int j = 0; j < 3; ++j) {
				BC(0, j) = (B(0, i) * C_(0, j) + B(1, i) * C_(2, j)) * dJ;
				BC(1, j) = (B(1, i) * C_(1, j) + B(0, i) * C_(2, j)) * dJ;
			}

			Ke(i2, i2) += BC(0, 0) * B(0, i) + BC(0, 2) * B(1, i);
			Ke(i2 + 1, i2) += BC(1, 0) * B(0, i) + BC(1, 2) * B(1, i);
			Ke(i2 + 1, i2 + 1) += BC(1, 1) * B(1, i) + BC(1, 2) * B(0, i);

			for (int j = 0; j < i; ++j) {
				int j2 = j * 2;
				Ke(i2, j2) += BC(0, 0) * B(0, j) + BC(0, 2) * B(1, j);
				Ke(i2, j2 + 1) += BC(0, 1) * B(1, j) + BC(0, 2) * B(0, j);
				Ke(i2 + 1, j2) += BC(1, 0) * B(0, j) + BC(1, 2) * B(1, j);
				Ke(i2 + 1, j2 + 1) += BC(1, 1) * B(1, j) + BC(1, 2) * B(0, j);
			}
		}
	}
	for (int i = 0; i < Ke.dataSize(); ++i)
		Ke.data[i] *= h;

	//if (e == 2)
	//Ke.print();
}

template<typename fp>
__host__ __device__
static inline void formKe4(SymMatrix<8, fp>& Ke, const int e, const StaticMatrix<2, 4, fp>* Bs, const fp* detJ, const StaticMatrix<3, 3, fp>& C_, const fp* h) {
	//if (e == 0) \
		printf("%f %f %f %f\n", h[4 * e], h[4 * e + 1], h[4 * e + 2], h[4 * e + 3]);
	for (int k = 0; k < 4; ++k) {
		const StaticMatrix<2, 4, fp>& B = Bs[4 * e + k];
		StaticMatrix<2, 3, fp> BC;
		const fp coef = detJ[4 * e + k] * h[4 * e + k];

		for (int i = 0; i < 4; ++i) {
			int i2 = i * 2;
			for (int j = 0; j < 3; ++j) {
				BC(0, j) = (B(0, i) * C_(0, j) + B(1, i) * C_(2, j)) * coef;
				BC(1, j) = (B(1, i) * C_(1, j) + B(0, i) * C_(2, j)) * coef;
			}

			Ke(i2, i2) += BC(0, 0) * B(0, i) + BC(0, 2) * B(1, i);
			Ke(i2 + 1, i2) += BC(1, 0) * B(0, i) + BC(1, 2) * B(1, i);
			Ke(i2 + 1, i2 + 1) += BC(1, 1) * B(1, i) + BC(1, 2) * B(0, i);

			for (int j = 0; j < i; ++j) {
				int j2 = j * 2;
				Ke(i2, j2) += BC(0, 0) * B(0, j) + BC(0, 2) * B(1, j);
				Ke(i2, j2 + 1) += BC(0, 1) * B(1, j) + BC(0, 2) * B(0, j);
				Ke(i2 + 1, j2) += BC(1, 0) * B(0, j) + BC(1, 2) * B(1, j);
				Ke(i2 + 1, j2 + 1) += BC(1, 1) * B(1, j) + BC(1, 2) * B(0, j);
			}
		}
	}
	//if (e == 0) \
		Ke.print();
}

template<typename fp>
__host__ __device__
static inline void formKe8(SymMatrix<16, fp>& Ke, const int e, const StaticMatrix<2, 8, fp>* Bs, const fp* detJ, const StaticMatrix<3, 3, fp>& C_, const fp h) {
	const fp GS_coef[3] = { 0.555'555'555'555'555, 0.888'888'888'888'888, 0.555'555'555'555'555 };
	//const fp GS_coef[4] = { 0.3478548451374538, 0.652145154862546, 0.652145154862546, 0.3478548451374538 };
	int begin = e * secIntPs * secIntPs;
	for (int ig = 0; ig < secIntPs; ++ig)
		for (int jg = 0; jg < secIntPs; ++jg) {
			int k = ig * secIntPs + jg;
			const StaticMatrix<2, 8, fp>& B = Bs[begin + k];
			StaticMatrix<2, 3, fp> BC;
			const fp dJ = detJ[begin + k];
			fp coef = GS_coef[ig] * GS_coef[jg];

			for (int i = 0; i < 8; ++i) {
				int i2 = i * 2;
				for (int j = 0; j < 3; ++j) {
					BC(0, j) = (B(0, i) * C_(0, j) + B(1, i) * C_(2, j)) * dJ;
					BC(1, j) = (B(1, i) * C_(1, j) + B(0, i) * C_(2, j)) * dJ;
				}

				Ke(i2, i2) += (BC(0, 0) * B(0, i) + BC(0, 2) * B(1, i)) * coef;
				Ke(i2 + 1, i2) += (BC(1, 0) * B(0, i) + BC(1, 2) * B(1, i)) * coef;
				Ke(i2 + 1, i2 + 1) += (BC(1, 1) * B(1, i) + BC(1, 2) * B(0, i)) * coef;

				for (int j = 0; j < i; ++j) {
					int j2 = j * 2;
					Ke(i2, j2) += (BC(0, 0) * B(0, j) + BC(0, 2) * B(1, j)) * coef;
					Ke(i2, j2 + 1) += (BC(0, 1) * B(1, j) + BC(0, 2) * B(0, j)) * coef;
					Ke(i2 + 1, j2) += (BC(1, 0) * B(0, j) + BC(1, 2) * B(1, j)) * coef;
					Ke(i2 + 1, j2 + 1) += (BC(1, 1) * B(1, j) + BC(1, 2) * B(0, j)) * coef;
				}
			}
		}
	for (int i = 0; i < Ke.dataSize(); ++i)
		Ke.data[i] *= h;
	//TODO: optimize coefs mults
}

template<typename fp>
__host__ __device__
static inline void formKe8(SymMatrix<16, fp>& Ke, const int e, const StaticMatrix<2, 8, fp>* Bs, const fp* detJ, const StaticMatrix<3, 3, fp>& C_, const fp* h) {
	const fp GS_coef[3] = { 0.555'555'555'555'555, 0.888'888'888'888'888, 0.555'555'555'555'555 };
	//const fp GS_coef[4] = { 0.3478548451374538, 0.652145154862546, 0.652145154862546, 0.3478548451374538 };
	int begin = e * secIntPs * secIntPs;
	for (int ig = 0; ig < secIntPs; ++ig)
		for (int jg = 0; jg < secIntPs; ++jg) {
			int k = ig * secIntPs + jg;
			const StaticMatrix<2, 8, fp>& B = Bs[begin + k];
			StaticMatrix<2, 3, fp> BC;
			const fp dJ = detJ[begin + k];
			fp coef = GS_coef[ig] * GS_coef[jg] * h[begin + k];

			for (int i = 0; i < 8; ++i) {
				int i2 = i * 2;
				for (int j = 0; j < 3; ++j) {
					BC(0, j) = (B(0, i) * C_(0, j) + B(1, i) * C_(2, j)) * dJ;
					BC(1, j) = (B(1, i) * C_(1, j) + B(0, i) * C_(2, j)) * dJ;
				}

				Ke(i2, i2) += (BC(0, 0) * B(0, i) + BC(0, 2) * B(1, i)) * coef;
				Ke(i2 + 1, i2) += (BC(1, 0) * B(0, i) + BC(1, 2) * B(1, i)) * coef;
				Ke(i2 + 1, i2 + 1) += (BC(1, 1) * B(1, i) + BC(1, 2) * B(0, i)) * coef;

				for (int j = 0; j < i; ++j) {
					int j2 = j * 2;
					Ke(i2, j2) += (BC(0, 0) * B(0, j) + BC(0, 2) * B(1, j)) * coef;
					Ke(i2, j2 + 1) += (BC(0, 1) * B(1, j) + BC(0, 2) * B(0, j)) * coef;
					Ke(i2 + 1, j2) += (BC(1, 0) * B(0, j) + BC(1, 2) * B(1, j)) * coef;
					Ke(i2 + 1, j2 + 1) += (BC(1, 1) * B(1, j) + BC(1, 2) * B(0, j)) * coef;
				}
			}
		}
	//TODO: optimize coefs mults
}





template<typename fp>
__host__ static void formC_planeStress(StaticMatrix<3, 3, fp>& C_, fp E, fp nu) {
	//Плоское напряжённое состояние
	C_(0, 0) = C_(1, 1) = 1., C_(0, 1) = C_(1, 0) = nu, C_(2, 2) = (1.f - nu) * 0.5f;
	fp coef = E / (1.f - nu * nu);
	for (int i = 0; i < 9; ++i)
		C_.data[i] *= coef;
}

template<typename fp>
__host__ static void formC_planeStrain(StaticMatrix<3, 3, fp>& C_, fp E, fp nu) {
	//Плоское деформированное состояние
	C_(0, 0) = C_(1, 1) = 1., C_(0, 1) = C_(1, 0) = nu / (1. - nu), C_(2, 2) = (1. - 2. * nu) * 0.5 / (1. - nu);
	fp coef = E * (1. - nu) / ((1. + nu) * (1 - 2. * nu));
	for (int i = 0; i < 9; ++i)
		C_.data[i] *= coef;
}

template<typename fp>
__device__ static void dev_formC_planeStress(StaticMatrix<3, 3, fp>& C_, fp E, fp nu) {
	//Плоское напряжённое состояние
	C_(0, 0) = C_(1, 1) = 1., C_(0, 1) = C_(1, 0) = nu, C_(2, 2) = (1.f - nu) * 0.5f;
	fp coef = E / (1.f - nu * nu);
	for (int i = 0; i < 9; ++i)
		C_.data[i] *= coef;
}

template<typename fp>
__device__ static void dev_formC_planeStrain(StaticMatrix<3, 3, fp>& C_, fp E, fp nu) {
	//Плоское деформированное состояние
	C_(0, 0) = C_(1, 1) = 1., C_(0, 1) = C_(1, 0) = nu / (1. - nu), C_(2, 2) = (1. - 2. * nu) * 0.5 / (1. - nu);
	fp coef = E * (1. - nu) / ((1. + nu) * (1 - 2. * nu));
	for (int i = 0; i < 9; ++i)
		C_.data[i] *= coef;
}

template<typename fp>
__device__ static void (*ptr_stress)(StaticMatrix<3, 3, fp>&, fp, fp) = dev_formC_planeStress;

template<typename fp>
__device__ static void (*ptr_strain)(StaticMatrix<3, 3, fp>&, fp, fp) = dev_formC_planeStrain;


void PlasticitySolver::setPlaneCondition(planeCond pc) {
	PlasticitySolver::pc = pc;
	switch (pc) {
	case planeCond::stress:
		formC = formC_planeStress;
		if (mesh.useCuda) {
			cudaMemcpyFromSymbol(&dd_formC, ptr_stress<double>, sizeof(ptr_stress<double>));
			cudaMemcpyFromSymbol(&df_formC, ptr_stress<float>, sizeof(ptr_stress<float>));
			//cudaMemcpyFromSymbol(&df_formC, dev_formC_planeStress<float>, sizeof(ptr_stress<float>));
		}
		break;
	case planeCond::strain:
		formC = formC_planeStrain;
		if (mesh.useCuda) {
			cudaMemcpyFromSymbol(&dd_formC, ptr_strain<double>, sizeof(ptr_strain<double>));
			cudaMemcpyFromSymbol(&df_formC, ptr_strain<float>, sizeof(ptr_strain<float>));
		}
		break;
	}
}


double PlasticitySolver::solveLinearCPU() {
	std::cout << "Solving..." << std::endl;
	StripSLAE K(uv.size(), 2 * mesh.findMaxIndexDiff() + 1);
	K.factN = mesh.nodeCount * 2;
	std::cout << "K size: " << K.size() << "\n";
	std::cout << "Strip width: " << 2 * K.width() + 1 << "\n\n";

	double solvingTime = -omp_get_wtime();

	if (m.h == 0.)
		calcThickness_();

	std::cout << "Gradients calculation...";
	calcBs_();

	std::cout << "\rInitialization...       ";
	SparseSLAE spK(mesh, 2);
	uv.resize(2 * mesh.nodeCount, 0.);
	initConditions_(uv.data(), kinMask, spK.rp, lineCount, lines, lineRows);
	//ConjGradSolver cjs(spK);
	//CGV2 cjs(spK, uv.data(), kinMask);

	std::cout << "\rMatrix forming...";
	fillGlobalStiffness(K, spK);

	//JCGV cjs(spK, uv.data(), kinMask, preconditioning, false, false, 2, lines, lineRows, lineCount);
	JCGV cjs(spK.N, spK.data, (unsigned*)spK.rows, (unsigned*)spK.cols, spK.rp, \
		uv.data(), kinMask, preconditioning, false, false, 2, lines, lineRows, lineCount);

	std::cout << "\rSLAE solving...   ";
	size_t insideIter = 0;
	//cjs.solve(uv.data(), kinMask, insideIter, 1e-7);
	//cjs.solve(insideIter, 1e-7);
	insideIter = cjs.solve(1e-7);

	std::cout << "\rStress calculation...";
	updateParameters_();

	solvingTime += omp_get_wtime();
	std::cout << "\rSolved               " \
		<< "\nSLAE iterations: " << insideIter \
		<< "\nTime: " << solvingTime << " s\n" << std::endl;
	plastSolved = false;

	delete[] h;
	h = nullptr;

	return solvingTime;
}

//DEBUG
static void saveArray(double* data, size_t size, const std::string fileName) {
	std::ofstream file(fileName, std::ios_base::out);
	file.precision(15);
	for (size_t i = 0; i < size; ++i)
		file << data[i] << " ";
	file.close();
}

//DEBUG
static double checkArray(double* data, size_t size, const std::string fileName) {
	std::ifstream file(fileName);
	double* refData = new double[size];
	for (size_t i = 0; i < size; ++i)
		file >> refData[i];

	double res = 0.;
	for (size_t i = 0; i < size; ++i) {
		double diff = fabs(data[i] - refData[i]);
		if (diff > res)
			res = diff;
	}

	delete[] refData;
	return res;
}


double PlasticitySolver::solveCPU_() {
	std::cout << "Solving..." << std::endl;
	StripSLAE stripK(2 * mesh.nodeCount, 2 * mesh.findMaxIndexDiff() + 1);
	//StripSLAE stripK;
	stripK.factN = 2 * mesh.nodeCount;
	std::cout << "K size: " << stripK.size() << "\n";
	std::cout << "Strip width: " << 2 * stripK.width() + 1 << "\n";

	double solvingTime = -omp_get_wtime();

	if (m.h == 0.)
		calcThickness_();
	calcBs_();
	
	calcBs();
	std::cout << mesh.elemInfo[0].elemCount << " " << mesh.elemInfo[0].intPointsCount \
		<< " " << mesh.elemInfo[0].nodeDim << " " << mesh.elemInfo[0].nodeCount << "\n";
	/*for (unsigned e = 0; e < mesh.elemInfo[0].elemCount; ++e) {
		for (unsigned p = 0; p < mesh.elemInfo[0].intPointsCount; ++p) {
			for (unsigned i = 0; i < mesh.elemInfo[0].nodeDim; ++i) {
				for (unsigned j = 0; j < mesh.elemInfo[0].nodeCount; ++j)
					printf("%10.5f ", B4[e * 4 + p](i, j));
				std::cout << "\n";
			}
			std::cout << "\n";
			for (unsigned i = 0; i < mesh.elemInfo[0].nodeDim; ++i) {
				for (unsigned j = 0; j < mesh.elemInfo[0].nodeCount; ++j)
					printf("%10.5f ", B[(e * 4 + p) * 8 + i * 4 + j]);
				std::cout << "\n";
			}
			std::cout << "\n\n";
		}
	}*/

	uv.resize(2 * mesh.nodeCount, 0.);
	SparseSLAE spK(mesh, 2);
	initConditions_(uv.data(), kinMask, spK.rp, lineCount, lines, lineRows);

	mesh.checkNodeAdjStruct();
	//mesh.nodeAdjStruct.print();
	//CSR K(mesh.nodeAdjStruct, 2);
	CSR K(mesh.nodeAdjStruct, mesh.elemTypes, mesh.elemInfo, mesh.elem);
	//K.printStruct();

	//ConjGradSolver cjs(spK);

	//ConjGradSolver3 cjs(spK, uv.data());
	//PCG cjs(spK, uv.data(), kinMask);
	//CGV2 cjs(spK, uv.data(), kinMask);
	//BJCG2 cjs(spK, uv.data(), kinMask);
	//JCGV_ cjs(spK, uv.data(), kinMask, 2, lines, lineRows, lineCount);
	//JCGV cjs(spK, uv.data(), kinMask, preconditioning, true, false, 2, lines, lineRows, lineCount);
	//JCGV cjs(spK.N, spK.data, (unsigned*)spK.rows, (unsigned*)spK.cols, spK.rp, \
		uv.data(), kinMask, preconditioning, true, false, 2, lines, lineRows, lineCount);
	JCGV cjs((unsigned)spK.N, spK.data, (unsigned*)spK.rows, (unsigned*)spK.cols, spK.rp, \
		uv.data(), kinMask, preconditioning, true, false, 2, lines, lineRows, lineCount);
	//cjs.setLinesDim(2);

	//TEMP
	//cgInitCPU(spK.N, spK.data, (unsigned*)spK.rows, (unsigned*)spK.cols, spK.rp, uv.data(), preconditioning, true, false);
	//cgInitCuda(spK.N, spK.data, (unsigned*)spK.rows, (unsigned*)spK.cols, spK.rp, uv.data(), 0 * preconditioning, true, false);
	//cgInitCudaSD(spK.N, spK.data, (unsigned*)spK.rows, (unsigned*)spK.cols, spK.rp, uv.data(), preconditioning, true, true);

	//std::list<size_t> iterHistory;  //DEBUG

	size_t iterNum = 0;
	double relErr = 0.;
	do {
		fillGlobalStiffness(stripK, spK);
		//assemble(K);
		
		/*for (unsigned i = 0; i < 1000 + 0 * K.dataSize(); ++i) {
			printf("%10.5e  %10.5e\n", spK.data[i], K.data[i]);
		}*/
		//unsigned k = 0;
		//for (unsigned i = 0; i < K.N; ++i) {
		//	//std::cout << i << "\n";
		//	for (unsigned j = K.rows[i]; j < K.rows[i + 1]; ++j) {
		//		if (fabs(spK.data[j] - K.data[j]) > 1e-5)
		//			printf("%5d %5d %15.5e %15.5e\n", i, K.cols[j], spK.data[j], K.data[j]);
		//		++k;
		//	}
		//	//if (k > 1000) break;
		//}

		//TEMP
		/*switch (iterNum) {
		case 0: spK.saveToFile("slae0.txt"); break;
		case 1: spK.saveToFile("slae1.txt"); break;
		case 2: spK.saveToFile("slae2.txt"); break;
		}*/

		//size_t insideIter = 0;
		unsigned insideIter = 0;
		//cjs.solve2(uv.data(), kinMask, insideIter, 1e-7);
		//cjs.solve(insideIter, 1e-7);
		insideIter = cjs.solve(1e-7);
		//cgSolveCPU(1e-7, &insideIter);
		//cgSolveCuda(1e-7, &insideIter);
		//cgSolveCudaSD(1e-5, 1e-7, &insideIter);
		//cgSolveCudaSDNG(1e-5, 1e-7, &insideIter);

		updateParameters_(iterNum);
		relErr = exitCondition();
		printIter(iterNum, insideIter, relErr);

		//iterHistory.push_back(insideIter); //DEBUG

		++iterNum;
		//if (iterNum >= 50) break;
	} while (relErr > 1e-5);
	//} while (iterNum < 3);
	calcPlastDeform();
	solvingTime += omp_get_wtime();
	std::cout << "\n\rIterations: " << iterNum << "                                             " \
		<< "\nExit error: " << exitCondition() \
		<< "\nTime: " << solvingTime << " s\n" << std::endl;

	//std::cin.get();

	//DEBUG
	/*for (size_t iter : iterHistory)
		std::cout << iter << ",";
	std::cout << "\n";*/

	//DEBUG
	//saveArray(uv.data(), uv.size(), "../data/refDispls.txt");
	//std::cout << "\nresidual: " << checkArray(uv.data(), uv.size(), "../data/refDispls.txt") << "\n";

	//TEMP
	//cgFreeCPU();
	//cgFreeCuda();
	//cgFreeCudaSD();

	plastSolved = true;

	delete[] h;
	h = nullptr;

	return solvingTime;
}


template<typename fp>
__global__ static void initPlastDouble(fp* psi, fp* E_c, fp* nu_c, double E, double nu) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	psi[i] = (fp)1.;
	E_c[i] = (fp)E;
	nu_c[i] = (fp)nu;
}

__global__ static void initParams(double* psi, double* E_c, double* nu_c, double E, double nu, double* intS, double* tableS) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	psi[i] = 1.;
	E_c[i] = E;
	nu_c[i] = nu;
	intS[i] = {};
	tableS[i] = {};
}

void PlasticitySolver::initPlastParams() {
	int elemMemLen = (mesh.elemCount() + BS - 1) / BS * BS;
	cudaMemset(dd_E_c, 0, elemMemLen * sizeof(double));
	cudaMemset(dd_nu_c, 0, elemMemLen * sizeof(double));
	cudaMemset(dd_psi, 0, elemMemLen * sizeof(double));
	/*cudaMemset(df_E_c, 0, elemMemLen * sizeof(float));
	cudaMemset(df_nu_c, 0, elemMemLen * sizeof(float));
	cudaMemset(df_psi, 0, elemMemLen * sizeof(float));*/
	initPlastDouble<<<elemMemLen / BS, BS>>>(dd_psi, dd_E_c, dd_nu_c, m.E, m.nu);
	cudaMemset(dd_intensityS, 0, elemMemLen * sizeof(double));
	cudaMemset(dd_tableS, 0, elemMemLen * sizeof(double));
	/*cudaMemset(df_intensityS, 0, elemMemLen * sizeof(float));
	cudaMemset(df_tableS, 0, elemMemLen * sizeof(float));*/
	//initParams<<<elemMemLen / BS, BS>>>(dd_psi, dd_E_c, dd_nu_c, m.E, m.nu, dd_intensityS, dd_tableS);

	/*cudaMemset(dd_exx, 0, elemMemLen * sizeof(double));
	cudaMemset(dd_eyy, 0, elemMemLen * sizeof(double));
	cudaMemset(dd_gamma, 0, elemMemLen * sizeof(double));
	cudaMemset(dd_intensityE, 0, elemMemLen * sizeof(double));
	cudaMemset(dd_sxx, 0, elemMemLen * sizeof(double));
	cudaMemset(dd_syy, 0, elemMemLen * sizeof(double));
	cudaMemset(dd_tau, 0, elemMemLen * sizeof(double));*/

	cudaMemset(dd_uv, 0, (2 * mesh.nodeCount + BS - 1) / BS * BS * sizeof(double));
	//cudaMemset(df_uv, 0, (2 * mesh.nodeCount + BS - 1) / BS * BS * sizeof(float));
}

void PlasticitySolver::initPlastParamsFloat() {
	int elemMemLen = (mesh.elemCount() + BS - 1) / BS * BS;
	cudaMemset(df_E_c, 0, elemMemLen * sizeof(float));
	cudaMemset(df_nu_c, 0, elemMemLen * sizeof(float));
	cudaMemset(df_psi, 0, elemMemLen * sizeof(float));
	initPlastDouble<<<elemMemLen / BS, BS>>>(df_psi, df_E_c, df_nu_c, m.E, m.nu);
	cudaMemset(df_intensityS, 0, elemMemLen * sizeof(float));
	cudaMemset(df_tableS, 0, elemMemLen * sizeof(float));
	cudaMemset(df_uv, 0, (2 * mesh.nodeCount + BS - 1) / BS * BS * sizeof(float));
}


template<typename fp>
static __device__ void formC_planeStressIsotropic(fp* C, fp E, fp nu) {
	//Плоское напряжённое состояние
	double coef = E / (1. - nu * nu);
	C[0] = C[4] = coef;
	C[1] = C[3] = nu * coef;
	C[8] = (1. - nu) * 0.5 * coef;
	C[2] = C[5] = C[6] = C[7] = 0.;
}


template<typename fp>
static __device__ void formC_planeStrainIsotropic(fp* C, fp E, fp nu) {
	//Плоское деформированное состояние
	fp coef = E * (1. - nu) / ((1. + nu) * (1 - 2. * nu));
	C[0] = C[4] = 1. * coef;
	C[1] = C[3] = nu / (1. - nu) * coef;
	C[8] = (1. - 2. * nu) * 0.5 / (1. - nu) * coef;
	C[2] = C[5] = C[6] = C[7] = 0.;
}


template<typename fp>
static __device__ void formK_planeStress_sym(fp* Ke, unsigned nodeCount, const fp* C_, const fp* B, const fp* detJ, const fp* intCoefs, const fp* h, unsigned intPointsCount) {
	for (unsigned p = 0; p < intPointsCount; ++p) {
		fp BC[2][3] = {};
		fp coef = intCoefs[p] * detJ[p] * h[p];
		fp CC[3][3];
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


template<typename fp>
static __device__ void formK_planeStrain_sym(fp* Ke, unsigned nodeCount, const fp* C_, const fp* B, \
	const fp* detJ, const fp* intCoefs, unsigned intPointsCount) {
	for (unsigned p = 0; p < intPointsCount; ++p) {
		fp BC[2][3] = {};
		fp coef = intCoefs[p] * detJ[p];
		fp CC[3][3];
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


// sym Ke
template<typename fp>
static __global__ void assemble_quad1_planeStress(
	unsigned count, unsigned elemType,
	fp* data, const unsigned* rows,
	const unsigned* elemIns, const unsigned* elemInsIdx, 
	const unsigned* elemInsSize, const unsigned* nodeLine,
	const unsigned* colorMap,
	const fp* B, const fp* detJ, fp* C, const fp* E, const fp* nu,
	const unsigned* elem,
	unsigned intPointsCount, const fp* intCoefs, const fp* h) {

	unsigned e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= count) return;

	e = colorMap[e];

	formC_planeStressIsotropic(C + e * 9, E[e], nu[e]);
	fp Ke[36] = {};
	formK_planeStress_sym(Ke, 4, C + e * 9,
		B + e * 8 * intPointsCount,
		detJ + e * intPointsCount,
		intCoefs,
		h + e * intPointsCount,
		intPointsCount);
	cudaCSR_insertKeSym(data, rows, elemIns, elemInsIdx, elemInsSize, nodeLine, \
		elemType, e, Ke, elem + 4 * e, 4, 2);
}


// sym Ke
template<typename fp>
static __global__ void assemble_quad1_planeStrain(
	unsigned count, unsigned elemType,
	fp* data, const unsigned* rows,
	const unsigned* elemIns, const unsigned* elemInsIdx,
	const unsigned* elemInsSize, const unsigned* nodeLine,
	const unsigned* colorMap,
	const fp* B, const fp* detJ, fp* C, const fp* E, const fp* nu,
	const unsigned* elem,
	unsigned intPointsCount, const fp* intCoefs) {

	unsigned e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= count) return;

	e = colorMap[e];

	formC_planeStrainIsotropic(C + e * 9, E[e], nu[e]);
	fp Ke[36] = {};
	formK_planeStrain_sym(Ke, 4, C + e * 9,
		B + e * 8 * intPointsCount,
		detJ + e * intPointsCount,
		intCoefs,
		intPointsCount);
	cudaCSR_insertKeSym(data, rows, elemIns, elemInsIdx, elemInsSize, nodeLine, \
		elemType, e, Ke, elem + 4 * e, 4, 2);
}


// sym Ke
template<typename fp>
static __global__ void assemble_quad2_planeStress(
	unsigned count, unsigned elemType,
	fp* data, const unsigned* rows,
	const unsigned* elemIns, const unsigned* elemInsIdx,
	const unsigned* elemInsSize, const unsigned* nodeLine,
	const unsigned* colorMap,
	const fp* B, const fp* detJ, fp* C, const fp* E, const fp* nu,
	const unsigned* elem,
	unsigned intPointsCount, const fp* intCoefs, const fp* h) {

	unsigned e = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("%d\n", e);
	if (e >= count) return;

	e = colorMap[e];

	formC_planeStressIsotropic(C + e * 9, E[e], nu[e]);
	fp Ke[136] = {};
	formK_planeStress_sym(Ke, 8, C + e * 9,
		B + e * 16 * intPointsCount,
		detJ + e * intPointsCount,
		intCoefs,
		h + e * intPointsCount,
		intPointsCount);
	cudaCSR_insertKeSym(data, rows, elemIns, elemInsIdx, elemInsSize, nodeLine, \
		elemType, e, Ke, elem + 8 * e, 8, 2);
}


// sym Ke
template<typename fp>
static __global__ void assemble_quad2_planeStrain(
	unsigned count, unsigned elemType,
	fp* data, const unsigned* rows,
	const unsigned* elemIns, const unsigned* elemInsIdx,
	const unsigned* elemInsSize, const unsigned* nodeLine,
	const unsigned* colorMap,
	const fp* B, const fp* detJ, fp* C, const fp* E, const fp* nu,
	const unsigned* elem,
	unsigned intPointsCount, const fp* intCoefs) {

	unsigned e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= count) return;

	e = colorMap[e];

	formC_planeStrainIsotropic(C + e * 9, E[e], nu[e]);
	fp Ke[136] = {};
	formK_planeStrain_sym(Ke, 8, C + e * 9,
		B + e * 16 * intPointsCount,
		detJ + e * intPointsCount,
		intCoefs,
		intPointsCount);
	cudaCSR_insertKeSym(data, rows, elemIns, elemInsIdx, elemInsSize, nodeLine, \
		elemType, e, Ke, elem + 8 * e, 8, 2);
}


void PlasticitySolver::assemble(CudaCSR& K, unsigned* colorMap) {
	K.data.setZero(K.dataSize);
	for (unsigned c = 0; c < mesh.colorMap.colorCount; ++c) {

		for (unsigned t = 0; t < mesh.elemTypes; ++t) {
			const FiniteElement& type = mesh.elemInfo[t];

			// counts...
			unsigned* locColorMapIdx = mesh.colorMap.colorMapIdx + t * (mesh.colorMap.colorCount + 1);
			unsigned count = locColorMapIdx[c + 1] - locColorMapIdx[c];

			//std::cout << "count: " << count << "\n";

			unsigned grid, block;
			switch (type.elemType) {
			case ElemType::quad1:
				block = 256;
				grid = (count + block - 1) / block;
				switch (type.geomType) {
					case GeomType::planeStress:
						assemble_quad1_planeStress<double><<<grid, block>>>(
							count, t,
							K.data.raw(), K.rows.raw(), 
							K.elemIns.raw(), K.elemInsIdx.raw(), K.elemInsSize.raw(), K.nodeLine.raw(),
							colorMap + type.poolIdx + locColorMapIdx[c],
							dd_B + Bidx[t],
							dd_detJ + detJidx[t],
							(double*)dd_C, // + offset
							dd_E_c, dd_nu_c,
							cudaMesh.elem + type.memIdx,
							type.intPointsCount, cudaMesh.elemInfo[t].intCoefs.raw(),
							dd_hsj + hsjIdx[t]);
						break;
					case GeomType::planeStrain:
						assemble_quad1_planeStrain<double><<<grid, block>>>(
							count, t,
							K.data.raw(), K.rows.raw(),
							K.elemIns.raw(), K.elemInsIdx.raw(), K.elemInsSize.raw(), K.nodeLine.raw(),
							colorMap + type.poolIdx + locColorMapIdx[c],
							dd_B + Bidx[t],
							dd_detJ + detJidx[t],
							(double*)dd_C, // + offset
							dd_E_c, dd_nu_c,
							cudaMesh.elem + type.memIdx,
							type.intPointsCount, cudaMesh.elemInfo[t].intCoefs.raw());
						break;
					case GeomType::axiSym:

						break;
				}
				break;
			case ElemType::quad2:
				block = 256;
				grid = (count + block - 1) / block;
				switch (type.geomType) {
				case GeomType::planeStress:
					//std::cout << "assemble quad2  grid: " << grid << "\n";
					assemble_quad2_planeStress<double><<<grid, block>>>(
						count, t,
						K.data.raw(), K.rows.raw(),
						K.elemIns.raw(), K.elemInsIdx.raw(), K.elemInsSize.raw(), K.nodeLine.raw(),
						colorMap + type.poolIdx + locColorMapIdx[c],
						dd_B + Bidx[t],
						dd_detJ + detJidx[t],
						(double*)dd_C, // + offset
						dd_E_c, dd_nu_c,
						cudaMesh.elem + type.memIdx,
						type.intPointsCount, cudaMesh.elemInfo[t].intCoefs.raw(),
						dd_hsj + hsjIdx[t]);
					break;
				case GeomType::planeStrain:
					assemble_quad2_planeStrain<double><<<grid, block>>>(
						count, t,
						K.data.raw(), K.rows.raw(),
						K.elemIns.raw(), K.elemInsIdx.raw(), K.elemInsSize.raw(), K.nodeLine.raw(),
						colorMap + type.poolIdx + locColorMapIdx[c],
						dd_B + Bidx[t],
						dd_detJ + detJidx[t],
						(double*)dd_C, // + offset
						dd_E_c, dd_nu_c,
						cudaMesh.elem + type.memIdx,
						type.intPointsCount, cudaMesh.elemInfo[t].intCoefs.raw());
				case GeomType::axiSym:

					break;
				}
				break;
			}
		}
		cudaDeviceSynchronize();
		//std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";
	}
}


void PlasticitySolver::assembleFloat(CudaCSR& K, unsigned* colorMap) {
	K.data_f.setZero(K.dataSize);
	for (unsigned c = 0; c < mesh.colorMap.colorCount; ++c) {

		for (unsigned t = 0; t < mesh.elemTypes; ++t) {
			const FiniteElement& type = mesh.elemInfo[t];

			// counts...
			unsigned* locColorMapIdx = mesh.colorMap.colorMapIdx + t * (mesh.colorMap.colorCount + 1);
			unsigned count = locColorMapIdx[c + 1] - locColorMapIdx[c];

			unsigned grid, block;
			switch (type.elemType) {
			case ElemType::quad1:
				block = 256;
				grid = (count + block - 1) / block;
				switch (type.geomType) {
				case GeomType::planeStress:
					assemble_quad1_planeStress<float><<<grid, block>>>(
						count, t,
						K.data_f.raw(), K.rows.raw(),
						K.elemIns.raw(), K.elemInsIdx.raw(), K.elemInsSize.raw(), K.nodeLine.raw(),
						colorMap + type.poolIdx + locColorMapIdx[c],
						df_B + Bidx[t],
						df_detJ + detJidx[t],
						(float*)df_C, // + offset
						df_E_c, df_nu_c,
						cudaMesh.elem + type.memIdx,
						type.intPointsCount, cudaMesh.elemInfo[t].intCoefs_f.raw(),
						df_hsj + hsjIdx[t]);
					break;
				case GeomType::planeStrain:
					assemble_quad1_planeStrain<float><<<grid, block>>>(
						count, t,
						K.data_f.raw(), K.rows.raw(),
						K.elemIns.raw(), K.elemInsIdx.raw(), K.elemInsSize.raw(), K.nodeLine.raw(),
						colorMap + type.poolIdx + locColorMapIdx[c],
						df_B + Bidx[t],
						df_detJ + detJidx[t],
						(float*)df_C, // + offset
						df_E_c, df_nu_c,
						cudaMesh.elem + type.memIdx,
						type.intPointsCount, cudaMesh.elemInfo[t].intCoefs_f.raw());
					break;
				case GeomType::axiSym:

					break;
				}
				break;
			case ElemType::quad2:
				block = 256;
				grid = (count + block - 1) / block;
				switch (type.geomType) {
				case GeomType::planeStress:
					assemble_quad2_planeStress<float><<<grid, block>>>(
						count, t,
						K.data_f.raw(), K.rows.raw(),
						K.elemIns.raw(), K.elemInsIdx.raw(), K.elemInsSize.raw(), K.nodeLine.raw(),
						colorMap + type.poolIdx + locColorMapIdx[c],
						df_B + Bidx[t],
						df_detJ + detJidx[t],
						(float*)df_C, // + offset
						df_E_c, df_nu_c,
						cudaMesh.elem + type.memIdx,
						type.intPointsCount, cudaMesh.elemInfo[t].intCoefs_f.raw(),
						df_hsj + hsjIdx[t]);
					break;
				case GeomType::planeStrain:
					assemble_quad2_planeStrain<float><<<grid, block>>>(
						count, t,
						K.data_f.raw(), K.rows.raw(),
						K.elemIns.raw(), K.elemInsIdx.raw(), K.elemInsSize.raw(), K.nodeLine.raw(),
						colorMap + type.poolIdx + locColorMapIdx[c],
						df_B + Bidx[t],
						df_detJ + detJidx[t],
						(float*)df_C, // + offset
						df_E_c, df_nu_c,
						cudaMesh.elem + type.memIdx,
						type.intPointsCount, cudaMesh.elemInfo[t].intCoefs_f.raw());
				case GeomType::axiSym:

					break;
				}
				break;
			}
		}
		cudaDeviceSynchronize();
	}
}


double PlasticitySolver::solveLinearCUDA() {
	std::cout << "Solving..." << std::endl;
	int memLen = (2 * mesh.nodeCount + BS - 1) / BS * BS;
	//std::cout << "UV: " << memLen << "\n";
	CudaSLAE<double> K(memLen, 2 * mesh.nodeCount, 2 * mesh.findMaxIndexDiff() + 1);
	std::cout << "K size: " << K.size() << "\n";
	std::cout << "Strip width: " << 2 * K.width() + 1 << "\n\n";

	double solvingTime = -omp_get_wtime();

	if (m.h == 0.) {
		calcThickness_();
		copyThickness_();
	}
	std::cout << "Gradients calculation...";
	calcBsCuda_();

	std::cout << "\rInitialization...       ";
	initPlastParams();
	//std::clog << "log1\n";
	CudaSparseSLAE<double> spK(mesh, 2);
	//std::clog << "log2\n";
	initConditions_(spK, dd_uv);
	//std::clog << "log3\n";
	//
	//ConjGradCuda2<double> cjc(spK);
	//ConjGradCudaGW<double> cjc(spK, dd_uv, dev_kinNodes);
	//ConjGradCudaV2<double> cjc(spK, dd_uv, dev_kinNodes);
	

	//delete[] h;
	//h = nullptr;
	
	std::cout << "\rMatrix forming...";
	fillGlobalStiffness(K, spK);
	//K.print();
	//spK.copy(K);
	//std::cout << "hash " << spK.rpHash() << "\n";
	//std::cout << "hash " << spK.matrixHash() << "\n";

	std::cout << "\rSLAE solving...   ";
	CudaJCGV<double> cjc((unsigned)spK.N, spK.data, (unsigned*)spK.rows, (unsigned*)spK.cols, spK.rp, \
		dd_uv, dev_kinNodes, preconditioning, false, false, 1, 512, \
		dd_lines, dev_lineRows, lineCount);
	size_t insideIter = 0;
	//cjc.solve(dd_uv, dev_kinNodes, insideIter, 1e-7);
	insideIter = cjc.solve(1e-7);

	std::cout << "\rStress calculation...";
	updateParamDouble_();
	
	solvingTime += omp_get_wtime();
	ramSaved = false;
	std::cout << "\rSolved               " \
		<< "\nSLAE iterations: " << insideIter \
		<< "\nTime: " << solvingTime << " s\n" << std::endl;
	plastSolved = false;

	cudaFree(dd_h);
	dd_h = nullptr;

	return solvingTime;
}


double PlasticitySolver::solveCUDA_() {
	std::cout << "Solving..." << std::endl;
	CudaSLAE<double> K(uv.size(), 2 * mesh.nodeCount, 2 * mesh.findMaxIndexDiff() + 1);
	std::cout << "K size: " << K.size() << "\n";
	std::cout << "Strip width: " << 2 * K.width() + 1 << "\n";
	mesh.meshToGPU();
	//std::cout << "log1\n";

	double solvingTime = -omp_get_wtime();

	if (m.h == 0.) {
		calcThickness_();
		copyThickness_();
	}
	//std::cout << "log2\n";
	calcBsCuda_();
	//std::cout << "log3\n";
	initPlastParams();
	//std::cout << "log4\n";

	CudaSparseSLAE<double> spK(mesh, 2);
	//initConditions_(spK, dd_uv);
	cudaInitConditions(dd_uv, dev_kinNodes, spK.rp, lineCount, dd_lines, dev_lineRows);
	//std::cout << "log5\n";
	
	delete[] h;
	h = nullptr;

	//std::cout << "log6\n";

	//ConjGradCuda<double> cjc(spK);
	//ConjGradCuda2<double> cjc(spK);
	//ConjGradCudaGW<double> cjc(spK, dd_uv, dev_kinNodes);
	//ConjGradCudaV2<double> cjc(spK, dd_uv, dev_kinNodes);
	CudaJCGV<double> cjc((unsigned)spK.N, spK.data, (unsigned*)spK.rows, (unsigned*)spK.cols, spK.rp, \
		dd_uv, dev_kinNodes, preconditioning, true, false, 1, 512, \
		dd_lines, dev_lineRows, lineCount);

	//std::cout << "log7\n";

	size_t iterNum = 0;
	double relErr = 0.;
	do {
		fillGlobalStiffness(K, spK);
		//if (iterNum == 0) std::cout << "hash " << spK.matrixHash() << " " << spK.rpHash() << "\n";

		size_t insideIter = 0;
		//cjc.solve(dd_uv, dev_kinNodes, insideIter, 1e-7);
		insideIter = cjc.solve(1e-7);
		//insideIter = cjc.solveDebug(1e-7);

		updateParamDouble_();

		relErr = exitCondition(dd_intensityS, dd_tableS, dd_exit);
		printIter(iterNum, insideIter, relErr);
		//std::cout << "hash " << spK.matrixHash() << "";
		++iterNum;
		//if (iterNum > 1) break;
	} while (relErr > 1e-5);
	calcPlastDeformCuda();
	solvingTime += omp_get_wtime();
	ramSaved = false;
	std::cout << "\n\rIterations: " << iterNum << "                                             " \
		<< "\nExit error: " << relErr \
		<< "\nTime: " << solvingTime << " s\n" << std::endl;
	plastSolved = true;

	cudaFree(dd_h);
	dd_h = nullptr;

	return solvingTime;
}


double PlasticitySolver::solveCUDA() {
	std::cout << "Solving on GPU..." << (iterOutput ? "\n" : "") << std::endl;
	const unsigned blockSize = 512;

	double solvingTime = -omp_get_wtime();
	
	calcThickness();
	hostToDevice(dd_hsj, hsj, hsjIdx[mesh.elemTypes]);

	// TODO: construct Bs graph
	calcBsCuda();
	initPlastParams();

	mesh.checkNodeAdjStruct();
	CudaCSR K(mesh.nodeAdjStruct, mesh.elemTypes, mesh.elemInfo, mesh.elem, blockSize);
	K.mallocDouble();

	mesh.checkColorMap();
	dptr<unsigned> dev_colorMap(mesh.elemCount());
	hostToDevice(dev_colorMap, mesh.colorMap.colorMap, mesh.elemCount());

	// TODO: construct assemble graph

	unsigned memLen = (K.N + blockSize - 1) / blockSize * blockSize;
	dd_u.realloc(memLen);
	dptr<double> R(memLen);

	cudaInitConditions(dd_u, dev_kinNodes, R, lineCount, dd_lines, dev_lineRows);

	CudaJCGV<double> cgSolver(K.N, K.data, K.rows, K.cols, R, \
		dd_u, dev_kinNodes, preconditioning, true, false, 2, blockSize, \
		dd_lines, dev_lineRows, lineCount);

	unsigned iterNum = 0;
	double relErr = 0.;
	do {
		assemble(K, dev_colorMap);

		unsigned slaeIter = 0;
		slaeIter = cgSolver.solve(1e-7);

		updateParamDouble();

		relErr = exitCondition(dd_intensityS, dd_tableS, dd_exit);
		printIter(iterNum, slaeIter, relErr);

		++iterNum;
	} while (relErr > 1e-5);

	calcPlastDeformCuda();

	solvingTime += omp_get_wtime();

	std::cout << (iterOutput ? "\n" : "") << "Solved [" << solvingTime << " sec]\n";
	if (!iterOutput)
		std::cout << "Iterations: " << iterNum << "                                             " \
		<< "\nExit error: " << exitCondition() << "\n";

	ramSaved = false;
	plastSolved = true;

	return solvingTime;
}


template<int size>
__global__ static void copyB_(StaticMatrix<2, size, float>* B_f, StaticMatrix<2, size, double>* B, int count) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < count)
#pragma unroll
		for (int j = 0; j < 2 * size; ++j)
			B_f[i].data[j] = (float)B[i].data[j];
	//if (i == 5) \
		B_f[i].print();
}

__global__ static void copyJ_(float* detJ_f, double* detJ, int count, const int points) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < count)
#pragma unroll
		for (int j = points * i; j < points * (i + 1); ++j)
			detJ_f[j] = (float)detJ[j];
}

void PlasticitySolver::copyBs_() {
	int grid4b = (mesh.count4 * 4 + BS - 1) / BS,
		grid4j = (mesh.count4 + BS - 1) / BS,
		grid8b = (mesh.count8 * secIntPs2 + BS - 1) / BS,
		grid8j = (mesh.count8 + BS - 1) / BS;
	if (mesh.count4) {
		copyB_<<<grid4b, BS>>>(df_B4, dd_B4, mesh.count4 * 4);
		copyJ_<<<grid4j, BS>>>(df_detJ4, dd_detJ4, mesh.count4, 4);
	}
	if (mesh.count8) {
		copyB_<<<grid8b, BS>>>(df_B8, dd_B8, mesh.count8 * secIntPs2);
		copyJ_<<<grid8j, BS>>>(df_detJ8, dd_detJ8, mesh.count8, secIntPs2);
	}
	cudaDeviceSynchronize();
}

void PlasticitySolver::copyBs() {
	const unsigned block = 256;
	doubleToFloat<<<(Bidx[mesh.elemTypes] + block - 1) / block, block>>>(df_B, dd_B, Bidx[mesh.elemTypes]);
	doubleToFloat<<<(detJidx[mesh.elemTypes] + block - 1) / block, block>>>(df_detJ, dd_detJ, detJidx[mesh.elemTypes]);
	cudaDeviceSynchronize();
}

__global__ static void copyCond(float* R_f, double* R, float* u_f, double* u) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	R_f[i] = (float)R[i];
	u_f[i] = (float)u[i];
}

__global__ static void copyLines2(unsigned lineCount, float* df_lines, double* dd_lines) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= lineCount) return;
	df_lines[2 * i] = (float)dd_lines[2 * i];
	df_lines[2 * i + 1] = (float)dd_lines[2 * i + 1];
}

void PlasticitySolver::copyConditions_(float* R_f, double* R, unsigned memLen) {
	const unsigned block = 512;
	copyCond<<<memLen / block, block>>>(R_f, R, df_uv, dd_uv);
	if (lineCount) {
		df_lines.realloc(2 * lineCount);
		const unsigned linesBlock = 32;
		unsigned linesGrid = (lineCount + linesBlock - 1) / linesBlock;
		copyLines2<<<linesGrid, linesBlock>>>(lineCount, df_lines, dd_lines);
	}
}

void PlasticitySolver::copyConditions(float* R_f, double* R, unsigned memLen) {
	const unsigned block = 512;
	copyCond<<<memLen / block, block>>>(R_f, R, df_u, dd_u);
	if (lineCount) {
		df_lines.realloc(2 * lineCount);
		const unsigned linesBlock = 32;
		unsigned linesGrid = (lineCount + linesBlock - 1) / linesBlock;
		copyLines2<<<linesGrid, linesBlock>>>(lineCount, df_lines, dd_lines);
	}
	cudaDeviceSynchronize();
}

void static copySpK(CudaSparseSLAE<float>& f, CudaSparseSLAE<double>& d) {
	/*int N = 0; //размер СЛАУ
	int memLen = 0;
	int dim = 1;
	fp* rp = nullptr;
	fp* data = nullptr;
	int* rows = nullptr;
	int* cols = nullptr;
	int* adj = nullptr;
	int* adjPos = nullptr;*/
	f.N = d.N;
	f.memLen = d.memLen;
	f.dim = d.dim;
	cudaMalloc((void**)&f.rp, f.memLen * sizeof(float));

	int dataSize = 0;
	cudaMemcpy(&dataSize, d.rows + d.N, sizeof(int), cudaMemcpyDeviceToHost);
	//std::cout << "\n---------- " << dataSize << "\n";
	cudaMalloc((void**)&f.data, dataSize * sizeof(float));
	cudaMalloc((void**)&f.rows, (f.memLen + 1) * sizeof(int));
	cudaMalloc((void**)&f.cols, dataSize * sizeof(int));
	cudaMalloc((void**)&f.adjPos, (f.memLen / f.dim + 1) * sizeof(int));
	cudaMalloc((void**)&f.adj, dataSize / (f.dim * f.dim) * sizeof(int));
	cudaMemcpy(f.rows, d.rows, (f.memLen + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(f.cols, d.cols, dataSize * sizeof(int), cudaMemcpyDeviceToDevice);
	cudaMemcpy(f.adjPos, d.adjPos, (f.memLen / f.dim + 1) * sizeof(int), cudaMemcpyDeviceToDevice);
	//cudaMemcpy(f.adj, d.adj, dataSize / (f.dim * f.dim) * sizeof(int), cudaMemcpyDeviceToDevice);
}

__global__ static void copyElemData_(int count, \
	float* exx_f, float* eyy_f, float* gamma_f, float* intE_f, \
	float* sxx_f, float* syy_f, float* tau_f, float* intS_f, float* tableS_f, \
	float* E_c_f, float* nu_c_f, float* psi_f, \
	double* exx, double* eyy, double* gamma, double* intE, \
	double* sxx, double* syy, double* tau, double* intS, double* tableS, \
	double* E_c, double* nu_c, double* psi) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < count) {
		//exx[i] = (float)exx_f[i];
		//eyy[i] = (float)eyy_f[i];
		//gamma[i] = (float)gamma_f[i];
		//intE[i] = (float)intE_f[i];
		//sxx[i] = (float)sxx_f[i];
		//syy[i] = (float)syy_f[i];
		//tau[i] = (float)tau_f[i];
		//intS[i] = (float)intS_f[i];
		//tableS[i] = (float)tableS_f[i];
		E_c[i] = (float)E_c_f[i];
		nu_c[i] = (float)nu_c_f[i];
		//psi[i] = (float)psi_f[i];
	}
}

__global__ static void copyElemData(int count, \
	float* E_c_f, float* nu_c_f, \
	double* E_c, double* nu_c) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= count) return;

	E_c[i] = (float)E_c_f[i];
	nu_c[i] = (float)nu_c_f[i];
}

__global__ static void copyUv(float* uv_f, double* uv, bool* kinNodes) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (kinNodes[i])
		uv[i] = (double)uv_f[i];
}

__global__ static void refineLines2(unsigned lineCount, double* lines, unsigned* lineRows, double* uv) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= lineCount) return;
	vec2 uk = *reinterpret_cast<vec2*>(uv + lineRows[i]);
	double len = uk.norm();
	uk = *reinterpret_cast<vec2*>(lines + 2 * i) * len;
	*reinterpret_cast<vec2*>(uv + lineRows[i]) = uk;
}

void PlasticitySolver::copyFloatToDouble_(int memLen) {
	copyUv<<<memLen / BS, BS>>>(df_uv, dd_uv, dev_kinNodes);
	int elemGrid = (mesh.elemCount() + BS - 1) / BS;
	copyElemData_<<<elemGrid, BS>>>(mesh.elemCount(), df_exx, df_eyy, df_gamma, df_intensityE, \
		df_sxx, df_syy, df_tau, df_intensityS, df_tableS, df_E_c, df_nu_c, df_psi, \
		dd_exx, dd_eyy, dd_gamma, dd_intensityE, \
		dd_sxx, dd_syy, dd_tau, dd_intensityS, dd_tableS, dd_E_c, dd_nu_c, dd_psi);
	if (lineCount) {
		df_lines.free();
		const unsigned linesBlock = 32;
		unsigned linesGrid = (lineCount + linesBlock - 1) / linesBlock;
		refineLines2<<<linesGrid, linesBlock>>>(lineCount, dd_lines, dev_lineRows, dd_uv);
	}
	cudaMemset(dd_intensityS, 0, elemGrid * BS * sizeof(double));
	cudaMemset(dd_tableS, 0, elemGrid * BS * sizeof(double));
	cudaDeviceSynchronize();
}


void PlasticitySolver::copyFloatToDouble(unsigned memLen) {
	const unsigned  block = 1024;
	copyUv<<<memLen / block, block>>>(df_uv, dd_uv, dev_kinNodes);
	unsigned elemGrid = (mesh.elemCount() + block - 1) / block;
	copyElemData<<<elemGrid, block>>>(mesh.elemCount(), df_E_c, df_nu_c, \
		dd_E_c, dd_nu_c);
	if (lineCount) {
		df_lines.free();
		const unsigned linesBlock = 32;
		unsigned linesGrid = (lineCount + linesBlock - 1) / linesBlock;
		refineLines2<<<linesGrid, linesBlock>>>(lineCount, dd_lines, dev_lineRows, dd_uv);
	}
	cudaMemset(dd_intensityS, 0, elemGrid * block * sizeof(double));
	cudaMemset(dd_tableS, 0, elemGrid * block * sizeof(double));
	cudaDeviceSynchronize();
}


double PlasticitySolver::solveCUDA_FD_() {
	std::cout << "Solving..." << std::endl;
	CudaSLAE<float> K_f(uv.size(), 2 * mesh.nodeCount, 2 * mesh.findMaxIndexDiff() + 1);
	
	std::cout << "K size: " << K_f.size() << "\n";
	std::cout << "Strip width: " << 2 * K_f.width() + 1 << "\n";

	double solvingTime = -omp_get_wtime();

	//std::cout << "Initialization...";
	if (m.h == 0.) {
		calcThickness_();
		copyThickness_();
	}
	calcBsCuda_();
	copyBs_();
	initPlastParamsFloat();

	CudaSparseSLAE<double> spK(mesh, 2);
	CudaSparseSLAE<float> spK_f;
	copySpK(spK_f, spK);
	//TO DO: copy struct (done?)

	//init in double, copy to float (done?)
	//initConditions_(spK, dd_uv);
	cudaInitConditions(dd_uv, dev_kinNodes, spK.rp, lineCount, dd_lines, dev_lineRows);
	copyConditions_(spK_f.rp, spK.rp, spK.memLen);

	delete[] h;
	h = nullptr;

	//ConjGradCuda2<float> cjc_f(spK_f);
	//ConjGradCudaGW<float> cjc_f(spK_f, df_uv, dev_kinNodes);
	//ConjGradCudaV2<float> cjc_f(spK_f, df_uv, dev_kinNodes);
	CudaJCGV<float> cjc_f((unsigned)spK.N, spK_f.data, (unsigned*)spK.rows, (unsigned*)spK.cols, spK_f.rp, \
		df_uv, dev_kinNodes, preconditioning, true, false, 1, 512, \
		df_lines, dev_lineRows, lineCount);

	size_t iterNum = 0;
	float relErr_f = 0.f;
	do {
		fillGlobalStiffness(K_f, spK_f);//TO DO: fix floats (done?)

		//std::cout << "hash " << spK_f.matrixHash() << " " << spK_f.rpHash() << "\n";

		size_t insideIter = 0;
		//cjc_f.solve(df_uv, dev_kinNodes, insideIter, 2e-4f);
		insideIter = cjc_f.solve(2e-4f);

		updateParamFloat_();//TO DO (done?)
		relErr_f = exitCondition(df_intensityS, df_tableS, df_exit);
		printIter(iterNum, insideIter, (double)relErr_f);
		++iterNum;
	} while (relErr_f > 1e-4f);

	if (iterOutput) std::cout << "\nCopying from float to double...";

	K_f.~CudaSLAE();
	CudaSLAE<double> K(uv.size(), 2 * mesh.nodeCount, 2 * mesh.findMaxIndexDiff() + 1);

	//TO DO: copy ALL data
	copyFloatToDouble_(spK.memLen);

	//cjc_f.~ConjGradCuda();
	//ConjGradCuda2<double> cjc(spK);

	//cjc_f.~ConjGradCudaGW();
	//cjc_f.~ConjGradCudaV2();
	//ConjGradCudaGW<double> cjc(spK, dd_uv, dev_kinNodes);
	//ConjGradCudaV2<double> cjc(spK, dd_uv, dev_kinNodes);
	CudaJCGV<double> cjc((unsigned)spK.N, spK.data, (unsigned*)spK.rows, (unsigned*)spK.cols, spK.rp, \
		dd_uv, dev_kinNodes, preconditioning, true, false, 1, 512, \
		dd_lines, dev_lineRows, lineCount);

	double relErr = 0.;
	do {
		fillGlobalStiffness(K, spK);

		//std::cout << "hash " << spK.matrixHash() << " " << spK.rpHash() << "\n";

		size_t insideIter = 0;
		//cjc.solve(dd_uv, dev_kinNodes, insideIter, 1e-7);
		insideIter = cjc.solve(1e-7);

		updateParamDouble_();
		relErr = exitCondition(dd_intensityS, dd_tableS, dd_exit);
		printIter(iterNum, insideIter, relErr);
		++iterNum;
	} while (relErr > 1e-5);

	calcPlastDeformCuda();
	solvingTime += omp_get_wtime();
	ramSaved = false;
	std::cout << "\n\rIterations: " << iterNum << "                                             " \
		<< "\nExit error: " << relErr \
		<< "\nTime: " << solvingTime << " s\n" << std::endl;
	plastSolved = true;

	cudaFree(dd_h);
	cudaFree(df_h);
	dd_h = nullptr;
	df_h = nullptr;

	return solvingTime;
}


double PlasticitySolver::solveCUDA_FD() {
	std::cout << "Solving on GPU..." << (iterOutput ? "\n" : "") << std::endl;
	const unsigned blockSize = 512;

	double solvingTime = -omp_get_wtime();

	cudaMesh.addFloat(mesh);

	df_hsj.realloc(hsjIdx[mesh.elemTypes]);
	df_B.realloc(Bidx[mesh.elemTypes]);
	df_detJ.realloc(detJidx[mesh.elemTypes]);

	calcThickness();
	hostToDevice(dd_hsj, hsj, hsjIdx[mesh.elemTypes]);
	copyThickness();

	// TODO: construct Bs graph
	calcBsCuda();
	copyBs();
	initPlastParamsFloat();

	mesh.checkNodeAdjStruct();
	CudaCSR K(mesh.nodeAdjStruct, mesh.elemTypes, mesh.elemInfo, mesh.elem, blockSize);
	K.mallocFloat();

	mesh.checkColorMap();
	dptr<unsigned> dev_colorMap(mesh.elemCount());
	hostToDevice(dev_colorMap, mesh.colorMap.colorMap, mesh.elemCount());

	unsigned memLen = (K.N + blockSize - 1) / blockSize * blockSize;
	df_u.realloc(memLen);
	dd_u.realloc(memLen);
	dd_u.setZero(memLen);
	dptr<float> R_f(memLen);
	R_f.setZero(memLen);
	dptr<double> R(memLen);  // TODO: set zero last chunk
	R.setZero(memLen);

	cudaInitConditions(dd_u, dev_kinNodes, R, lineCount, dd_lines, dev_lineRows);
	copyConditions(R_f, R, memLen);

	CudaJCGV<float> cgSolver_f(K.N, K.data_f, K.rows, K.cols, R_f, \
		df_u, dev_kinNodes, preconditioning, true, false, 1, 512, \
		df_lines, dev_lineRows, lineCount);

	unsigned iterNum = 0;
	float relErr_f = 0.f;
	do {
		assembleFloat(K, dev_colorMap);

		unsigned slaeIter = 0;
		slaeIter = cgSolver_f.solve(2e-4f);

		updateParamFloat();

		relErr_f = exitCondition(df_intensityS, df_tableS, df_exit);
		printIter(iterNum, slaeIter, (double)relErr_f);

		++iterNum;
	} while (relErr_f > 1e-4f);

	if (iterOutput) std::cout << "Copying from float to double...\n";

	df_hsj.free();
	df_B.free();
	df_detJ.free();
	cgSolver_f.~CudaJCGV();
	K.data_f.free();
	K.mallocDouble();

	copyFloatToDouble(memLen);

	CudaJCGV<double> cgSolver(K.N, K.data, K.rows, K.cols, R, \
		dd_u, dev_kinNodes, preconditioning, true, false, 1, blockSize, \
		dd_lines, dev_lineRows, lineCount);

	double relErr = 0.;
	do {
		assemble(K, dev_colorMap);

		unsigned slaeIter = 0;
		slaeIter = cgSolver.solve(1e-7);

		updateParamDouble();

		relErr = exitCondition(dd_intensityS, dd_tableS, dd_exit);
		printIter(iterNum, slaeIter, relErr);

		++iterNum;
	} while (relErr > 1e-5);

	calcPlastDeformCuda();

	solvingTime += omp_get_wtime();

	std::cout << (iterOutput ? "\n" : "") << "Solved [" << solvingTime << " sec]\n";
	if (!iterOutput)
		std::cout << "Iterations: " << iterNum << "                                             " \
		<< "\nExit error: " << exitCondition() << "\n";

	ramSaved = false;
	plastSolved = true;

	return solvingTime;
}


//size template?
void PlasticitySolver::distribKe4(StripSLAE& K, const SymMatrix<8, double>& Ke, int e) const {
	int begin = 4 * (e - mesh.elemPos[0]), size = 4;
	for (int i = 0; i < size; ++i) {
		int gi = 2 * mesh.elem4[begin + i];
#pragma omp atomic
		K(gi, gi) += Ke(2 * i, 2 * i);
#pragma omp atomic
		K(gi + 1, gi) += Ke(2 * i + 1, 2 * i);
#pragma omp atomic
		K(gi, gi + 1) += Ke(2 * i + 1, 2 * i);
#pragma omp atomic
		K(gi + 1, gi + 1) += Ke(2 * i + 1, 2 * i + 1);
	}
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < i; ++j) {
			int gi = 2 * mesh.elem4[begin + i];  //глобальне индексы
			int gj = 2 * mesh.elem4[begin + j];
#pragma omp atomic
			K(gi, gj) += Ke(2 * i, 2 * j);
#pragma omp atomic
			K(gi + 1, gj) += Ke(2 * i + 1, 2 * j);
#pragma omp atomic
			K(gi, gj + 1) += Ke(2 * i, 2 * j + 1);
#pragma omp atomic
			K(gi + 1, gj + 1) += Ke(2 * i + 1, 2 * j + 1);

#pragma omp atomic
			K(gj, gi) += Ke(2 * i, 2 * j);
#pragma omp atomic
			K(gj, gi + 1) += Ke(2 * i + 1, 2 * j);
#pragma omp atomic
			K(gj + 1, gi) += Ke(2 * i, 2 * j + 1);
#pragma omp atomic
			K(gj + 1, gi + 1) += Ke(2 * i + 1, 2 * j + 1);
		}
}

void PlasticitySolver::distribKe8(StripSLAE& K, const SymMatrix<16, double>& Ke, int e) const {
	int begin = 8 * (e - mesh.elemPos[1]), size = 8;
	for (int i = 0; i < size; ++i) {
		int gi = 2 * mesh.elem8[begin + i];
#pragma omp atomic
		K(gi, gi) += Ke(2 * i, 2 * i);
#pragma omp atomic
		K(gi + 1, gi) += Ke(2 * i + 1, 2 * i);
#pragma omp atomic
		K(gi, gi + 1) += Ke(2 * i + 1, 2 * i);
#pragma omp atomic
		K(gi + 1, gi + 1) += Ke(2 * i + 1, 2 * i + 1);
	}
	for (int i = 0; i < size; ++i)
		for (int j = 0; j < i; ++j) {
			int gi = 2 * mesh.elem8[begin + i];  //глобальне индексы
			int gj = 2 * mesh.elem8[begin + j];
#pragma omp atomic
			K(gi, gj) += Ke(2 * i, 2 * j);
#pragma omp atomic
			K(gi + 1, gj) += Ke(2 * i + 1, 2 * j);
#pragma omp atomic
			K(gi, gj + 1) += Ke(2 * i, 2 * j + 1);
#pragma omp atomic
			K(gi + 1, gj + 1) += Ke(2 * i + 1, 2 * j + 1);

#pragma omp atomic
			K(gj, gi) += Ke(2 * i, 2 * j);
#pragma omp atomic
			K(gj, gi + 1) += Ke(2 * i + 1, 2 * j);
#pragma omp atomic
			K(gj + 1, gi) += Ke(2 * i, 2 * j + 1);
#pragma omp atomic
			K(gj + 1, gi + 1) += Ke(2 * i + 1, 2 * j + 1);
		}
}


void PlasticitySolver::fillGlobalStiffness(StripSLAE& K, SparseSLAE& spK) {
	spK.clearStrip(K);
	if (m.h != 0.) {
		//цикл по элементам
		//formKe3
#pragma omp parallel for
		for (int e = mesh.elemPos[0]; e < mesh.elemPos[1]; ++e) {
			StaticMatrix<3, 3, double> Ce;
			//formC_(Ce, E_c[e], nu_c[e]);
			(*formC)(Ce, E_c[e], nu_c[e]);
			SymMatrix<8, double> Ke;
			memcpy(C_ + e, &Ce, sizeof(Ce));
			formKe4(Ke, e - mesh.elemPos[0], B4, detJ4, Ce, m.h);
			distribKe4(K, Ke, e);
		}
#pragma omp parallel for
		for (int e = mesh.elemPos[1]; e < mesh.elemPos[2]; ++e) {
			StaticMatrix<3, 3, double> Ce;
			(*formC)(Ce, E_c[e], nu_c[e]);
			memcpy(C_ + e, &Ce, sizeof(Ce));
			SymMatrix<16, double> Ke;
			formKe8(Ke, e - mesh.elemPos[1], B8, detJ8, Ce, m.h);
			distribKe8(K, Ke, e);
		}
		spK.copy(K);
	}
	else {
		//цикл по элементам
		//formKe3
#pragma omp parallel for
		for (int e = mesh.elemPos[0]; e < mesh.elemPos[1]; ++e) {
			StaticMatrix<3, 3, double> Ce;
			//formC_(Ce, E_c[e], nu_c[e]);
			(*formC)(Ce, E_c[e], nu_c[e]);
			SymMatrix<8, double> Ke;
			memcpy(C_ + e, &Ce, sizeof(Ce));
			formKe4(Ke, e - mesh.elemPos[0], B4, detJ4, Ce, h);
			distribKe4(K, Ke, e);
		}
#pragma omp parallel for
		for (int e = mesh.elemPos[1]; e < mesh.elemPos[2]; ++e) {
			StaticMatrix<3, 3, double> Ce;
			(*formC)(Ce, E_c[e], nu_c[e]);
			memcpy(C_ + e, &Ce, sizeof(Ce));
			SymMatrix<16, double> Ke;
			formKe8(Ke, e - mesh.elemPos[1], B8, detJ8, Ce, h);
			distribKe8(K, Ke, e);
		}
		spK.copy(K);
	}
}

void PlasticitySolver::initConditions_(double* uvw, bool* mask, double* rhs, unsigned& lineCount, hptr<double>& lines, hptr<unsigned>& lineRows) {
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
				double R[6];
				vec2 node1 = mesh.node2[border[i - 2]], node2 = mesh.node2[border[i - 1]], node3 = mesh.node2[border[i]];
				double halfDiffX1 = 0.5 * (node1.x - node2.x), halfDiffY1 = 0.5 * (node1.y - node2.y), \
					halfDiffX2 = 0.5 * (node2.x - node3.x), halfDiffY2 = 0.5 * (node2.y - node3.y);
				double dl1 = sqrt(halfDiffX1 * halfDiffX1 + halfDiffY1 * halfDiffY1), \
					dl2 = sqrt(halfDiffX2 * halfDiffX2 + halfDiffY2 * halfDiffY2);
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
				R[0] = F1.x / 3.;
				R[1] = F1.y / 3.;
				R[2] = F2.x * 4. / 3.;
				R[3] = F2.y * 4. / 3.;
				R[4] = F3.x / 3.;
				R[5] = F3.y / 3.;

				R[0] *= m.h * dl1 * 2.;
				R[1] *= m.h * dl1 * 2.;
				R[2] *= m.h * (dl1 + dl2);
				R[3] *= m.h * (dl1 + dl2);
				R[4] *= m.h * dl2 * 2.;
				R[5] *= m.h * dl2 * 2.;

				int gi = 2 * border[i - 2];
				int gj = 2 * border[i - 1];
				int gk = 2 * border[i];

				rhs[gi] += R[0];
				rhs[gi + 1] += R[1];
				rhs[gj] += R[2];
				rhs[gj + 1] += R[3];
				rhs[gk] += R[4];
				rhs[gk + 1] += R[5];
			}
			else {
				double R[4];
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
				R[0] = F1.x;
				R[1] = F1.y;
				R[2] = F2.x;
				R[3] = F2.y;

				for (int j = 0; j < 4; ++j) \
					R[j] *= m.h * dl;

				int gi = 2 * border[i - 1];
				int gj = 2 * border[i];

				rhs[gi] += R[0];
				rhs[gi + 1] += R[1];
				rhs[gj] += R[2];
				rhs[gj + 1] += R[3];
			}
		}
	}
	if (cond.Rset) {  //TODO: переделать учёт толщины
		for (int e = 0; e < mesh.count4; ++e) {  //объёмные силы для 4-узловых элементов
			double space = mesh.elemSpace4(e);
			double he = 0.; //толщина элемента
			if (m.h != 0)
				he = m.h;
			else {
				int begin = mesh.count3 + 4 * e;
				he = 0.25 * (h[begin] + h[begin + 1] + h[begin + 2] + h[begin + 3]);
			}
			double coef = 0.25 * m.rho * he * space;
			for (int i = 4 * e; i < 4 * (e + 1); ++i) {
				int gi = mesh.elem4[i];
				vec2 node = mesh.node2[gi];
				vec2 R = cond.R(node);
				rhs[2 * gi] += coef * R.x;
				rhs[2 * gi + 1] += coef * R.y;
			}
		}
		for (int e = 0; e < mesh.count8; ++e) {  //объёмные силы для 8-узловых элементов
			double space = mesh.elemSpace8(e);
			double he = 0.; //толщина элемента
			if (m.h != 0)
				he = m.h;
			else {
				int begin = mesh.count3 + 4 * mesh.count4 + secIntPs2 * e;
				he = 0.;
				for (int k = 0; k < secIntPs2; ++k)
					he += h[begin + k];
				he *= 0.125;
			}
			double coef1 = -m.rho * he * space / 12.;
			double coef2 = m.rho * he * space / 3.;
			for (int i = 8 * e; i < 8 * e + 4; ++i) {
				int gi = mesh.elem8[i];
				vec2 node = mesh.node2[gi];
				vec2 R = cond.R(node);
				rhs[2 * gi] += coef1 * R.x;
				rhs[2 * gi + 1] += coef1 * R.y;
			}
			for (int i = 8 * e + 4; i < 8 * (e + 1); ++i) {
				int gi = mesh.elem8[i];
				vec2 node = mesh.node2[gi];
				vec2 R = cond.R(node);
				rhs[2 * gi] += coef2 * R.x;
				rhs[2 * gi + 1] += coef2 * R.y;
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
	//for (unsigned i = 0; i < lineCount; ++i)
	//	std::cout << lineRows[i] << " ";
}


void PlasticitySolver::cudaInitConditions(double* u, bool* mask, double* rhs, unsigned& lineCount, dptr<double>& lines, dptr<unsigned>& lineRows) {
	hptr<double> host_u(2 * mesh.nodeCount);
	hptr<bool> host_mask(2 * mesh.nodeCount);
	hptr<double> host_rhs(2 * mesh.nodeCount);
	hptr<unsigned> host_lineRows;
	hptr<double> host_lines;

	initConditions(host_u, host_mask, host_rhs, lineCount, host_lines, host_lineRows);

	cudaMemcpy(mask, host_mask, 2 * mesh.nodeCount * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(rhs, host_rhs, 2 * mesh.nodeCount * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(u, host_u, 2 * mesh.nodeCount * sizeof(double), cudaMemcpyHostToDevice);

	if (lineCount) {
		dd_lines.realloc(2 * lineCount);
		dev_lineRows.realloc(lineCount);
		hostToDevice(dd_lines, host_lines, lineCount * 2);
		hostToDevice(dev_lineRows, host_lineRows, lineCount);
	}
	else {
		dd_lines.free();
		dev_lineRows.free();
	}
}


//TODO: совместить с процессорной функцией
template<typename fp>
void PlasticitySolver::initConditions_(CudaSparseSLAE<fp>& K, fp* uv) {
	//uv.resize(2 * mesh.nodeCount, 0.);
	fp* loc_uv = new fp[K.memLen];
	fp* rp = new fp[K.memLen];
	bool* kinMask = new bool[K.memLen];
	/*fp* loc_uv = new fp[2 * mesh.nodeCount];
	fp* rp = new fp[2 * mesh.nodeCount];
	bool* kinMask = new bool[2 * mesh.nodeCount];*/
	for (int i = 0; i < K.memLen; ++i) {
		rp[i] = {};
		loc_uv[i] = {};
		kinMask[i] = true;
		//std::cout << kinMask[i] << " ";
	}
	for (const auto& [id, force] : cond.forceCond) {  //граничные силовые условия
		int* border = mesh.borders[id];
		for (int i = 1; i < mesh.borderLength[id]; ++i) {
			if (mesh.secOrdNodes[border[i]]) {
				++i;
				double R[6];
				vec2 node1 = mesh.node2[border[i - 2]], node2 = mesh.node2[border[i - 1]], node3 = mesh.node2[border[i]];
				double halfDiffX1 = 0.5 * (node1.x - node2.x), halfDiffY1 = 0.5 * (node1.y - node2.y), \
					halfDiffX2 = 0.5 * (node2.x - node3.x), halfDiffY2 = 0.5 * (node2.y - node3.y);
				double dl1 = sqrt(halfDiffX1 * halfDiffX1 + halfDiffY1 * halfDiffY1), \
					dl2 = sqrt(halfDiffX2 * halfDiffX2 + halfDiffY2 * halfDiffY2);
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
				R[0] = F1.x / 3.;
				R[1] = F1.y / 3.;
				R[2] = F2.x * 4. / 3.;
				R[3] = F2.y * 4. / 3.;
				R[4] = F3.x / 3.;
				R[5] = F3.y / 3.;

				R[0] *= m.h * dl1 * 2.;
				R[1] *= m.h * dl1 * 2.;
				R[2] *= m.h * (dl1 + dl2);
				R[3] *= m.h * (dl1 + dl2);
				R[4] *= m.h * dl2 * 2.;
				R[5] *= m.h * dl2 * 2.;

				int gi = 2 * border[i - 2];
				int gj = 2 * border[i - 1];
				int gk = 2 * border[i];

				rp[gi] += R[0];
				rp[gi + 1] += R[1];
				rp[gj] += R[2];
				rp[gj + 1] += R[3];
				rp[gk] += R[4];
				rp[gk + 1] += R[5];
			}
			else {
				double R[4];
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
				/*R[0] = (2. * F1.x + F2.x) / 3.;
				R[1] = (2. * F1.y + F2.y) / 3.;
				R[2] = (F1.x + 2. * F2.x) / 3.;
				R[3] = (F1.y + 2. * F2.y) / 3.;*/
				R[0] = F1.x;
				R[1] = F1.y;
				R[2] = F2.x;
				R[3] = F2.y;

				for (int j = 0; j < 4; ++j) \
					R[j] *= m.h * dl;

				int gi = 2 * border[i - 1];
				int gj = 2 * border[i];

				rp[gi] += R[0];
				rp[gi + 1] += R[1];
				rp[gj] += R[2];
				rp[gj + 1] += R[3];
			}
		}
	}
	if (cond.Rset) {  //TODO: переделать учёт толщины
		for (int e = 0; e < mesh.count4; ++e) {  //объёмные силы для 4-узловых элементов
			double space = mesh.elemSpace4(e);
			double he = 0.; //толщина элемента
			if (m.h != 0)
				he = m.h;
			else {
				int begin = mesh.count3 + 4 * e;
				he = 0.25 * (h[begin] + h[begin + 1] + h[begin + 2] + h[begin + 3]);
			}
			double coef = 0.25 * m.rho * he * space;
			for (int i = 4 * e; i < 4 * (e + 1); ++i) {
				int gi = mesh.elem4[i];
				vec2 node = mesh.node2[gi];
				vec2 R = cond.R(node);
				rp[2 * gi] += coef * R.x;
				rp[2 * gi + 1] += coef * R.y;
			}
		}
		for (int e = 0; e < mesh.count8; ++e) {  //объёмные силы для 8-узловых элементов
			double space = mesh.elemSpace8(e);
			double he = 0.; //толщина элемента
			if (m.h != 0)
				he = m.h;
			else {
				int begin = mesh.count3 + 4 * mesh.count4 + secIntPs2 * e;
				he = 0.;
				for (int k = 0; k < secIntPs2; ++k)
					he += h[begin + k];
				he *= 0.125;
			}
			double coef1 = -m.rho * he * space / 12.;
			double coef2 = m.rho * he * space / 3.;
			for (int i = 8 * e; i < 8 * e + 4; ++i) {
				int gi = mesh.elem8[i];
				vec2 node = mesh.node2[gi];
				vec2 R = cond.R(node);
				rp[2 * gi] += coef1 * R.x;
				rp[2 * gi + 1] += coef1 * R.y;
			}
			for (int i = 8 * e + 4; i < 8 * (e + 1); ++i) {
				int gi = mesh.elem8[i];
				vec2 node = mesh.node2[gi];
				vec2 R = cond.R(node);
				rp[2 * gi] += coef2 * R.x;
				rp[2 * gi + 1] += coef2 * R.y;
			}
		}
	}
	for (const auto& [id, displ] : cond.displCond) { //общие кинематические условие
		int* border = mesh.borders[id];
		size_t end = mesh.borderLength[id] - (border[0] == border[mesh.borderLength[id] - 1] ? 1 : 0);
		if (displ.fixMidX) {
			int gi = 2 * border[mesh.borderLength[id] / 2];
			kinMask[gi] = false;
			loc_uv[gi] = displ.displFunc(mesh.node2[border[mesh.borderLength[id] / 2]]).x;
		}
		else
			for (size_t i = 0; i < end; ++i) {
				int gi = 2 * border[i];
				kinMask[gi] = false;
				loc_uv[gi] = displ.displFunc(mesh.node2[border[i]]).x;
			}
		if (displ.fixMidY) {
			int gi = 2 * border[mesh.borderLength[id] / 2] + 1;
			kinMask[gi] = false;
			loc_uv[gi] = displ.displFunc(mesh.node2[border[mesh.borderLength[id] / 2]]).y;

		}
		else
			for (size_t i = 0; i < end; ++i) {
				int gi = 2 * border[i] + 1;
				kinMask[gi] = false;
				loc_uv[gi] = displ.displFunc(mesh.node2[border[i]]).y;
			}
	}
	for (const auto& fixedAxis : cond.fixedAxis) {  //зафиксированные оси
		if (fixedAxis.vertical) {
			for (int i = 0; i < mesh.nodeCount; ++i) {
				//if (fabs(mesh.node[i].x - fixedAxis.coord) < (fabs(mesh.node[i].x) + fabs(fixedAxis.coord)) * 1e-10) {
				if (fabs(mesh.node2[i].x - fixedAxis.coord) < 1e-14) {
					kinMask[2 * i] = false;
				}
			}
		}
		else {
			for (int i = 0; i < mesh.nodeCount; ++i)
				//if (fabs(mesh.node[i].y - fixedAxis.coord) < (fabs(mesh.node[i].y) + fabs(fixedAxis.coord)) * 1e-14)
				if (fabs(mesh.node2[i].y - fixedAxis.coord) < 1e-14) {
					kinMask[2 * i + 1] = false;
				}
		}
	}
	for (const auto& [id, fixedBorder] : cond.fixedBorder) {  //зафиксированные по одной оси границы
		int* border = mesh.borders[id];
		int end = mesh.borderLength[id] - (border[0] == border[mesh.borderLength[id] - 1] ? 1 : 0);
		if (fixedBorder.vertical) {
			for (int i = 0; i < end; ++i)
				kinMask[2 * border[i]] = false;
		}
		else {
			for (int i = 0; i < end; ++i)
				kinMask[2 * border[i] + 1] = false;
		}
	}

	unsigned lineMem = 0;
	for (const auto& [id, line] : cond.alongLine) // подсчёт закреплений границ вдоль прямых
		lineMem += mesh.borderLength[id];
	lineMem += cond.pointOnLine.size();
	if (lineMem) {
		lines.realloc(2 * lineMem);
		lineRows.realloc(lineMem);
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
				rp[2 * i] += forcePoint.value.x;
				rp[2 * i + 1] += forcePoint.value.y;
				break;
			}
	}
	for (const auto& displPoint : cond.displPoint) {  //перемещение в точке
		for (int i = 0; i < mesh.nodeCount; ++i)
			if ((mesh.node2[i] - displPoint.point).norm() < 1e-14) {
				loc_uv[2 * i] = displPoint.value.x;
				loc_uv[2 * i + 1] = displPoint.value.y;
				kinMask[2 * i] = false;
				kinMask[2 * i + 1] = false;
				break;
			}
	}
	//for (int i = 0; i < K.memLen; ++i) \
		std::cout << kinMask[i] << " ";
	cudaMemcpy(dev_kinNodes, kinMask, 2 * mesh.nodeCount * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(K.rp, rp, K.memLen * sizeof(fp), cudaMemcpyHostToDevice);
	cudaMemcpy(uv, loc_uv, 2 * mesh.nodeCount * sizeof(fp), cudaMemcpyHostToDevice);
	/*cudaMemcpy(dev_kinNodes, kinMask, 2 * mesh.nodeCount * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(K.rp, rp, 2 * mesh.nodeCount * sizeof(fp), cudaMemcpyHostToDevice);
	cudaMemcpy(uv, loc_uv, 2 * mesh.nodeCount * sizeof(fp), cudaMemcpyHostToDevice);*/
	delete[] kinMask;
	delete[] loc_uv;
	delete[] rp;

	if (lineCount) {
		dd_lines.realloc(2 * lineCount);
		dev_lineRows.realloc(lineCount);
		hostToDevice(dd_lines, lines, lineCount * 2);
		hostToDevice(dev_lineRows, lineRows, lineCount);
		lines.free();
		lineRows.free();
	}
	else {
		dd_lines.free();
		dev_lineRows.free();
	}
}

void PlasticitySolver::calcBs_() {
	//calsB3
#pragma omp parallel for
	for (int e = 0; e < mesh.count4; ++e) {
		calcB4<double>(e, B4, detJ4, mesh.node2, mesh.elem4);
	}
#pragma omp parallel for
	for (int e = 0; e < mesh.count8; ++e) {
		calcB8<double>(e, B8, detJ8, mesh.node2, mesh.elem8);
	}
}


void PlasticitySolver::calcBsCuda_() {
	//std::cout << "lol B\n";
	const int locBS = 512;
	//calcBs3
	if (mesh.count4)
		calcBs4<<<(mesh.count4 + locBS - 1) / locBS, locBS>>>(dd_B4, dd_detJ4, mesh.dev_node, mesh.dev_elem4, mesh.count4);
	if (mesh.count8)
		calcBs8<<<(mesh.count8 + locBS - 1) / locBS, locBS>>>(dd_B8, dd_detJ8, mesh.dev_node, mesh.dev_elem8, mesh.count8);
	cudaDeviceSynchronize();
}


__global__ void calcBsKernel(unsigned elemCount, double* B, double* detJ, \
	unsigned nodeCount, unsigned intPointsCount, unsigned Bsize, unsigned elemDim, unsigned meshDim, \
	const unsigned* elem, const double* node, const double* intPoints, \
	void (*calcB)(double* B, double* detJ, const double* x, const double* xi)) {

	unsigned e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= elemCount) return;

	double x[60];  // max size for 20-node 3D element
	for (unsigned i = 0; i < nodeCount; ++i) {
		unsigned n = elem[e * nodeCount + i];
		for (unsigned j = 0; j < meshDim; ++j)
			x[i * meshDim + j] = node[n * meshDim + j];
	}

	for (unsigned p = 0; p < intPointsCount; ++p) {
		unsigned locIdx = e * intPointsCount + p;
		(*calcB)(B + Bsize * locIdx, detJ + locIdx, x, intPoints + p * elemDim);
	}
}


__global__ void calcBsAxiSymKernel(unsigned elemCount, double* B, double* detJ, \
	unsigned nodeCount, unsigned intPointsCount, unsigned Bsize, unsigned elemDim, \
	const unsigned* elem, const double* node, const double* intPoints, \
	void (*calcB)(double* B, double* detJ, const double* x, const double* xi),
	void (*calcN)(double* N, const double* xi)) {

	unsigned e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= elemCount) return;

	double x[16];  // max size for 8-node 2D element
	for (unsigned i = 0; i < nodeCount; ++i) {
		unsigned n = elem[e * nodeCount + i];
		for (unsigned j = 0; j < 2; ++j)
			x[i * 2 + j] = node[n * 2 + j];
	}

	for (unsigned p = 0; p < intPointsCount; ++p) {
		unsigned locIdx = e * intPointsCount + p;
		calcB(B + Bsize * locIdx, detJ + locIdx, x, intPoints + p * elemDim);
		calcN(B + Bsize * locIdx + elemDim * nodeCount, intPoints + p * elemDim);
	}
}


void PlasticitySolver::calcBsCuda() {
	for (unsigned t = 0; t < mesh.elemTypes; ++t) {
		const FiniteElement& type = mesh.elemInfo[t];
		const CudaFiniteElement& cudaType = cudaMesh.elemInfo[t];
		calcBsKernel<<<(type.elemCount + 255) / 256, 256>>>(
			type.elemCount, dd_B + Bidx[t], dd_detJ + detJidx[t],
			type.nodeCount, type.intPointsCount, Bsize[t], type.elemDim, mesh.dim,
			cudaMesh.elem, cudaMesh.node, cudaType.intPoints,
			cudaType.calcB);
		cudaDeviceSynchronize();
		//std::cout << cudaGetErrorString(cudaGetLastError()) << "\n";
		// if axisym...
	}
}


void PlasticitySolver::calcThickness_() {
	int dataSize = mesh.count3 + mesh.count4 * 4 + mesh.count8 * secIntPs2;
	h = new double[dataSize];
	const double GS_point2[2] = { -0.577'350'269'189'626, 0.577'350'269'189'626 };
	const double GS_point3[3] = { -0.774'596'669'241'483, 0., 0.774'596'669'241'483 };

#pragma omp parallel for
	for (int e = 0; e < mesh.count4; ++e) {
		int begin = 4 * e;
		double he[4] = {};
		for (int k = 0; k < 4; ++k)
			he[k] = m.hf(mesh.node2[mesh.elem4[begin + k]]);

		begin = 4 * e + mesh.count3;
		for (int i = 0; i < 2; ++i)
			for (int j = 0; j < 2; ++j) {
				int k = 2 * i + j;
				double r = GS_point2[i], s = GS_point2[j];
				h[begin + k] = 0.25 * (\
					(1. + r) * (1. + s) * he[0] + (1. - r) * (1. + s) * he[1] + \
					(1. - r) * (1. - s) * he[2] + (1. + r) * (1. - s) * he[3]);
			}
	}
#pragma omp parallel for
	for (int e = 0; e < mesh.count8; ++e) {
		int begin = 8 * e;
		double he[8] = {};
		for (int k = 0; k < 8; ++k)
			he[k] = m.hf(mesh.node2[mesh.elem8[begin + k]]);

		begin = secIntPs2 * e + 4 * mesh.count4 + mesh.count3;
		for (int i = 0; i < secIntPs; ++i)
			for (int j = 0; j < secIntPs; ++j) {
				int k = secIntPs * i + j;
				double r = GS_point3[i], s = GS_point3[j];
				h[begin + k] = 0.25 * (\
						(1. + r) * (1. + s) * (r + s - 1.) * he[0] \
						- (1. - r) * (1. + s) * (r - s + 1.) * he[1] \
						- (1. - r) * (1. - s) * (r + s + 1.) * he[2] \
						+ (1. + r) * (1. - s) * (r - s - 1.) * he[3] \
					) + 0.5 * ( \
						(1. - r * r) * (1. + s) * he[4] + \
						(1. - r) * (1. - s * s) * he[5] + \
						(1. - r * r) * (1. - s) * he[6] + \
						(1. + r) * (1. - s * s) * he[7]
					);
			}
	}
	//DEBUG
	/*std::cout << "\n";
	for (int i = 0; i < dataSize; ++i) {
		std::cout << h[i] << " ";
	}
	std::cout << "\n";*/
}


__global__ static void copyThicknessToFloat(double* dh, float* fh, int dataSize) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < dataSize)
		fh[i] = (float)dh[i];
}

void PlasticitySolver::copyThickness_() {
	int dataSize = mesh.count3 + mesh.count4 * 4 + mesh.count8 * secIntPs2;
	cudaMalloc((void**)&dd_h, dataSize * sizeof(double));
	cudaMemcpy(dd_h, h, dataSize * sizeof(double), cudaMemcpyHostToDevice);
	if (floatBoost) {
		cudaMalloc((void**)&df_h, dataSize * sizeof(double));
		copyThicknessToFloat<<<(dataSize + 1023) / 1024, 1024>>>(dd_h, df_h, dataSize);
		cudaDeviceSynchronize();
	}
}


void PlasticitySolver::copyThickness() {
	const unsigned block = 512;
	doubleToFloat<<<(hsjIdx[mesh.elemTypes] + block - 1) / block, block>>>(df_hsj, dd_hsj, hsjIdx[mesh.elemTypes]);
	cudaDeviceSynchronize();
}


void PlasticitySolver::updateParameters_(int iterNum) {
	auto sqr = [](double x) { return x * x; };

#pragma omp parallel for
	for (int e = mesh.elemPos[0]; e < mesh.elemPos[1]; ++e) {
		int begin = 4 * (e - mesh.elemPos[0]);
		double
			u1 = uv[2 * mesh.elem4[begin]],
			v1 = uv[2 * mesh.elem4[begin] + 1],
			u2 = uv[2 * mesh.elem4[begin + 1]],
			v2 = uv[2 * mesh.elem4[begin + 1] + 1],
			u3 = uv[2 * mesh.elem4[begin + 2]],
			v3 = uv[2 * mesh.elem4[begin + 2] + 1],
			u4 = uv[2 * mesh.elem4[begin + 3]],
			v4 = uv[2 * mesh.elem4[begin + 3] + 1];

		exx[e] = 0.;
		eyy[e] = 0.;
		gamma[e] = 0.;
		for (int i = 0; i < 4; ++i) {
			StaticMatrix<2, 4, double>& B = B4[begin + i];
			exx[e] += u1 * B(0, 0) + u2 * B(0, 1) + u3 * B(0, 2) + u4 * B(0, 3);
			eyy[e] += v1 * B(1, 0) + v2 * B(1, 1) + v3 * B(1, 2) + v4 * B(1, 3);
			gamma[e] += u1 * B(1, 0) + u2 * B(1, 1) + u3 * B(1, 2) + u4 * B(1, 3) + \
				v1 * B(0, 0) + v2 * B(0, 1) + v3 * B(0, 2) + v4 * B(0, 3);
		}
		exx[e] *= 0.25;
		eyy[e] *= 0.25;
		gamma[e] *= 0.25;
	}
	
	const double GS_coef[3] = { 0.555'555'555'555'555, 0.888'888'888'888'888, 0.555'555'555'555'555 };
	//const double GS_coef[4] = { 0.3478548451374538, 0.652145154862546, 0.652145154862546, 0.3478548451374538 };
#pragma omp parallel for
	for (int e = mesh.elemPos[1]; e < mesh.elemPos[2]; ++e) {
		int begin = 8 * (e - mesh.elemPos[1]);
		double u[8], v[8];
		for (int i = 0; i < 8; ++i) {
			u[i] = uv[2 * mesh.elem8[begin + i]];
			v[i] = uv[2 * mesh.elem8[begin + i] + 1];
		}

		exx[e] = 0.;
		eyy[e] = 0.;
		gamma[e] = 0.;
		begin = secIntPs * secIntPs * (e - mesh.elemPos[1]);
		for (int i = 0; i < secIntPs; ++i)
			for (int j = 0; j < secIntPs; ++j) {
				double exxLoc = 0., eyyLoc = 0., gammaLoc = 0.;
				StaticMatrix<2, 8, double>& B = B8[begin + i * secIntPs + j];
				for (int k = 0; k < 8; ++k) {
					exxLoc += u[k] * B(0, k);
					eyyLoc += v[k] * B(1, k);
					gammaLoc += u[k] * B(1, k) + v[k] * B(0, k);
				}
				double coef = GS_coef[i] * GS_coef[j];
				exx[e] += exxLoc * coef;
				eyy[e] += eyyLoc * coef;
				gamma[e] += gammaLoc * coef;
			}
		exx[e] *= 0.25;
		eyy[e] *= 0.25;
		gamma[e] *= 0.25;
	}

	/*intS_2 = intS_1;
	intS_1 = intensityS;
	intensityS = intS_2;*/
	//std::swap(intensityS, intS_2);
	//std::swap(intS_2, intS_1);
	//if (iterNum < 3)
#pragma omp parallel for
		for (int e = 0; e < mesh.elemCount(); ++e) {

			sxx[e] = C_[e](0, 0) * exx[e] + C_[e](0, 1) * eyy[e];
			syy[e] = C_[e](0, 1) * exx[e] + C_[e](1, 1) * eyy[e];
			tau[e] = C_[e](2, 2) * gamma[e];

			intensityS[e] = sqrt(0.5 * (sqr(sxx[e]) + sqr(syy[e]) + sqr(sxx[e] - syy[e]) + 6. * sqr(tau[e])));

			intensityE[e] = intensityS[e] / E_c[e];
			tableS[e] = m.f(intensityE[e]);
			psi[e] = m.E * intensityE[e] / tableS[e];
			E_c[e] = m.E / psi[e];
			nu_c[e] = 0.5 * (1. - (1. - 2. * m.nu) / psi[e]);
			//std::cout << tableS[e] << " " << intensityS[e] << "\n";
		}
//	else
//#pragma omp parallel for
//		for (int e = 0; e < mesh.elemCount(); ++e) {
//
//			sxx[e] = C[e](0, 0) * exx[e] + C[e](0, 1) * eyy[e];
//			syy[e] = C[e](0, 1) * exx[e] + C[e](1, 1) * eyy[e];
//			tau[e] = C[e](2, 2) * gamma[e];
//
//			intensityS[e] = sqrt(0.5 * (sqr(sxx[e]) + sqr(syy[e]) + sqr(sxx[e] - syy[e]) + 6. * sqr(tau[e])));
//			intensityS[e] = intS_2[e] + sqr(intS_2[e] - intS_1[e]) / (2. * intS_1[e] - intS_2[e] - intensityS[e]);
//
//			intensityE[e] = intensityS[e] / E_c[e];
//			tableS[e] = m.f(intensityE[e]);
//			psi[e] = m.E * intensityE[e] / tableS[e];
//			E_c[e] = m.E / psi[e];
//			nu_c[e] = 0.5 * (1. - (1. - 2. * m.nu) / psi[e]);
//			//std::cout << tableS[e] << " " << intensityS[e] << "\n";
//		}
}




__host__ __device__
static inline int strip(int i, int j, int W) {
	/*if (j > i + W || j < i - W)
		printf("!!! %d %d\n", i, j);*/
	return i * (2 * W + 1) + j + W - i;
}

#if __CUDA_ARCH__ < 600
template<typename fp>
__global__ static void insertKe4(fp* K, int W, StaticMatrix<2, 4, fp>* Bs, fp* detJ, int* elem, fp* E_c, fp* nu_c, fp h, int count) {
	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= count) return;

	StaticMatrix<3, 3, fp> C_;
	formC_(C_, E_c[e], nu_c[e]);
	SymMatrix<8, fp> Ke;
	formKe4(Ke, e, Bs, detJ, C_, h);

	int begin = e * 4;
	for (int i = 0; i < 4; ++i) {
		int gi = 2 * elem[begin + i];
		atomicAdd_arch52(K + strip(gi, gi, W), Ke(2 * i, 2 * i));
		atomicAdd_arch52(K + strip(gi + 1, gi, W), Ke(2 * i + 1, 2 * i));
		atomicAdd_arch52(K + strip(gi, gi + 1, W), Ke(2 * i + 1, 2 * i));
		atomicAdd_arch52(K + strip(gi + 1, gi + 1, W), Ke(2 * i + 1, 2 * i + 1));
	}
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < i; ++j) {
			int gi = 2 * elem[begin + i];  //глобальне индексы
			int gj = 2 * elem[begin + j];

			atomicAdd_arch52(K + strip(gi, gj, W), Ke(2 * i, 2 * j));
			atomicAdd_arch52(K + strip(gi + 1, gj, W), Ke(2 * i + 1, 2 * j));
			atomicAdd_arch52(K + strip(gi, gj + 1, W), Ke(2 * i, 2 * j + 1));
			atomicAdd_arch52(K + strip(gi + 1, gj + 1, W), Ke(2 * i + 1, 2 * j + 1));

			atomicAdd_arch52(K + strip(gj, gi, W), Ke(2 * i, 2 * j));
			atomicAdd_arch52(K + strip(gj, gi + 1, W), Ke(2 * i + 1, 2 * j));
			atomicAdd_arch52(K + strip(gj + 1, gi, W), Ke(2 * i, 2 * j + 1));
			atomicAdd_arch52(K + strip(gj + 1, gi + 1, W), Ke(2 * i + 1, 2 * j + 1));
		}
}

template<typename fp>
__global__ static void insertKe4(fp* K, int W, StaticMatrix<3, 3, fp>* C_, void (*formC)(StaticMatrix<3, 3, fp>&, fp, fp), SymMatrix<8, fp>* Ke4, \
	StaticMatrix<2, 4, fp>* Bs, fp* detJ, int* elem, fp* E_c, fp* nu_c, fp h, int count, int elemPos) {
	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= count) return;

	StaticMatrix<3, 3, fp> Ce;
	(*formC)(Ce, E_c[e], nu_c[e]);
	for (int i = 0; i < 9; ++i)
		C_[elemPos + e].data[i] = Ce.data[i];
	SymMatrix<8, fp>& Ke = Ke4[e];
	for (int i = 0; i < Ke.dataSize(); ++i)
		Ke.data[i] = {};
	formKe4(Ke, e, Bs, detJ, Ce, h);

	int begin = e * 4;
	for (int i = 0; i < 4; ++i) {
		int gi = 2 * elem[begin + i];
		atomicAdd_arch52(K + strip(gi, gi, W), Ke(2 * i, 2 * i));
		atomicAdd_arch52(K + strip(gi + 1, gi, W), Ke(2 * i + 1, 2 * i));
		atomicAdd_arch52(K + strip(gi, gi + 1, W), Ke(2 * i + 1, 2 * i));
		atomicAdd_arch52(K + strip(gi + 1, gi + 1, W), Ke(2 * i + 1, 2 * i + 1));
	}
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < i; ++j) {
			int gi = 2 * elem[begin + i];  //глобальне индексы
			int gj = 2 * elem[begin + j];

			atomicAdd_arch52(K + strip(gi, gj, W), Ke(2 * i, 2 * j));
			atomicAdd_arch52(K + strip(gi + 1, gj, W), Ke(2 * i + 1, 2 * j));
			atomicAdd_arch52(K + strip(gi, gj + 1, W), Ke(2 * i, 2 * j + 1));
			atomicAdd_arch52(K + strip(gi + 1, gj + 1, W), Ke(2 * i + 1, 2 * j + 1));

			atomicAdd_arch52(K + strip(gj, gi, W), Ke(2 * i, 2 * j));
			atomicAdd_arch52(K + strip(gj, gi + 1, W), Ke(2 * i + 1, 2 * j));
			atomicAdd_arch52(K + strip(gj + 1, gi, W), Ke(2 * i, 2 * j + 1));
			atomicAdd_arch52(K + strip(gj + 1, gi + 1, W), Ke(2 * i + 1, 2 * j + 1));
		}
}

template<typename fp>
__global__ static void insertKe4(fp* K, int W, StaticMatrix<3, 3, fp>* C_, void (*formC)(StaticMatrix<3, 3, fp>&, fp, fp), SymMatrix<8, fp>* Ke4, \
	StaticMatrix<2, 4, fp>* Bs, fp* detJ, int* elem, fp* E_c, fp* nu_c, const fp* h, int count, int elemPos) {
	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= count) return;

	StaticMatrix<3, 3, fp> Ce;
	(*formC)(Ce, E_c[e], nu_c[e]);
	for (int i = 0; i < 9; ++i)
		C_[elemPos + e].data[i] = Ce.data[i];
	SymMatrix<8, fp>& Ke = Ke4[e];
	for (int i = 0; i < Ke.dataSize(); ++i)
		Ke.data[i] = {};
	formKe4(Ke, e, Bs, detJ, Ce, h);

	int begin = e * 4;
	for (int i = 0; i < 4; ++i) {
		int gi = 2 * elem[begin + i];
		atomicAdd_arch52(K + strip(gi, gi, W), Ke(2 * i, 2 * i));
		atomicAdd_arch52(K + strip(gi + 1, gi, W), Ke(2 * i + 1, 2 * i));
		atomicAdd_arch52(K + strip(gi, gi + 1, W), Ke(2 * i + 1, 2 * i));
		atomicAdd_arch52(K + strip(gi + 1, gi + 1, W), Ke(2 * i + 1, 2 * i + 1));
	}
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < i; ++j) {
			int gi = 2 * elem[begin + i];  //глобальне индексы
			int gj = 2 * elem[begin + j];

			atomicAdd_arch52(K + strip(gi, gj, W), Ke(2 * i, 2 * j));
			atomicAdd_arch52(K + strip(gi + 1, gj, W), Ke(2 * i + 1, 2 * j));
			atomicAdd_arch52(K + strip(gi, gj + 1, W), Ke(2 * i, 2 * j + 1));
			atomicAdd_arch52(K + strip(gi + 1, gj + 1, W), Ke(2 * i + 1, 2 * j + 1));

			atomicAdd_arch52(K + strip(gj, gi, W), Ke(2 * i, 2 * j));
			atomicAdd_arch52(K + strip(gj, gi + 1, W), Ke(2 * i + 1, 2 * j));
			atomicAdd_arch52(K + strip(gj + 1, gi, W), Ke(2 * i, 2 * j + 1));
			atomicAdd_arch52(K + strip(gj + 1, gi + 1, W), Ke(2 * i + 1, 2 * j + 1));
		}
}

template<typename fp>
__global__ static void insertKe8(fp* K, int W, StaticMatrix<3, 3, fp>* C_, void (*formC)(StaticMatrix<3, 3, fp>&, fp, fp), SymMatrix<16, fp>* Ke8, \
	StaticMatrix<2, 8, fp>* Bs, fp* detJ, int* elem, fp* E_c, fp* nu_c, fp h, int count, int elemPos) {
	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= count) return;

	StaticMatrix<3, 3, fp> Ce;
	(*formC)(Ce, E_c[e], nu_c[e]);
	for (int i = 0; i < 9; ++i)
		C_[elemPos + e].data[i] = Ce.data[i];
	SymMatrix<16, fp>& Ke = Ke8[e];
	for (int i = 0; i < Ke.dataSize(); ++i)
		Ke.data[i] = {};
	formKe8(Ke, e, Bs, detJ, Ce, h);

	int begin = e * 8;
	for (int i = 0; i < 8; ++i) {
		int gi = 2 * elem[begin + i];
		atomicAdd_arch52(K + strip(gi, gi, W), Ke(2 * i, 2 * i));
		atomicAdd_arch52(K + strip(gi + 1, gi, W), Ke(2 * i + 1, 2 * i));
		atomicAdd_arch52(K + strip(gi, gi + 1, W), Ke(2 * i + 1, 2 * i));
		atomicAdd_arch52(K + strip(gi + 1, gi + 1, W), Ke(2 * i + 1, 2 * i + 1));
	}
	for (int i = 0; i < 8; ++i)
		for (int j = 0; j < i; ++j) {
			int gi = 2 * elem[begin + i];  //глобальне индексы
			int gj = 2 * elem[begin + j];

			atomicAdd_arch52(K + strip(gi, gj, W), Ke(2 * i, 2 * j));
			atomicAdd_arch52(K + strip(gi + 1, gj, W), Ke(2 * i + 1, 2 * j));
			atomicAdd_arch52(K + strip(gi, gj + 1, W), Ke(2 * i, 2 * j + 1));
			atomicAdd_arch52(K + strip(gi + 1, gj + 1, W), Ke(2 * i + 1, 2 * j + 1));

			atomicAdd_arch52(K + strip(gj, gi, W), Ke(2 * i, 2 * j));
			atomicAdd_arch52(K + strip(gj, gi + 1, W), Ke(2 * i + 1, 2 * j));
			atomicAdd_arch52(K + strip(gj + 1, gi, W), Ke(2 * i, 2 * j + 1));
			atomicAdd_arch52(K + strip(gj + 1, gi + 1, W), Ke(2 * i + 1, 2 * j + 1));
		}
}

template<typename fp>
__global__ static void insertKe8(fp* K, int W, StaticMatrix<3, 3, fp>* C_, void (*formC)(StaticMatrix<3, 3, fp>&, fp, fp), SymMatrix<16, fp>* Ke8, \
	StaticMatrix<2, 8, fp>* Bs, fp* detJ, int* elem, fp* E_c, fp* nu_c, const fp* h, int count, int elemPos) {
	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= count) return;

	StaticMatrix<3, 3, fp> Ce;
	(*formC)(Ce, E_c[e], nu_c[e]);
	for (int i = 0; i < 9; ++i)
		C_[elemPos + e].data[i] = Ce.data[i];
	SymMatrix<16, fp>& Ke = Ke8[e];
	for (int i = 0; i < Ke.dataSize(); ++i)
		Ke.data[i] = {};
	formKe8(Ke, e, Bs, detJ, Ce, h);

	int begin = e * 8;
	for (int i = 0; i < 8; ++i) {
		int gi = 2 * elem[begin + i];
		atomicAdd_arch52(K + strip(gi, gi, W), Ke(2 * i, 2 * i));
		atomicAdd_arch52(K + strip(gi + 1, gi, W), Ke(2 * i + 1, 2 * i));
		atomicAdd_arch52(K + strip(gi, gi + 1, W), Ke(2 * i + 1, 2 * i));
		atomicAdd_arch52(K + strip(gi + 1, gi + 1, W), Ke(2 * i + 1, 2 * i + 1));
	}
	for (int i = 0; i < 8; ++i)
		for (int j = 0; j < i; ++j) {
			int gi = 2 * elem[begin + i];  //глобальне индексы
			int gj = 2 * elem[begin + j];

			atomicAdd_arch52(K + strip(gi, gj, W), Ke(2 * i, 2 * j));
			atomicAdd_arch52(K + strip(gi + 1, gj, W), Ke(2 * i + 1, 2 * j));
			atomicAdd_arch52(K + strip(gi, gj + 1, W), Ke(2 * i, 2 * j + 1));
			atomicAdd_arch52(K + strip(gi + 1, gj + 1, W), Ke(2 * i + 1, 2 * j + 1));

			atomicAdd_arch52(K + strip(gj, gi, W), Ke(2 * i, 2 * j));
			atomicAdd_arch52(K + strip(gj, gi + 1, W), Ke(2 * i + 1, 2 * j));
			atomicAdd_arch52(K + strip(gj + 1, gi, W), Ke(2 * i, 2 * j + 1));
			atomicAdd_arch52(K + strip(gj + 1, gi + 1, W), Ke(2 * i + 1, 2 * j + 1));
		}
}
#else
template<typename fp>
__global__ static void insertKe4(fp* K, int W, StaticMatrix<2, 4, fp>* Bs, fp* detJ, int* elem, fp* E_c, fp* nu_c, fp h, int count) {
	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= count) return;

	StaticMatrix<3, 3, fp> C;
	formC_(C, E_c[e], nu_c[e]);
	SymMatrix<8, fp> Ke;
	formKe4(Ke, e, Bs, detJ, C, h);

	int begin = e * 4;
	for (int i = 0; i < 4; ++i) {
		int gi = 2 * elem[begin + i];
		atomicAdd(K + strip(gi, gi, W), Ke(2 * i, 2 * i));
		atomicAdd(K + strip(gi + 1, gi, W), Ke(2 * i + 1, 2 * i));
		atomicAdd(K + strip(gi, gi + 1, W), Ke(2 * i + 1, 2 * i));
		atomicAdd(K + strip(gi + 1, gi + 1, W), Ke(2 * i + 1, 2 * i + 1));
	}
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < i; ++j) {
			int gi = 2 * elem[begin + i];  //глобальне индексы
			int gj = 2 * elem[begin + j];

			atomicAdd(K + strip(gi, gj, W), Ke(2 * i, 2 * j));
			atomicAdd(K + strip(gi + 1, gj, W), Ke(2 * i + 1, 2 * j));
			atomicAdd(K + strip(gi, gj + 1, W), Ke(2 * i, 2 * j + 1));
			atomicAdd(K + strip(gi + 1, gj + 1, W), Ke(2 * i + 1, 2 * j + 1));

			atomicAdd(K + strip(gj, gi, W), Ke(2 * i, 2 * j));
			atomicAdd(K + strip(gj, gi + 1, W), Ke(2 * i + 1, 2 * j));
			atomicAdd(K + strip(gj + 1, gi, W), Ke(2 * i, 2 * j + 1));
			atomicAdd(K + strip(gj + 1, gi + 1, W), Ke(2 * i + 1, 2 * j + 1));
		}
}

template<typename fp>
__global__ static void insertKe4(fp* K, int W, StaticMatrix<3, 3, fp>* C, void (*formC)(StaticMatrix<3, 3, fp>&, fp, fp), SymMatrix<8, fp>* Ke4, \
	StaticMatrix<2, 4, fp>* Bs, fp* detJ, int* elem, fp* E_c, fp* nu_c, fp h, int count, int elemPos) {
	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= count) return;

	StaticMatrix<3, 3, fp> Ce;
	(*formC)(Ce, E_c[e], nu_c[e]);
	for (int i = 0; i < 9; ++i)
		C[elemPos + e].data[i] = Ce.data[i];
	SymMatrix<8, fp>& Ke = Ke4[e];
	for (int i = 0; i < Ke.dataSize(); ++i)
		Ke.data[i] = {};
	formKe4(Ke, e, Bs, detJ, Ce, h);

	int begin = e * 4;
	for (int i = 0; i < 4; ++i) {
		int gi = 2 * elem[begin + i];
		atomicAdd(K + strip(gi, gi, W), Ke(2 * i, 2 * i));
		atomicAdd(K + strip(gi + 1, gi, W), Ke(2 * i + 1, 2 * i));
		atomicAdd(K + strip(gi, gi + 1, W), Ke(2 * i + 1, 2 * i));
		atomicAdd(K + strip(gi + 1, gi + 1, W), Ke(2 * i + 1, 2 * i + 1));
	}
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < i; ++j) {
			int gi = 2 * elem[begin + i];  //глобальне индексы
			int gj = 2 * elem[begin + j];

			atomicAdd(K + strip(gi, gj, W), Ke(2 * i, 2 * j));
			atomicAdd(K + strip(gi + 1, gj, W), Ke(2 * i + 1, 2 * j));
			atomicAdd(K + strip(gi, gj + 1, W), Ke(2 * i, 2 * j + 1));
			atomicAdd(K + strip(gi + 1, gj + 1, W), Ke(2 * i + 1, 2 * j + 1));

			atomicAdd(K + strip(gj, gi, W), Ke(2 * i, 2 * j));
			atomicAdd(K + strip(gj, gi + 1, W), Ke(2 * i + 1, 2 * j));
			atomicAdd(K + strip(gj + 1, gi, W), Ke(2 * i, 2 * j + 1));
			atomicAdd(K + strip(gj + 1, gi + 1, W), Ke(2 * i + 1, 2 * j + 1));
		}
}

template<typename fp>
__global__ static void insertKe4(fp* K, int W, StaticMatrix<3, 3, fp>* C, void (*formC)(StaticMatrix<3, 3, fp>&, fp, fp), SymMatrix<8, fp>* Ke4, \
	StaticMatrix<2, 4, fp>* Bs, fp* detJ, int* elem, fp* E_c, fp* nu_c, const fp* h, int count, int elemPos) {
	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= count) return;

	StaticMatrix<3, 3, fp> Ce;
	(*formC)(Ce, E_c[e], nu_c[e]);
	for (int i = 0; i < 9; ++i)
		C[elemPos + e].data[i] = Ce.data[i];
	SymMatrix<8, fp>& Ke = Ke4[e];
	for (int i = 0; i < Ke.dataSize(); ++i)
		Ke.data[i] = {};
	formKe4(Ke, e, Bs, detJ, Ce, h);

	int begin = e * 4;
	for (int i = 0; i < 4; ++i) {
		int gi = 2 * elem[begin + i];
		atomicAdd(K + strip(gi, gi, W), Ke(2 * i, 2 * i));
		atomicAdd(K + strip(gi + 1, gi, W), Ke(2 * i + 1, 2 * i));
		atomicAdd(K + strip(gi, gi + 1, W), Ke(2 * i + 1, 2 * i));
		atomicAdd(K + strip(gi + 1, gi + 1, W), Ke(2 * i + 1, 2 * i + 1));
	}
	for (int i = 0; i < 4; ++i)
		for (int j = 0; j < i; ++j) {
			int gi = 2 * elem[begin + i];  //глобальне индексы
			int gj = 2 * elem[begin + j];

			atomicAdd(K + strip(gi, gj, W), Ke(2 * i, 2 * j));
			atomicAdd(K + strip(gi + 1, gj, W), Ke(2 * i + 1, 2 * j));
			atomicAdd(K + strip(gi, gj + 1, W), Ke(2 * i, 2 * j + 1));
			atomicAdd(K + strip(gi + 1, gj + 1, W), Ke(2 * i + 1, 2 * j + 1));

			atomicAdd(K + strip(gj, gi, W), Ke(2 * i, 2 * j));
			atomicAdd(K + strip(gj, gi + 1, W), Ke(2 * i + 1, 2 * j));
			atomicAdd(K + strip(gj + 1, gi, W), Ke(2 * i, 2 * j + 1));
			atomicAdd(K + strip(gj + 1, gi + 1, W), Ke(2 * i + 1, 2 * j + 1));
		}
}

template<typename fp>
__global__ static void insertKe8(fp* K, int W, StaticMatrix<3, 3, fp>* C, void (*formC)(StaticMatrix<3, 3, fp>&, fp, fp), SymMatrix<16, fp>* Ke8, \
	StaticMatrix<2, 8, fp>* Bs, fp* detJ, int* elem, fp* E_c, fp* nu_c, fp h, int count, int elemPos) {
	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= count) return;

	StaticMatrix<3, 3, fp> Ce;
	(*formC)(Ce, E_c[e], nu_c[e]);
	for (int i = 0; i < 9; ++i)
		C[elemPos + e].data[i] = Ce.data[i];
	SymMatrix<16, fp>& Ke = Ke8[e];
	for (int i = 0; i < Ke.dataSize(); ++i)
		Ke.data[i] = {};
	formKe8(Ke, e, Bs, detJ, Ce, h);

	int begin = e * 8;
	for (int i = 0; i < 8; ++i) {
		int gi = 2 * elem[begin + i];
		atomicAdd(K + strip(gi, gi, W), Ke(2 * i, 2 * i));
		atomicAdd(K + strip(gi + 1, gi, W), Ke(2 * i + 1, 2 * i));
		atomicAdd(K + strip(gi, gi + 1, W), Ke(2 * i + 1, 2 * i));
		atomicAdd(K + strip(gi + 1, gi + 1, W), Ke(2 * i + 1, 2 * i + 1));
	}
	for (int i = 0; i < 8; ++i)
		for (int j = 0; j < i; ++j) {
			int gi = 2 * elem[begin + i];  //глобальне индексы
			int gj = 2 * elem[begin + j];

			atomicAdd(K + strip(gi, gj, W), Ke(2 * i, 2 * j));
			atomicAdd(K + strip(gi + 1, gj, W), Ke(2 * i + 1, 2 * j));
			atomicAdd(K + strip(gi, gj + 1, W), Ke(2 * i, 2 * j + 1));
			atomicAdd(K + strip(gi + 1, gj + 1, W), Ke(2 * i + 1, 2 * j + 1));

			atomicAdd(K + strip(gj, gi, W), Ke(2 * i, 2 * j));
			atomicAdd(K + strip(gj, gi + 1, W), Ke(2 * i + 1, 2 * j));
			atomicAdd(K + strip(gj + 1, gi, W), Ke(2 * i, 2 * j + 1));
			atomicAdd(K + strip(gj + 1, gi + 1, W), Ke(2 * i + 1, 2 * j + 1));
		}
}

template<typename fp>
__global__ static void insertKe8(fp* K, int W, StaticMatrix<3, 3, fp>* C, void (*formC)(StaticMatrix<3, 3, fp>&, fp, fp), SymMatrix<16, fp>* Ke8, \
	StaticMatrix<2, 8, fp>* Bs, fp* detJ, int* elem, fp* E_c, fp* nu_c, const fp* h, int count, int elemPos) {
	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= count) return;

	StaticMatrix<3, 3, fp> Ce;
	(*formC)(Ce, E_c[e], nu_c[e]);
	for (int i = 0; i < 9; ++i)
		C[elemPos + e].data[i] = Ce.data[i];
	SymMatrix<16, fp>& Ke = Ke8[e];
	for (int i = 0; i < Ke.dataSize(); ++i)
		Ke.data[i] = {};
	formKe8(Ke, e, Bs, detJ, Ce, h);

	int begin = e * 8;
	for (int i = 0; i < 8; ++i) {
		int gi = 2 * elem[begin + i];
		atomicAdd(K + strip(gi, gi, W), Ke(2 * i, 2 * i));
		atomicAdd(K + strip(gi + 1, gi, W), Ke(2 * i + 1, 2 * i));
		atomicAdd(K + strip(gi, gi + 1, W), Ke(2 * i + 1, 2 * i));
		atomicAdd(K + strip(gi + 1, gi + 1, W), Ke(2 * i + 1, 2 * i + 1));
	}
	for (int i = 0; i < 8; ++i)
		for (int j = 0; j < i; ++j) {
			int gi = 2 * elem[begin + i];  //глобальне индексы
			int gj = 2 * elem[begin + j];

			atomicAdd(K + strip(gi, gj, W), Ke(2 * i, 2 * j));
			atomicAdd(K + strip(gi + 1, gj, W), Ke(2 * i + 1, 2 * j));
			atomicAdd(K + strip(gi, gj + 1, W), Ke(2 * i, 2 * j + 1));
			atomicAdd(K + strip(gi + 1, gj + 1, W), Ke(2 * i + 1, 2 * j + 1));

			atomicAdd(K + strip(gj, gi, W), Ke(2 * i, 2 * j));
			atomicAdd(K + strip(gj, gi + 1, W), Ke(2 * i + 1, 2 * j));
			atomicAdd(K + strip(gj + 1, gi, W), Ke(2 * i, 2 * j + 1));
			atomicAdd(K + strip(gj + 1, gi + 1, W), Ke(2 * i + 1, 2 * j + 1));
		}
}
#endif


void PlasticitySolver::fillGlobalStiffness(CudaSLAE<double>& K, CudaSparseSLAE<double>& spK) {
	spK.clearStrip(K);
	const int locBS = 512;
	if (m.h != 0.) {
		if (mesh.count4)
			insertKe4<double><<<(mesh.count4 + locBS - 1) / locBS, locBS>>>(K.matrix, K.W, dd_C, dd_formC, dd_Ke4, dd_B4, dd_detJ4, mesh.dev_elem4, dd_E_c, dd_nu_c, m.h, mesh.count4, mesh.elemPos[0]);
		if (mesh.count8)
			insertKe8<double><<<(mesh.count8 + 512 - 1) / 512, 512>>>(K.matrix, K.W, dd_C, dd_formC, dd_Ke8, dd_B8, dd_detJ8, mesh.dev_elem8, dd_E_c, dd_nu_c, m.h, mesh.count8, mesh.elemPos[1]);
	}
	else {
		if (mesh.count4)
			insertKe4<double><<<(mesh.count4 + locBS - 1) / locBS, locBS>>>(K.matrix, K.W, dd_C, dd_formC, dd_Ke4, dd_B4, dd_detJ4, mesh.dev_elem4, dd_E_c, dd_nu_c, dd_h, mesh.count4, mesh.elemPos[0]);
		if (mesh.count8)
			insertKe8<double><<<(mesh.count8 + 512 - 1) / 512, 512>>>(K.matrix, K.W, dd_C, dd_formC, dd_Ke8, dd_B8, dd_detJ8, mesh.dev_elem8, dd_E_c, dd_nu_c, dd_h, mesh.count8, mesh.elemPos[1]);
	}
	cudaDeviceSynchronize();
	spK.copy(K);
}

void PlasticitySolver::fillGlobalStiffness(CudaSLAE<float>& K, CudaSparseSLAE<float>& spK) {
	spK.clearStrip(K);
	const int locBS = 512;
	if (m.h != 0.) {
		if (mesh.count4)
			insertKe4<float><<<(mesh.count4 + locBS - 1) / locBS, locBS>>>(K.matrix, K.W, df_C, df_formC, df_Ke4, df_B4, df_detJ4, mesh.dev_elem4, df_E_c, df_nu_c, m.h, mesh.count4, mesh.elemPos[0]);
		if (mesh.count8)
			insertKe8<float><<<(mesh.count8 + locBS - 1) / locBS, locBS>>>(K.matrix, K.W, df_C, df_formC, df_Ke8, df_B8, df_detJ8, mesh.dev_elem8, df_E_c, df_nu_c, m.h, mesh.count8, mesh.elemPos[1]);
	}
	else {
		if (mesh.count4)
			insertKe4<float><<<(mesh.count4 + locBS - 1) / locBS, locBS>>>(K.matrix, K.W, df_C, df_formC, df_Ke4, df_B4, df_detJ4, mesh.dev_elem4, df_E_c, df_nu_c, df_h, mesh.count4, mesh.elemPos[0]);
		if (mesh.count8)
			insertKe8<float><<<(mesh.count8 + locBS - 1) / locBS, locBS>>>(K.matrix, K.W, df_C, df_formC, df_Ke8, df_B8, df_detJ8, mesh.dev_elem8, df_E_c, df_nu_c, df_h, mesh.count8, mesh.elemPos[1]);
	}
	cudaDeviceSynchronize();
	spK.copy(K);
}

//template<typename fp>
//__global__ void paramDouble3(double* uv, int* elem, int count, StaticMatrix<2, 3, double>* Bs, \
	double* exx, double* eyy, double* gamma, double* intensityE, \
	double* sxx, double* syy, double* tau, double* intensityS, double* tableS, \
	double* psi, double* E_c, double* nu_c, double E, double nu, /*F diagram*/ double sigmaT, double K_T);


template<typename fp>
__global__ static void strain4(fp* uv, unsigned* elem, unsigned elemPos, unsigned count, StaticMatrix<2, 4, fp>* Bs, \
	fp* exx, fp* eyy, fp* gamma) {

	unsigned e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= count) return;

	unsigned begin = 4 * e;

	fp u1 = uv[2 * elem[begin]],
		v1 = uv[2 * elem[begin] + 1],
		u2 = uv[2 * elem[begin + 1]],
		v2 = uv[2 * elem[begin + 1] + 1],
		u3 = uv[2 * elem[begin + 2]],
		v3 = uv[2 * elem[begin + 2] + 1],
		u4 = uv[2 * elem[begin + 3]],
		v4 = uv[2 * elem[begin + 3] + 1];

	e += elemPos;
	exx[e] = {};
	eyy[e] = {};
	gamma[e] = {};
	for (unsigned i = 0; i < 4; ++i) {
		StaticMatrix<2, 4, fp>& B = Bs[begin + i];
		exx[e] += u1 * B(0, 0) + u2 * B(0, 1) + u3 * B(0, 2) + u4 * B(0, 3);
		eyy[e] += v1 * B(1, 0) + v2 * B(1, 1) + v3 * B(1, 2) + v4 * B(1, 3);
		gamma[e] += u1 * B(1, 0) + u2 * B(1, 1) + u3 * B(1, 2) + u4 * B(1, 3) + \
			v1 * B(0, 0) + v2 * B(0, 1) + v3 * B(0, 2) + v4 * B(0, 3);
	}
	exx[e] *= 0.25f;
	eyy[e] *= 0.25f;
	gamma[e] *= 0.25f;
}

template<typename fp>
__global__ static void strain8(fp* uv, unsigned* elem, unsigned elemPos, unsigned count, StaticMatrix<2, 8, fp>* Bs, \
	fp* exx, fp* eyy, fp* gamma) {

	const fp GS_coef[3] = { 0.555'555'555'555'555, 0.888'888'888'888'888, 0.555'555'555'555'555 };
	//const fp GS_coef[4] = { 0.3478548451374538, 0.652145154862546, 0.652145154862546, 0.3478548451374538 };

	unsigned e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= count) return;
	unsigned begin = 8 * e;
	fp u[8], v[8];
	for (unsigned i = 0; i < 8; ++i) {
		u[i] = uv[2 * elem[begin + i]];
		v[i] = uv[2 * elem[begin + i] + 1];
	}
	begin = secIntPs * secIntPs * e;
	e += elemPos;
	exx[e] = {};
	eyy[e] = {};
	gamma[e] = {};
	for (unsigned i = 0; i < secIntPs; ++i)
		for (unsigned j = 0; j < secIntPs; ++j) {
			fp exxLoc = {}, eyyLoc = {}, gammaLoc = {};
			StaticMatrix<2, 8, fp>& B = Bs[begin + i * secIntPs + j];
			for (unsigned k = 0; k < 8; ++k) {
				exxLoc += u[k] * B(0, k);
				eyyLoc += v[k] * B(1, k);
				gammaLoc += u[k] * B(1, k) + v[k] * B(0, k);
			}
			fp coef = GS_coef[i] * GS_coef[j];
			exx[e] += exxLoc * coef;
			eyy[e] += eyyLoc * coef;
			gamma[e] += gammaLoc * coef;
		}
	exx[e] *= 0.25f;
	eyy[e] *= 0.25f;
	gamma[e] *= 0.25f;
}

template<typename fp>
__global__ static void plastStress(unsigned count, StaticMatrix<3, 3, fp>* C_, \
	fp* exx, fp* eyy, fp* gamma, fp* intensityE, \
	fp* sxx, fp* syy, fp* tau, fp* intensityS, fp* tableS, \
	fp* psi, fp* E_c, fp* nu_c, Material* m) {

	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= count) return;

	sxx[e] = C_[e](0, 0) * exx[e] + C_[e](0, 1) * eyy[e];
	syy[e] = C_[e](0, 1) * exx[e] + C_[e](1, 1) * eyy[e];
	tau[e] = C_[e](2, 2) * gamma[e];

	intensityS[e] = sqrt(0.5f * (sxx[e] * sxx[e] + syy[e] * syy[e] + (sxx[e] - syy[e]) * (sxx[e] - syy[e]) + 6.f * tau[e] * tau[e]));
	intensityE[e] = intensityS[e] / E_c[e];
	tableS[e] = fp(m->dev_f(intensityE[e]));

	psi[e] = (fp)m->E * intensityE[e] / tableS[e];
	E_c[e] = (fp)m->E / psi[e];
	nu_c[e] = 0.5f * (1.f - (1.f - 2.f * (fp)m->nu) / psi[e]);
}


void PlasticitySolver::updateParamDouble_() {
	int grid = (mesh.count4 + BS - 1) / BS;
	int grid8 = (mesh.count8 + BS - 1) / BS;
	
	if (mesh.count4)
		strain4<<<grid, BS>>>(dd_uv, (unsigned*)mesh.dev_elem4, mesh.elemPos[0], mesh.count4, dd_B4, \
			dd_exx, dd_eyy, dd_gamma);
	if (mesh.count8)
		strain8<<<grid8, BS>>>(dd_uv, (unsigned*)mesh.dev_elem8, mesh.elemPos[1], mesh.count8, dd_B8, \
			dd_exx, dd_eyy, dd_gamma);
	cudaDeviceSynchronize();
	plastStress<<<(mesh.elemCount() + 512 - 1) / 512, 512>>>(mesh.elemCount(), dd_C, dd_exx, dd_eyy, dd_gamma, dd_intensityE, \
		dd_sxx, dd_syy, dd_tau, dd_intensityS, dd_tableS, dd_psi, dd_E_c, dd_nu_c, dev_m);
	cudaDeviceSynchronize();
}

void PlasticitySolver::updateParamFloat_() {
	int grid = (mesh.count4 + BS - 1) / BS;
	int grid8 = (mesh.count8 + BS - 1) / BS;
	
	if (mesh.count4)
		strain4<<<grid, BS>>>(df_uv, (unsigned*)mesh.dev_elem4, mesh.elemPos[0], mesh.count4, df_B4, \
			df_exx, df_eyy, df_gamma);
	if (mesh.count8)
		strain8<<<grid8, BS>>>(df_uv, (unsigned*)mesh.dev_elem8, mesh.elemPos[1], mesh.count8, df_B8, \
			df_exx, df_eyy, df_gamma);
	cudaDeviceSynchronize();
	plastStress<<<(mesh.elemCount() + 512 - 1) / 512, 512>>>(mesh.elemCount(), df_C, df_exx, df_eyy, df_gamma, df_intensityE, \
		df_sxx, df_syy, df_tau, df_intensityS, df_tableS, df_psi, df_E_c, df_nu_c, dev_m);
	cudaDeviceSynchronize();
}


void PlasticitySolver::updateParamDouble() {
	const unsigned block = 256;
	for (unsigned t = 0; t < mesh.elemTypes; ++t) {
		const FiniteElement& type = mesh.elemInfo[t];
		unsigned grid = (type.elemCount + block - 1) / block;
		switch (type.elemType) {
		case ElemType::quad1:
			strain4<double> << <grid, block >> > (dd_u.raw(), cudaMesh.elem + type.memIdx, \
				type.poolIdx, type.elemCount, \
				(StaticMatrix<2, 4, double>*)(dd_B + Bidx[t]), \
				dd_exx, dd_eyy, dd_gamma);
			break;
		case ElemType::quad2:
			strain8<double> << <grid, block >> > (dd_u.raw(), cudaMesh.elem + type.memIdx, \
				type.poolIdx, type.elemCount, \
				(StaticMatrix<2, 8, double>*)(dd_B + Bidx[t]), \
				dd_exx, dd_eyy, dd_gamma);
			break;
		}
	}
	cudaDeviceSynchronize();
	plastStress << <(mesh.elemCount() + 512 - 1) / 512, 512 >> > (mesh.elemCount(), dd_C, dd_exx, dd_eyy, dd_gamma, dd_intensityE, \
		dd_sxx, dd_syy, dd_tau, dd_intensityS, dd_tableS, dd_psi, dd_E_c, dd_nu_c, dev_m);
	cudaDeviceSynchronize();
}


void PlasticitySolver::updateParamFloat() {
	const unsigned block = 256;
	for (unsigned t = 0; t < mesh.elemTypes; ++t) {
		const FiniteElement& type = mesh.elemInfo[t];
		unsigned grid = (type.elemCount + block - 1) / block;
		switch (type.elemType) {
		case ElemType::quad1:
			strain4<float><<<grid, block>>>(df_u.raw(), cudaMesh.elem + type.memIdx, \
				type.poolIdx, type.elemCount, \
				(StaticMatrix<2, 4, float>*)(df_B + Bidx[t]), \
				df_exx, df_eyy, df_gamma);
			break;
		case ElemType::quad2:
			strain8<float><<<grid, block>>>(df_u.raw(), cudaMesh.elem + type.memIdx, \
				type.poolIdx, type.elemCount, \
				(StaticMatrix<2, 8, float>*)(df_B + Bidx[t]), \
				df_exx, df_eyy, df_gamma);
			break;
		}
	}
	cudaDeviceSynchronize();
	plastStress<<<(mesh.elemCount() + 512 - 1) / 512, 512>>>(mesh.elemCount(), df_C, df_exx, df_eyy, df_gamma, df_intensityE, \
		df_sxx, df_syy, df_tau, df_intensityS, df_tableS, df_psi, df_E_c, df_nu_c, dev_m);
	cudaDeviceSynchronize();
}


__global__ static void plastDeform(int count, double* exx_p, double* eyy_p, double* gamma_p, double* intE_p, \
	double* sxx, double* syy, double* tau, \
	double* psi, double* E_c, double* nu_c, double E, double nu, double G) {
	
	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e >= count) return;
	exx_p[e] = (1. / E_c[e] - 1. / E) * sxx[e];
	eyy_p[e] = (1. / E_c[e] - 1. / E) * syy[e];
	double G_c = E_c[e] * 0.5 / (1. + nu_c[e]);
	gamma_p[e] = (1. / G_c - 1. / G) * tau[e];
	// скорее всего неправильно
	//intE_p[e] = sqrt(2.) / 3. * sqrt((exx_p[e] - eyy_p[e]) * (exx_p[e] - eyy_p[e]) + 6. * gamma_p[e] * gamma_p[e]);
	
	intE_p[e] = sqrt(2.) / 3. * sqrt((exx_p[e] - eyy_p[e]) * (exx_p[e] - eyy_p[e]) + 1.5 * gamma_p[e] * gamma_p[e]);
	
	psi[e] = E / E_c[e];
}

void PlasticitySolver::calcPlastDeformCuda() {
	double G = m.E * 0.5 / (1. + m.nu);
	int grid = (mesh.elemCount() + BS - 1) / BS;
	plastDeform<<<grid, BS>>>(mesh.elemCount(), dd_exx_p, dd_eyy_p, dd_gamma_p, dd_intE_p, \
		dd_sxx, dd_syy, dd_tau, \
		dd_psi, dd_E_c, dd_nu_c, m.E, m.nu, G);
	cudaDeviceSynchronize();
}

#if __CUDA_ARCH__ < 600
template<typename fp>
__global__ static void exitNorm2(fp* intS, fp* tableS, fp* norm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ fp sh_dif[BS];
	__shared__ fp sh_tab[BS];

	fp dif = intS[i] - tableS[i];
	sh_dif[tid] = dif * dif;
	sh_tab[tid] = tableS[i] * tableS[i];
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sh_dif[tid] += sh_dif[tid + s];
			sh_tab[tid] += sh_tab[tid + s];
		}
		__syncthreads();
	}
	if (tid < 32) {
		warpReduce(sh_dif, tid);
		warpReduce(sh_tab, tid);
	}
	if (tid == 0) {
		atomicAdd_arch52(norm, sh_dif[0]);
		atomicAdd_arch52(norm + 1, sh_tab[0]);
	}
}

template<typename fp>
__global__ static void exitNorm2_(fp* intS, fp* tableS, fp* norm, int count) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ fp sh_dif[BS];
	__shared__ fp sh_tab[BS];

	fp dif = intS[i] - tableS[i];
	sh_dif[tid] = (i < count) ? dif * dif : (fp)0.;
	sh_tab[tid] = tableS[i] * tableS[i];
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sh_dif[tid] += sh_dif[tid + s];
			sh_tab[tid] += sh_tab[tid + s];
		}
		__syncthreads();
	}
	if (tid < 32) {
		warpReduce(sh_dif, tid);
		warpReduce(sh_tab, tid);
	}
	if (tid == 0) {
		atomicAdd_arch52(norm, sh_dif[0]);
		atomicAdd_arch52(norm + 1, sh_tab[0]);
	}
}
#else
template<typename fp>
__global__ static void exitNorm2(fp* intS, fp* tableS, fp* norm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ fp sh_dif[BS];
	__shared__ fp sh_tab[BS];

	fp dif = intS[i] - tableS[i];
	sh_dif[tid] = dif * dif;
	sh_tab[tid] = tableS[i] * tableS[i];
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sh_dif[tid] += sh_dif[tid + s];
			sh_tab[tid] += sh_tab[tid + s];
		}
		__syncthreads();
	}
	if (tid < 32) {
		warpReduce(sh_dif, tid);
		warpReduce(sh_tab, tid);
	}
	if (tid == 0) {
		atomicAdd(norm, sh_dif[0]);
		atomicAdd(norm + 1, sh_tab[0]);
	}
}

template<typename fp>
__global__ static void exitNorm2_(fp* intS, fp* tableS, fp* norm, int count) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int tid = threadIdx.x;
	__shared__ fp sh_dif[BS];
	__shared__ fp sh_tab[BS];

	fp dif = intS[i] - tableS[i];
	sh_dif[tid] = (i < count) ? dif * dif : (fp)0.;
	sh_tab[tid] = tableS[i] * tableS[i];
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s) {
			sh_dif[tid] += sh_dif[tid + s];
			sh_tab[tid] += sh_tab[tid + s];
		}
		__syncthreads();
	}
	if (tid < 32) {
		warpReduce(sh_dif, tid);
		warpReduce(sh_tab, tid);
	}
	if (tid == 0) {
		atomicAdd(norm, sh_dif[0]);
		atomicAdd(norm + 1, sh_tab[0]);
	}
}
#endif

template<typename fp>
fp PlasticitySolver::exitCondition(fp* intS, fp* tableS, fp* dev_norm) {
	/*fp zeros[2] = {};
	cudaMemcpy(dev_norm, zeros, 2 * sizeof(fp), cudaMemcpyHostToDevice);*/
	cudaMemset(dev_norm, 0, 2 * sizeof(fp));
	int grid = (mesh.elemCount() + BS - 1) / BS;
	exitNorm2<<<grid, BS>>>(intS, tableS, dev_norm);
	//exitNorm2_<<<grid, BS>>>(intS, tableS, dev_norm, mesh.elemCount());
	cudaDeviceSynchronize();
	fp norm[2] = {};
	cudaMemcpy(norm, dev_norm, 2 * sizeof(fp), cudaMemcpyDeviceToHost);
	//std::cout << norm[0] << " " << norm[1] << "\n";
	return sqrt(norm[0] / norm[1]);
	//return (norm[0] / norm[1]);
}


void PlasticitySolver::dataToRAM() {
	/*exx = new double[mesh.count4];
	eyy = new double[mesh.count4];
	gamma = new double[mesh.count4];
	intensityE = new double[mesh.count4];
	exx_p = new double[mesh.count4];
	eyy_p = new double[mesh.count4];
	gamma_p = new double[mesh.count4];
	intE_p = new double[mesh.count4];
	sxx = new double[mesh.count4];
	syy = new double[mesh.count4];
	tau = new double[mesh.count4];
	intensityS = new double[mesh.count4];
	tableS = new double[mesh.count4];
	psi = new double[mesh.count4];
	E_c = new double[mesh.count4];
	nu_c = new double[mesh.count4];*/
	cudaMemcpy(exx, dd_exx, mesh.elemCount() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(eyy, dd_eyy, mesh.elemCount() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(gamma, dd_gamma, mesh.elemCount() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(intensityE, dd_intensityE, mesh.elemCount() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(exx_p, dd_exx_p, mesh.elemCount() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(eyy_p, dd_eyy_p, mesh.elemCount() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(gamma_p, dd_gamma_p, mesh.elemCount() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(intE_p, dd_intE_p, mesh.elemCount() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(sxx, dd_sxx, mesh.elemCount() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(syy, dd_syy, mesh.elemCount() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(tau, dd_tau, mesh.elemCount() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(intensityS, dd_intensityS, mesh.elemCount() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(tableS, dd_tableS, mesh.elemCount() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(psi, dd_psi, mesh.elemCount() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(E_c, dd_E_c, mesh.elemCount() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy(nu_c, dd_nu_c, mesh.elemCount() * sizeof(double), cudaMemcpyDeviceToHost);

	uv.resize(2 * mesh.nodeCount);
	if (dd_u)  // TEMP
		cudaMemcpy(uv.data(), dd_u, 2 * mesh.nodeCount * sizeof(double), cudaMemcpyDeviceToHost);
	else
		cudaMemcpy(uv.data(), dd_uv, 2 * mesh.nodeCount * sizeof(double), cudaMemcpyDeviceToHost);

	/*double* locSol = new double[2 * mesh.nodeCount];
	cudaMemcpy(locSol, dd_uv, 2 * mesh.nodeCount * sizeof(double), cudaMemcpyDeviceToHost);
	std::cout << "\n";
	for (int i = 0; i < 2 * mesh.nodeCount; ++i)
		std::cout << locSol[i] << " ";
	delete[] locSol;*/
}

//Сохранить в файл формата .vtk для отображения результатов в ParaView
void PlasticitySolver::saveAsVtk(const std::string& fileName) {
	if (!mesh.ramSaved)
		mesh.meshToRAM();
	if (!ramSaved)
		dataToRAM();

	std::ofstream file(fileName, std::ios_base::out);
	file << "# vtk DataFile Version 2.0\n";
	file << "Solution\n";
	file << "ASCII\n";
	file << "DATASET POLYDATA\n";
	file << "POINTS " << mesh.nodeCount << " float\n";
	for (size_t i = 0; i < mesh.nodeCount; ++i)
		file << mesh.node2[i].x << " " << mesh.node2[i].y << " " << 0 << "\n";
	file << "POLYGONS " << mesh.elemCount() << " " << 4 * mesh.count3 + 5 * mesh.count4 + 9 * mesh.count8;
	for (size_t i = 0; i < mesh.count3; ++i) {
		file << "\n3 ";
		for (size_t j = 3 * i; j < 3 * (i + 1); ++j)
			file << mesh.elem3[j] << " ";
	}
	for (size_t i = 0; i < mesh.count4; ++i) {
		file << "\n4 ";
		for (size_t j = 4 * i; j < 4 * (i + 1); ++j)
			file << mesh.elem4[j] << " ";
	}
	for (size_t i = 0; i < mesh.count8; ++i) {
		file << "\n8 ";
		for (size_t j = 8 * i; j < 8 * i + 4; ++j)
			file << mesh.elem8[j] << " " << mesh.elem8[j + 4] << " ";
	}
	file << "\nCELL_DATA " << mesh.elemCount() << "\n";
	file << "SCALARS ElementID int 1\n";
	file << "LOOKUP_TABLE default\n";
	for (size_t i = 0; i < mesh.elemCount(); ++i) {
		file << i << "\n";
	}

	//file.precision(16);
	int fields = 8 + (plastSolved ? 7 : 0) + (mesh.analysed ? 3 : 0);
	file << "FIELD FieldData " << fields;
	if (polarCoord) {
		//std::vector<double> exy(gamma);
		double* exy = new double[mesh.count4];
		memcpy(exy, gamma, mesh.count4 * sizeof(double));
		/*for (auto& e : exy)
			e *= 0.5;*/
		for (int i = 0; i < mesh.count4; ++i)
			exy[i] *= 0.5;
		file << "\nEps_rr 1 " << mesh.count4 << " float\n";
		for (size_t i = 0; i < mesh.count4; ++i)
			file << _rr(i, exx, eyy, exy) << " ";
		file << "\nEps_phiphi 1 " << mesh.count4 << " float\n";
		for (size_t i = 0; i < mesh.count4; ++i)
			file << _phiphi(i, exx, eyy, exy) << " ";
		file << "\nGamma_rphi 1 " << mesh.count4 << " float\n";
		for (size_t i = 0; i < mesh.count4; ++i)
			file << _rphi(i, exx, eyy, exy) << " ";
		delete[] exy;
	}
	else {
		file << "\nEps_xx 1 " << mesh.elemCount() << " float\n";
		for (size_t i = 0; i < mesh.elemCount(); ++i)
			file << exx[i] << " ";
		file << "\nEps_yy 1 " << mesh.elemCount() << " float\n";
		for (size_t i = 0; i < mesh.elemCount(); ++i)
			file << eyy[i] << " ";
		file << "\nGamma_xy 1 " << mesh.elemCount() << " float\n";
		for (size_t i = 0; i < mesh.elemCount(); ++i)
			file << gamma[i] << " ";
	}
	file << "\nEps_intensity 1 " << mesh.elemCount() << " float\n";
	for (size_t i = 0; i < mesh.elemCount(); ++i)
		file << intensityE[i] << " ";
	if (plastSolved) {
		if (polarCoord) {
			/*std::vector<double> exy_p(gamma_p);
			for (auto& e : exy_p)
				e *= 0.5;*/
			double* exy_p = new double[mesh.count4];
			memcpy(exy_p, gamma_p, mesh.count4 * sizeof(double));
			for (int i = 0; i < mesh.count4; ++i)
				exy_p[i] *= 0.5;
			file << "\nEps_p_rr 1 " << mesh.count4 << " float\n";
			for (size_t i = 0; i < mesh.count4; ++i)
				file << _rr(i, exx_p, eyy_p, exy_p) << " ";
			file << "\nEps_p_phiphi 1 " << mesh.count4 << " float\n";
			for (size_t i = 0; i < mesh.count4; ++i)
				file << _phiphi(i, exx_p, eyy_p, exy_p) << " ";
			file << "\nGamma_p_rphi 1 " << mesh.count4 << " float\n";
			for (size_t i = 0; i < mesh.count4; ++i)
				file << _rphi(i, exx_p, eyy_p, exy_p) << " ";
			delete[] exy_p;
		}
		else {
			file << "\nEps_p_xx 1 " << mesh.elemCount() << " float\n";
			for (size_t i = 0; i < mesh.elemCount(); ++i)
				file << exx_p[i] << " ";
			file << "\nEps_p_yy 1 " << mesh.elemCount() << " float\n";
			for (size_t i = 0; i < mesh.elemCount(); ++i)
				file << eyy_p[i] << " ";
			file << "\nGamma_p_xy 1 " << mesh.elemCount() << " float\n";
			for (size_t i = 0; i < mesh.elemCount(); ++i)
				file << gamma_p[i] << " ";
		}
		file << "\nEps_p_intensity 1 " << mesh.elemCount() << " float\n";
		for (size_t i = 0; i < mesh.elemCount(); ++i)
			file << intE_p[i] << " ";
		file << "\nPsi 1 " << mesh.elemCount() << " float\n";
		for (size_t i = 0; i < mesh.elemCount(); ++i)
			file << psi[i] << " ";
		file << "\nE_c 1 " << mesh.elemCount() << " float\n";
		for (size_t i = 0; i < mesh.elemCount(); ++i)
			file << E_c[i] << " ";
		file << "\nnu_c 1 " << mesh.elemCount() << " float\n";
		for (size_t i = 0; i < mesh.elemCount(); ++i)
			file << nu_c[i] << " ";
	}
	if (polarCoord) {
		file << "\nSigma_rr 1 " << mesh.count4 << " float\n";
		for (size_t i = 0; i < mesh.count4; ++i)
			file << _rr(i, sxx, syy, tau) << " ";
		file << "\nSigma_phiphi 1 " << mesh.count4 << " float\n";
		for (size_t i = 0; i < mesh.count4; ++i)
			file << _phiphi(i, sxx, syy, tau) << " ";
		file << "\nTau_rphi 1 " << mesh.count4 << " float\n";
		for (size_t i = 0; i < mesh.count4; ++i)
			file << _rphi(i, sxx, syy, tau) << " ";
	}
	else {
		file << "\nSigma_xx 1 " << mesh.elemCount() << " float\n";
		for (size_t i = 0; i < mesh.elemCount(); ++i)
			file << sxx[i] << " ";
		file << "\nSigma_yy 1 " << mesh.elemCount() << " float\n";
		for (size_t i = 0; i < mesh.elemCount(); ++i)
			file << syy[i] << " ";
		file << "\nTau_xy 1 " << mesh.elemCount() << " float\n";
		for (size_t i = 0; i < mesh.elemCount(); ++i)
			file << tau[i] << " ";
	}
	file << "\nSigma_intensity 1 " << mesh.elemCount() << " float\n";
	for (size_t i = 0; i < mesh.elemCount(); ++i)
		file << intensityS[i] << " ";
	if (mesh.analysed) {
		file << "\nElement_space 1 " << mesh.count4 << " float\n";
		for (size_t i = 0; i < mesh.count4; ++i)
			file << mesh.spaces[i] << " ";
		file << "\nElement_aspect_ratio 1 " << mesh.count4 << " float\n";
		for (size_t i = 0; i < mesh.count4; ++i)
			file << mesh.aspects[i] << " ";
		file << "\nElement_skew_angle 1 " << mesh.count4 << " float\n";
		for (size_t i = 0; i < mesh.count4; ++i)
			file << mesh.skewAngles[i] << " ";
	}

	file << "\nPOINT_DATA " << mesh.nodeCount << \
		"\nSCALARS NodeID int 1\nLOOKUP_TABLE my_table";
	for (size_t i = 0; i < mesh.nodeCount; ++i)
		file << "\n" << i;
	file << "\nFIELD FieldData2 2\nDisplacement 3 " << mesh.nodeCount << " float\n";
	for (size_t i = 0; i < mesh.nodeCount; ++i)
		file << uv[2 * i] << " " << uv[2 * i + 1] << " 0 ";
	file << "\nThickness 1 " << mesh.nodeCount << " float\n";
	if (m.h != 0.)
		for (size_t i = 0; i < mesh.nodeCount; ++i)
			file << m.h << " ";
	else
		for (size_t i = 0; i < mesh.nodeCount; ++i)
			file << m.hf(mesh.node2[i]) << " ";

	file.close();
}


//Сохранить в файл формата .vtu для отображения результатов в ParaView
//TODO: add polar and nodal data, mesh quality
void PlasticitySolver::saveAsVtu(const std::string& fileName) {
	std::cout << "\nSaving...\n";
	double t = -omp_get_wtime();

	if (!mesh.ramSaved)
		mesh.meshToRAM();
	if (!ramSaved)
		dataToRAM();

	std::ofstream file(fileName, std::ios_base::out);

	file << "<?xml version=\"1.0\"?>" \
		<< "\n<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt64\" >" \
		<< "\n\t<UnstructuredGrid>" \
		<< "\n\t\t<Piece NumberOfPoints=\"" << mesh.nodeCount << "\" NumberOfCells=\"" << mesh.elemCount() << "\">";

	file << "\n\t\t\t<PointData>";  // Данные в узлах
	// Номера узлов
	file << "\n\t\t\t\t<DataArray type=\"UInt32\" Name=\"NodeID\" NumberOfComponents=\"1\" format=\"ascii\" >\n";
	for (unsigned i = 0; i < mesh.nodeCount; ++i)
		file << i << " ";
	file << "\n\t\t\t\t</DataArray>";
	// Перемещения
	file << "\n\t\t\t\t<DataArray type=\"Float64\" Name=\"Displacement\" NumberOfComponents=\"3\" format=\"ascii\" >\n";
	for (unsigned i = 0; i < mesh.nodeCount; ++i)
		file << uv[2 * i] << " " << uv[2 * i + 1] << " 0 ";
	file << "\n\t\t\t\t</DataArray>";
	file << "\n\t\t\t</PointData>";

	file << "\n\t\t\t<CellData>";  // Данные в элементах
	// Номера элементов
	file << "\n\t\t\t\t<DataArray type=\"UInt32\" Name=\"ElementID\" NumberOfComponents=\"1\" format=\"ascii\" >\n";
	for (unsigned i = 0; i < mesh.elemCount(); ++i)
		file << i << " ";
	file << "\n\t\t\t\t</DataArray>";
	// Деформации
	file << "\n\t\t\t\t<DataArray type=\"Float64\" Name=\"Strain" << (polarCoord ? " (Cartesian)" : "") << "\" NumberOfComponents=\"4\" " \
		<< "ComponentName0=\"XX\" ComponentName1=\"YY\" ComponentName2=\"XY\" ComponentName3=\"Intensity\"" \
		<< " format=\"ascii\" >\n";
	for (unsigned i = 0; i < mesh.elemCount(); ++i)
		file << exx[i] << " " << eyy[i] << " " << 0.5 * gamma[i] << " " << intensityE[i] << " ";
	file << "\n\t\t\t\t</DataArray>";
	if (plastSolved) {
		// Пластические деформации
		file << "\n\t\t\t\t<DataArray type=\"Float64\" Name=\"Plast Strain" << (polarCoord ? " (Cartesian)" : "") << "\" NumberOfComponents=\"4\" " \
			<< "ComponentName0=\"XX\" ComponentName1=\"YY\" ComponentName2=\"XY\" ComponentName3=\"Intensity\"" \
			<< " format=\"ascii\" >\n";
		for (unsigned i = 0; i < mesh.elemCount(); ++i)
			file << exx_p[i] << " " << eyy_p[i] << " " << 0.5 * gamma_p[i] << " " << intE_p[i] << " ";
		file << "\n\t\t\t\t</DataArray>";
		// Параметры пластичности
		file << "\n\t\t\t\t<DataArray type=\"Float64\" Name=\"Psi\" NumberOfComponents=\"1\" format=\"ascii\" >\n";
		for (unsigned i = 0; i < mesh.elemCount(); ++i)
			file << psi[i] << " ";
		file << "\n\t\t\t\t</DataArray>";
		file << "\n\t\t\t\t<DataArray type=\"Float64\" Name=\"E_c\" NumberOfComponents=\"1\" format=\"ascii\" >\n";
		for (unsigned i = 0; i < mesh.elemCount(); ++i)
			file << E_c[i] << " ";
		file << "\n\t\t\t\t</DataArray>";
		file << "\n\t\t\t\t<DataArray type=\"Float64\" Name=\"nu_c\" NumberOfComponents=\"1\" format=\"ascii\" >\n";
		for (unsigned i = 0; i < mesh.elemCount(); ++i)
			file << nu_c[i] << " ";
		file << "\n\t\t\t\t</DataArray>";
	}
	// Напряжения
	file << "\n\t\t\t\t<DataArray type=\"Float64\" Name=\"Stress" << (polarCoord ? " (Cartesian)" : "") << "\" NumberOfComponents=\"4\" " \
		<< "ComponentName0=\"XX\" ComponentName1=\"YY\" ComponentName2=\"XY\" ComponentName3=\"Intensity\"" \
		<< " format=\"ascii\" >\n";
	for (unsigned i = 0; i < mesh.elemCount(); ++i)
		file << sxx[i] << " " << syy[i] << " " << tau[i] << " " << intensityS[i] << " ";
	file << "\n\t\t\t\t</DataArray>";
	file << "\n\t\t\t</CellData>";

	file << "\n\t\t\t<Points>";  // Узлы
	file << "\n\t\t\t\t<DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"ascii\" >\n";
	for (unsigned i = 0; i < mesh.nodeCount; ++i)
		file << mesh.node2[i].x << " " << mesh.node2[i].y << " 0 ";
	file << "\n\t\t\t\t</DataArray>";
	file << "\n\t\t\t</Points>";

	file << "\n\t\t\t<Cells>";  // Элементы
	file << "\n\t\t\t\t<DataArray type=\"UInt32\" Name=\"connectivity\" format=\"ascii\" >\n";
	for (int i = 0; i < mesh.count3 * 3; ++i)
		file << mesh.elem3[i] << " ";
	for (int i = 0; i < mesh.count4 * 4; ++i)
		file << mesh.elem4[i] << " ";
	for (int i = 0; i < mesh.count8 * 8; ++i)
		file << mesh.elem8[i] << " ";
	file << "\n\t\t\t\t</DataArray>";
	file << "\n\t\t\t\t<DataArray type=\"UInt32\" Name=\"offsets\" format=\"ascii\" >\n";
	unsigned offset = 0;
	for (int i = 0; i < mesh.count3; ++i)
		file << (offset += 3) << " ";
	for (int i = 0; i < mesh.count4; ++i)
		file << (offset += 4) << " ";
	for (int i = 0; i < mesh.count8; ++i)
		file << (offset += 8) << " ";
	file << "\n\t\t\t\t</DataArray>";
	file << "\n\t\t\t\t<DataArray type=\"UInt32\" Name=\"types\" format=\"ascii\" >\n";
	for (int i = 0; i < mesh.count3; ++i)
		file << "5 ";
	for (int i = 0; i < mesh.count4; ++i)
		file << "9 ";
	for (int i = 0; i < mesh.count8; ++i)
		file << "23 ";
	file << "\n\t\t\t\t</DataArray>";
	file << "\n\t\t\t</Cells>";

	file << "\n\t\t</Piece>";
	file << "\n\t</UnstructuredGrid>";
	file << "\n</VTKFile>";

	file.close();
	t += omp_get_wtime();
	std::cout << "Solution saved in file " << fileName << " [" << t << " sec]\n";
}


void PlasticitySolver::saveResidualsAsVtk(const std::string& fileName, \
	std::function<double(vec2)> u_x,
	std::function<double(vec2)> u_y,
	std::function<double(vec2)> sigma_x,
	std::function<double(vec2)> sigma_y,
	std::function<double(vec2)> tau_xy
) {
	if (!mesh.ramSaved)
		mesh.meshToRAM();
	if (!ramSaved)
		dataToRAM();

	double maxDispl = 0.;
	for (int i = 0; i < mesh.nodeCount; ++i) {
		double displ = sqrt(uv[2 * i] * uv[2 * i] + uv[2 * i + 1] * uv[2 * i + 1]);
		if (displ > maxDispl)
			maxDispl = displ;
	}
	double maxStress = 0.;
	for (int e = 0; e < mesh.elemCount(); ++e) {
		double stress = intensityS[e];
		if (stress > maxStress)
			maxStress = stress;
	}
	std::cout << maxDispl << " " << maxStress << "\n";

	std::ofstream file(fileName, std::ios_base::out);
	file << "# vtk DataFile Version 2.0\n";
	file << "Residuals\n";
	file << "ASCII\n";
	file << "DATASET POLYDATA\n";
	file << "POINTS " << mesh.nodeCount << " float\n";
	for (size_t i = 0; i < mesh.nodeCount; ++i)
		file << mesh.node2[i].x << " " << mesh.node2[i].y << " " << 0 << "\n";
	file << "POLYGONS " << mesh.elemCount() << " " << 4 * mesh.count3 + 5 * mesh.count4 + 9 * mesh.count8;
	for (size_t i = 0; i < mesh.count3; ++i) {
		file << "\n3 ";
		for (size_t j = 3 * i; j < 3 * (i + 1); ++j)
			file << mesh.elem3[j] << " ";
	}
	for (size_t i = 0; i < mesh.count4; ++i) {
		file << "\n4 ";
		for (size_t j = 4 * i; j < 4 * (i + 1); ++j)
			file << mesh.elem4[j] << " ";
	}
	for (size_t i = 0; i < mesh.count8; ++i) {
		file << "\n8 ";
		for (size_t j = 8 * i; j < 8 * i + 4; ++j)
			file << mesh.elem8[j] << " " << mesh.elem8[j + 4] << " ";
	}
	file << "\nCELL_DATA " << mesh.elemCount() << "\n";
	file << "SCALARS ElementID int 1\n";
	file << "LOOKUP_TABLE default\n";
	for (size_t i = 0; i < mesh.elemCount(); ++i) {
		file << i << "\n";
	}

	//file.precision(16);
	int fields = 3;
	file << "FIELD FieldData " << fields;
	{
		file << "\nSigma_xx 1 " << mesh.elemCount() << " float\n";
		for (int i = 0; i < mesh.count4; ++i) { //TO DO: other element types
			vec2 r = 0.25 * (mesh.node2[mesh.elem4[4 * i]] + mesh.node2[mesh.elem4[4 * i + 1]] + \
				mesh.node2[mesh.elem4[4 * i + 2]] + mesh.node2[mesh.elem4[4 * i + 3]]);
			file << fabs(sxx[i] - sigma_x(r)) / maxStress << " ";
		}
		file << "\nSigma_yy 1 " << mesh.elemCount() << " float\n";
		for (int i = 0; i < mesh.count4; ++i) { //TO DO: other element types
			vec2 r = 0.25 * (mesh.node2[mesh.elem4[4 * i]] + mesh.node2[mesh.elem4[4 * i + 1]] + \
				mesh.node2[mesh.elem4[4 * i + 2]] + mesh.node2[mesh.elem4[4 * i + 3]]);
			file << fabs(syy[i] - sigma_y(r)) / maxStress << " ";
		}
		file << "\nTau_xy 1 " << mesh.elemCount() << " float\n";
		for (int i = 0; i < mesh.count4; ++i) { //TO DO: other element types
			vec2 r = 0.25 * (mesh.node2[mesh.elem4[4 * i]] + mesh.node2[mesh.elem4[4 * i + 1]] + \
				mesh.node2[mesh.elem4[4 * i + 2]] + mesh.node2[mesh.elem4[4 * i + 3]]);
			file << fabs(tau[i] - tau_xy(r)) / maxStress << " ";
		}
	}

	file << "\nPOINT_DATA " << mesh.nodeCount << \
		"\nSCALARS NodeID int 1\nLOOKUP_TABLE my_table";
	for (int i = 0; i < mesh.nodeCount; ++i)
		file << "\n" << i;
	file << "\nFIELD FieldData2 1\nDisplacement 3 " << mesh.nodeCount << " float\n";
	for (int i = 0; i < mesh.nodeCount; ++i)
		file << fabs(uv[2 * i] - u_x(mesh.node2[i])) / maxDispl \
		<< " " << fabs(uv[2 * i + 1] - u_y(mesh.node2[i])) / maxDispl << " 0 ";

	file.close();
}