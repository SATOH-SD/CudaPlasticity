#pragma once

#include "ptrs.h"

//enum class ElemType { none, beam1, beam2, tria1, tria2, quad1, quad2, tetra1, tetra2, hexa1, hexa2 };
//
//enum class GeomType { none, D1, planeStress, planeStrain, planeGeneral, axiSym, D3, shell };

enum class ElemType { none, tria1, quad1, quad2 };

enum class GeomType { none, planeStress, planeStrain, planeGeneral, axiSym };


unsigned vtuElemType(ElemType elemType);


class FiniteElement {

public:

	unsigned blockId = 0;

	unsigned nodeCount = 0;

	unsigned memIdx = 0;

	unsigned poolIdx = 0;

	unsigned elemCount = 0;

	ElemType elemType = ElemType::none;

	GeomType geomType = GeomType::none;

	unsigned elemDim = 0;

	unsigned nodeDim = 2;

	// число точек интегрирования
	unsigned intPointsCount = 0;

	// указатель на массив точек интегрирования
	hptr<double> intPoints;

	hptr<double> intCoefs;

	// указатель на функции формы
	void (*calcN)(double* N, const double* xi) = nullptr;

	// указатель на вычисление градиентов и якобиана
	void (*calcB)(double* B, double& detJ, const double* x, const double* xi) = nullptr;

	// указатель на доп данные (толщина, коэффициенты и т.п.)
	hptr<double> data;

	FiniteElement() = default;

	FiniteElement(ElemType elemType, unsigned gaussPoints);

	~FiniteElement() = default;

	void setType(ElemType elemType, unsigned gaussPoints);

	void setIntPointsCount(unsigned gaussPoints);

	void print() const;

};


class CudaFiniteElement {

public:

	unsigned blockId = 0;

	unsigned nodeCount = 0;

	unsigned memIdx = 0;

	unsigned poolIdx = 0;

	unsigned elemCount = 0;

	ElemType elemType = ElemType::none;

	GeomType geomType = GeomType::none;

	unsigned elemDim = 0;

	unsigned nodeDim = 2;

	// число точек интегрирования
	unsigned intPointsCount = 0;

	// указатель на массив точек интегрирования
	dptr<double> intPoints;
	dptr<float> intPoints_f;

	dptr<double> intCoefs;
	dptr<float> intCoefs_f;

	// указатель на функции формы
	void (*calcN)(double* N, const double* xi) = nullptr;
	void (*calcN_f)(double* N, const double* xi) = nullptr;

	// указатель на вычисление градиентов и якобиана
	void (*calcB)(double* B, double& detJ, const double* x, const double* xi) = nullptr;
	void (*calcB_f)(double* B, double& detJ, const double* x, const double* xi) = nullptr;

	// указатель на доп данные (толщина, коэффициенты и т.п.)
	dptr<double> data;
	dptr<float> data_f; // мб и не dptr

	CudaFiniteElement();

	CudaFiniteElement(const FiniteElement& fe);

	void addFloat();

	void addFloat(const FiniteElement& fe);

	void print();

};