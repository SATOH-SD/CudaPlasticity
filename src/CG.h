#pragma once

#include "ptrs.h"
#include "SparseSLAE.h"
#include "LoadConditions.h"


// (Jacobi Preconditioned) Conjugate Gradient Method with Vectorization Support
class JCGV {

private:

	SparseSLAE& slae;      // Ссылка на систему уравнений

	double* x = nullptr;           // Указатель на вектор решения
	bool* mask = nullptr;          // Указатель на маску кинематических условий
	double* lines = nullptr;       // Массив направлений для закреплений вдоль линий
	unsigned* lineRows = nullptr;  // Индексы первых строк системы, соответствующих закреплениям на линиях
	unsigned lineCount = 0;        // Количество закреплённых на линиях узлов
	int dim = 1;                   // Размерность узловых величин

	hptr<double>
		r,   // Вектор невязки
		z,   // Вектор поправки
		s,   // Вектор поправки, умноженный слева на матрицу системы
		q,   // Вектор невязки, умноженный слева на обратную матрицу предобуславливания
		DR;  // Диагональ обратной матрицы предобуславливания

	double normB = 1.;  // Нормировка критерия останова

public:

	JCGV() = delete;

	/*
	* Конструктор
	* @param slae             Ссылка на систему уравнений
	* @param solVector        Указатель на вектор решения
	* @param mask             Маска кинематических условий
	* @param preconditioning  Включение предобуславливания
	* @param updatingMatrix   Обновляется ли матрица системы между вызовами solve()
	* @param updatingRhs      Обновляется ли вектор правой части между вызовами solve()
	* @param dim              Размерность узловых величин
	* @param lines            Массив направлений для закреплений вдоль линий
	* @param lineRows         Индексы первых строк системы, соответствующих закреплениям на линиях
	* @param lineCount        Количество закреплённых на линиях узлов
	*/
	JCGV(SparseSLAE& slae,
		double* solVector,
		bool* mask,
		bool preconditioning = true,
		bool updatingMatrix = false,   
		bool updatingRhs = false,
		unsigned dim = 1,
		double* lines = nullptr,
		unsigned* lineRows = nullptr,
		unsigned lineCount = 0
	);

	~JCGV() = default;

	// Решить систему
	unsigned solve(double eps) {
		(this->*preSolve)();
		return (this->*solveMethod)(eps);
	}

	// Задать размерность направлений для закреплений вдоль линий
	void setLinesDim(unsigned dim) {
		if (JCGV::dim == 1) JCGV::dim = dim;
	}

private:

	double rx1(double alpha);
	double rxP1(double alpha);

	double rx1Lines(double alpha);
	double rxP1Lines(double alpha);

	double rx2(double alpha);
	double rxP2(double alpha);

	double rx2Lines(double alpha);
	double rxP2Lines(double alpha);

	double rx3(double alpha);
	double rxP3(double alpha);

	double rx3Lines(double alpha);
	double rxP3Lines(double alpha);

	// Solve without Vectorization
	unsigned solve1(double eps);
	unsigned solveP1(double eps);

	// Solve with Vectorization 2x2
	unsigned solve2(double eps);
	unsigned solveP2(double eps);

	// Solve with Vectorization 3x3
	unsigned solve3(double eps);
	unsigned solveP3(double eps);

	void emptyPreSolve() {};

	unsigned (JCGV::* solveMethod)(double);
	double (JCGV::* rxPtr)(double);
	void (JCGV::* preSolve)(void) = &JCGV::emptyPreSolve;

	void updateNorm();

	void updateNormP();

	void precond();

	void precondAndNorm();

};