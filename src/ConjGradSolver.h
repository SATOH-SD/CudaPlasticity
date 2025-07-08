#pragma once

#include "SparseSLAE.h"


class ConjGradSolver {

private:

	SparseSLAE& slae;

	double* xNext = nullptr,
		* xPrev = nullptr,
		* rNext = nullptr,
		* rPrev = nullptr,
		* zNext = nullptr,
		* zPrev = nullptr,
		* Az = nullptr;

	double _normB = 0.;


public:

	ConjGradSolver(SparseSLAE& sparseSlae)
		: slae(sparseSlae) {
		xPrev = new double[slae.N],
		rNext = new double[slae.N],
		rPrev = new double[slae.N],
		zNext = new double[slae.N],
		zPrev = new double[slae.N],
		Az = new double[slae.N];
		_normB = 0.;
		for (int i = 0; i < slae.N; ++i) {
			/*double value = fabs(slae.rp[i]);
			if (value > _normB)
				_normB = value;*/
			_normB += slae.rp[i] * slae.rp[i];
		}
		_normB = 1. / _normB;
	}

	~ConjGradSolver() {
		//delete[] xPrev;  //TODO: FIX
		delete[] rNext;
		delete[] rPrev;
		delete[] zNext;
		delete[] zPrev;
		delete[] Az;
	}

	void solve(double* solution, bool* mask, size_t& iterNum, double eps) {
		xNext = solution;
		double max = 0., rNextScal = 0., rPrevScal = 0., Az_z;

#pragma omp parallel for reduction (+ : rNextScal)
		for (int i = 0; i < slae.N; ++i) {
			double sum = 0.;
			for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
				sum += slae.data[j] * solution[slae.cols[j]];
			double rk = mask[i] * (slae.rp[i] - sum);
			zNext[i] = rNext[i] = rk;
			rNextScal += rk * rk;
		}

		
		do {    //основной цикл
			//std::cout << "help\n";
			std::swap(xNext, xPrev);
			std::swap(rNext, rPrev);
			std::swap(zNext, zPrev);

			rPrevScal = rNextScal, Az_z = 0.,
			rNextScal = 0.;
#pragma omp parallel for reduction (+ : Az_z)
			for (int i = 0; i < slae.N; ++i) {
				double sum = 0.;
				for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
					sum += slae.data[j] * zPrev[slae.cols[j]];
				Az[i] = sum;
				Az_z += sum * zPrev[i];
			}
			double alpha = rPrevScal / Az_z;
#pragma omp parallel for reduction (+ : rNextScal)
			for (int i = 0; i < slae.N; ++i) {
				xNext[i] = xPrev[i] + alpha * zPrev[i];
				double rk = rPrev[i] - mask[i] * alpha * Az[i];
				rNext[i] = rk;
				rNextScal += rk * rk;
			}
			double beta = rNextScal / rPrevScal;

#pragma omp parallel for
			for (int i = 0; i < slae.N; ++i)
				zNext[i] = rNext[i] + beta * zPrev[i];

			//if (iterNum > 600) break;
			++iterNum;
		} while (rNextScal * _normB > eps * eps);
		solution = xNext;
	}

};