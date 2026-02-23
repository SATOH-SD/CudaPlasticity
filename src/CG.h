#pragma once

#include "ptrs.h"
#include "SparseSLAE.h"
#include "LoadConditions.h"


// Conjugate Gradient Method with Vectorization Support
class CGV {

private:

	SparseSLAE& slae;

	double* x = nullptr;
	bool* mask = nullptr;
	double* lines = nullptr;
	unsigned* lineRows = nullptr;
	unsigned lineCount = 0;
	int dim = 1;

	hptr<double> r, z, s;

	double normB = 1.;

	double rx1(double alpha) {
		double rho = 0.;
#pragma omp parallel for reduction (+ : rho)
		for (int i = 0; i < slae.N; ++i) {
			x[i] += alpha * z[i];
			double rk = r[i] -= mask[i] * alpha * s[i];
			rho += rk * rk;
		}
		return rho;
	}

	double rx1Lines(double alpha) {
#pragma omp parallel for
		for (int i = 0; i < slae.N; ++i) {
			x[i] += alpha * z[i];
			r[i] -= mask[i] * alpha * s[i];
		}
		for (int k = 0; k < lineCount; ++k) {  // учёт ограничений вдоль прямых
			unsigned i = lineRows[k];
			double len = 0.;
			for (int j = 0; j < dim; ++j)
				len += r[i + j] * lines[dim * k + j];
			for (int j = 0; j < dim; ++j)
				r[i + j] = lines[dim * k + j] * len;
		}
		double rho = 0.;
#pragma omp parallel for reduction (+ : rho)
		for (int i = 0; i < slae.N; ++i) {
			double rk = r[i];
			rho += rk * rk;
		}
		return rho;
	}

	double rx2(double alpha) {
		double rho = 0.;
#pragma omp parallel for reduction (+ : rho)
		for (int i = 0; i < slae.N; i += 2) {
			x[i] += alpha * z[i], \
			x[i + 1] += alpha * z[i + 1];
			double rk1 = r[i] -= mask[i] * alpha * s[i];
			double rk2 = r[i + 1] -= mask[i + 1] * alpha * s[i + 1];
			rho += rk1 * rk1 + rk2 * rk2;
		}
		return rho;
	}

	double rx2Lines(double alpha) {
#pragma omp parallel for
		for (int i = 0; i < slae.N; i += 2) {
			x[i] += alpha * z[i], \
			x[i + 1] += alpha * z[i + 1];
			r[i] -= mask[i] * alpha * s[i], \
			r[i + 1] -= mask[i + 1] * alpha * s[i + 1];
		}
		for (int k = 0; k < lineCount; ++k) {  // учёт ограничений вдоль прямых
			unsigned i = lineRows[k];
			double len = r[i] * lines[2 * k] + r[i + 1] * lines[2 * k + 1];
			r[i] = lines[2 * k] * len, \
			r[i + 1] = lines[2 * k + 1] * len;
		}
		double rho = 0.;
#pragma omp parallel for reduction (+ : rho)
		for (int i = 0; i < slae.N; i += 2) {
			double rk1 = r[i], rk2 = r[i + 1];
			rho += rk1 * rk1 + rk2 * rk2;
		}
		return rho;
	}

	double rx3(double alpha) {
		double rho = 0.;
#pragma omp parallel for reduction (+ : rho)
		for (int i = 0; i < slae.N; i += 3) {
			x[i] += alpha * z[i], \
			x[i + 1] += alpha * z[i + 1], \
			x[i + 2] += alpha * z[i + 2];
			double rk1 = r[i] -= mask[i] * alpha * s[i];
			double rk2 = r[i + 1] -= mask[i + 1] * alpha * s[i + 1];
			double rk3 = r[i + 2] -= mask[i + 2] * alpha * s[i + 2];
			rho += rk1 * rk1 + rk2 * rk2 + rk3 * rk3;
		}
		return rho;
	}

	double rx3Lines(double alpha) {
#pragma omp parallel for
		for (int i = 0; i < slae.N; i += 3) {
			x[i] += alpha * z[i], \
			x[i + 1] += alpha * z[i + 1], \
			x[i + 2] += alpha * z[i + 2];
			r[i] -= mask[i] * alpha * s[i], \
			r[i + 1] -= mask[i + 1] * alpha * s[i + 1], \
			r[i + 2] -= mask[i + 2] * alpha * s[i + 2];
		}
		for (int k = 0; k < lineCount; ++k) {  // учёт ограничений вдоль прямых
			unsigned i = lineRows[k];
			double len = r[i] * lines[3 * k] + r[i + 1] * lines[3 * k + 1] + r[i + 2] * lines[3 * k + 2];
			r[i] = lines[3 * k] * len, \
			r[i + 1] = lines[3 * k + 1] * len, \
			r[i + 2] = lines[3 * k + 2] * len;
		}
		double rho = 0.;
#pragma omp parallel for reduction (+ : rho)
		for (int i = 0; i < slae.N; i += 3) {
			double rk1 = r[i], rk2 = r[i + 1], rk3 = r[i + 2];
			rho += rk1 * rk1 + rk2 * rk2 + rk3 * rk3;
		}
		return rho;
	}

	// Solve without Vectorization
	unsigned solve1(double eps) {
		double rhoNext = 0., rhoPrev = 1.;
		unsigned iterNum = 0;

#pragma omp parallel for
		for (int i = 0; i < slae.N; ++i) {
			double sum = 0.;
			for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
				sum += slae.data[j] * x[slae.cols[j]];
			r[i] = mask[i] * (slae.rp[i] - sum);
			z[i] = 0.;
		}
		for (int k = 0; k < lineCount; ++k) {  // учёт ограничений вдоль прямых
			unsigned i = lineRows[k];
			double len = 0.;
			for (int j = 0; j < dim; ++j)
				len += r[i + j] * lines[dim * k + j];
			for (int j = 0; j < dim; ++j)
				r[i + j] = lines[dim * k + j] * len;
		}
#pragma omp parallel for reduction (+ : rhoNext)
		for (int i = 0; i < slae.N; ++i) {
			double rk = r[i];
			rhoNext += rk * rk;
		}

		double eps2b = eps * eps * normB;

		while (rhoNext > eps2b) {    //основной цикл

			double beta = rhoNext / rhoPrev;

#pragma omp parallel for
			for (int i = 0; i < slae.N; ++i)
				z[i] = r[i] + beta * z[i];

			double omega = 0.;
#pragma omp parallel for reduction (+ : omega)
			for (int i = 0; i < slae.N; ++i) {
				double sum = 0.;
				for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
					sum += slae.data[j] * z[slae.cols[j]];
				s[i] = sum;
				omega += sum * z[i];
			}

			double alpha = rhoNext / omega;
			rhoPrev = rhoNext;

			rhoNext = (this->*rxPtr)(alpha);

			if (iterNum > slae.N) break;
			++iterNum;
		}
		return iterNum;
	}

	// Solve with Vectorization 2x2
	unsigned solve2(double eps) {
		double rhoNext = 0., rhoPrev = 1.;
		unsigned iterNum = 0;

#pragma omp parallel for
		for (int i = 0; i < slae.N; i += 2) {
			register double sum1 = 0., sum2 = 0.;
			int shift = slae.rows[i + 1] - slae.rows[i];
			for (int j = slae.rows[i]; j < slae.rows[i + 1]; j += 2) {
				int col = slae.cols[j];
				register double xe1 = x[col], xe2 = x[col + 1];
				sum1 += slae.data[j] * xe1 + slae.data[j + 1] * xe2;
				sum2 += slae.data[j + shift] * xe1 + slae.data[j + shift + 1] * xe2;
			}
			r[i] = mask[i] * (slae.rp[i] - sum1);
			r[i + 1] = mask[i + 1] * (slae.rp[i + 1] - sum2);
			z[i + 1] = z[i] = 0.;
		}
		for (int k = 0; k < lineCount; ++k) {  // учёт ограничений вдоль прямых
			unsigned i = lineRows[k];
			double len = r[i] * lines[2 * k] + r[i + 1] * lines[2 * k + 1];
			r[i] = lines[2 * k] * len, \
			r[i + 1] = lines[2 * k + 1] * len;
		}
#pragma omp parallel for reduction (+ : rhoNext)
		for (int i = 0; i < slae.N; i += 2) {
			double rk1 = r[i], rk2 = r[i + 1];
			rhoNext += rk1 * rk1 + rk2 * rk2;
		}

		double eps2b = eps * eps * normB;

		while (rhoNext > eps2b) {    //основной цикл

			double beta = rhoNext / rhoPrev;

#pragma omp parallel for
			for (int i = 0; i < slae.N; i += 2) {
				z[i] = r[i] + beta * z[i], \
				z[i + 1] = r[i + 1] + beta * z[i + 1];
			}

			double omega = 0.;
#pragma omp parallel for reduction (+ : omega)
			for (int i = 0; i < slae.N; i += 2) {
				register double sum1 = 0., sum2 = 0.;
				int shift = slae.rows[i + 1] - slae.rows[i];
				for (int j = slae.rows[i]; j < slae.rows[i + 1]; j += 2) {
					int col = slae.cols[j];
					register double ze1 = z[col], ze2 = z[col + 1];
					sum1 += slae.data[j] * ze1 + slae.data[j + 1] * ze2;
					sum2 += slae.data[j + shift] * ze1 + slae.data[j + shift + 1] * ze2;
				}
				s[i] = sum1, s[i + 1] = sum2;
				omega += sum1 * z[i] + sum2 * z[i + 1];
			}

			double alpha = rhoNext / omega;
			rhoPrev = rhoNext;

			rhoNext = (this->*rxPtr)(alpha);

			if (iterNum > slae.N) break;
			++iterNum;
		}
		return iterNum;
	}

	// Solve with Vectorization 3x3
	unsigned solve3(double eps) {
		double rhoNext = 0., rhoPrev = 1.;
		unsigned iterNum = 0;

#pragma omp parallel for
		for (int i = 0; i < slae.N; i += 3) {
			register double sum1 = 0., sum2 = 0., sum3 = 0.;
			int shift = slae.rows[i + 1] - slae.rows[i], shift2 = shift * 2;
			for (int j = slae.rows[i]; j < slae.rows[i + 1]; j += 3) {
				int col = slae.cols[j];
				register double xe1 = x[col], xe2 = x[col + 1], xe3 = x[col + 2];
				sum1 += slae.data[j] * xe1 + slae.data[j + 1] * xe2 + slae.data[j + 3] * xe3;
				sum2 += slae.data[j + shift] * xe1 + slae.data[j + shift + 1] * xe2 + slae.data[j + shift + 2] * xe3;
				sum3 += slae.data[j + shift2] * xe1 + slae.data[j + shift2 + 1] * xe2 + slae.data[j + shift2 + 2] * xe3;
			}
			r[i] = mask[i] * (slae.rp[i] - sum1);
			r[i + 1] = mask[i + 1] * (slae.rp[i + 1] - sum2);
			r[i + 2] = mask[i + 2] * (slae.rp[i + 2] - sum3);
			z[i + 2] = z[i + 1] = z[i] = 0.;
		}
		for (int k = 0; k < lineCount; ++k) {  // учёт ограничений вдоль прямых
			unsigned i = lineRows[k];
			double len = r[i] * lines[3 * k] + r[i + 1] * lines[3 * k + 1] + r[i + 2] * lines[3 * k + 2];
			r[i] = lines[3 * k] * len, \
			r[i + 1] = lines[3 * k + 1] * len, \
			r[i + 2] = lines[3 * k + 2] * len;
		}
#pragma omp parallel for reduction (+ : rhoNext)
		for (int i = 0; i < slae.N; i += 3) {
			double rk1 = r[i], rk2 = r[i + 1], rk3 = r[i + 2];
			rhoNext += rk1 * rk1 + rk2 * rk2 + rk3 * rk3;
		}

		double eps2b = eps * eps * normB;

		while (rhoNext > eps2b) {    //основной цикл

			double beta = rhoNext / rhoPrev;

#pragma omp parallel for
			for (int i = 0; i < slae.N; i += 3) {
				z[i] = r[i] + beta * z[i], \
				z[i + 1] = r[i + 1] + beta * z[i + 1], \
				z[i + 2] = r[i + 2] + beta * z[i + 2];
			}

			double omega = 0.;
#pragma omp parallel for reduction (+ : omega)
			for (int i = 0; i < slae.N; i += 3) {
				register double sum1 = 0., sum2 = 0., sum3 = 0.;
				int shift = slae.rows[i + 1] - slae.rows[i], shift2 = shift * 2;
				for (int j = slae.rows[i]; j < slae.rows[i + 1]; j += 3) {
					int col = slae.cols[j];
					register double ze1 = z[col], ze2 = z[col + 1], ze3 = z[col + 2];
					sum1 += slae.data[j] * ze1 + slae.data[j + 1] * ze2 + slae.data[j + 3] * ze3;
					sum2 += slae.data[j + shift] * ze1 + slae.data[j + shift + 1] * ze2 + slae.data[j + shift + 2] * ze3;
					sum3 += slae.data[j + shift2] * ze1 + slae.data[j + shift2 + 1] * ze2 + slae.data[j + shift2 + 2] * ze3;
				}
				s[i] = sum1, s[i + 1] = sum2, s[i + 2] = sum3;
				omega += sum1 * z[i] + sum2 * z[i + 1] + sum3 * z[i + 2];
			}

			double alpha = rhoNext / omega;
			rhoPrev = rhoNext;

			rhoNext = (this->*rxPtr)(alpha);

			if (iterNum > slae.N) break;
			++iterNum;
		}
		return iterNum;
	}

	unsigned (CGV::*solveMethod)(double);
	double (CGV::*rxPtr)(double);


public:

	CGV(SparseSLAE& slae, double* solVector, bool* mask, unsigned dim = 1, \
		double* lines = nullptr, unsigned* lineRows = nullptr, unsigned lineCount = 0)
		: slae(slae), x(solVector), mask(mask), dim(dim), \
		lines(lines), lineRows(lineRows), lineCount(lineCount) {

		r.malloc(slae.N);
		z.malloc(slae.N);
		s.malloc(slae.N);

		switch (dim) {
		case 2:
			solveMethod = &CGV::solve2;
			rxPtr = lineCount ? &CGV::rx2Lines : &CGV::rx2;
			break;
		case 3:
			solveMethod = &CGV::solve3;
			rxPtr = lineCount ? &CGV::rx3Lines : &CGV::rx3;
			break;
		default:
			solveMethod = &CGV::solve1;
			rxPtr = lineCount ? &CGV::rx1Lines : &CGV::rx1;
		}
	}

	~CGV() = default;

	void updateNorm() {
		double sum = 0.;
#pragma omp parallel for reduction(+ : sum)
		for (int i = 0; i < slae.N; ++i)
			sum += slae.rp[i] * slae.rp[i];
		if (sum < 1e-200)
#pragma omp parallel for reduction(+ : sum)
			for (int i = 0; i < slae.N; ++i)
				sum += x[i] * x[i];
		normB = sum;
	}

	unsigned solve(double eps) {
		return (this->*solveMethod)(eps);
	}

	void setLinesDim(unsigned dim) {
		if (CGV::dim == 1) CGV::dim = dim;
	}


};


// Jacobi Preconditioned Conjugate Gradient Method with Vectorization Support
class JCGV {

private:

	SparseSLAE& slae;

	double* x = nullptr;
	bool* mask = nullptr;
	double* lines = nullptr;
	unsigned* lineRows = nullptr;
	unsigned lineCount = 0;
	int dim = 1;

	hptr<double> r, z, s, q, DR;

	double normB = 1.;

	double rx1(double alpha) {
		double rho = 0.;
#pragma omp parallel for reduction (+ : rho)
		for (int i = 0; i < slae.N; ++i) {
			x[i] += alpha * z[i];
			double rk = r[i] -= mask[i] * alpha * s[i];
			double qk = q[i] = DR[i] * rk;
			rho += qk * rk;
		}
		return rho;
	}

	double rx1Lines(double alpha) {
#pragma omp parallel for
		for (int i = 0; i < slae.N; ++i) {
			x[i] += alpha * z[i];
			double rk = r[i] -= mask[i] * alpha * s[i];
			q[i] = DR[i] * rk;
		}
		for (int k = 0; k < lineCount; ++k) {  // учёт ограничений вдоль прямых
			unsigned i = lineRows[k];
			double len = 0.;
			for (int j = 0; j < dim; ++j)
				len += r[i + j] * lines[dim * k + j];
			for (int j = 0; j < dim; ++j)
				r[i + j] = lines[dim * k + j] * len;
			len = 0.;
			for (int j = 0; j < dim; ++j)
				len += q[i + j] * lines[dim * k + j];
			for (int j = 0; j < dim; ++j)
				q[i + j] = lines[dim * k + j] * len;
		}
		double rho = 0.;
#pragma omp parallel for reduction (+ : rho)
		for (int i = 0; i < slae.N; ++i)
			rho += q[i] * r[i];
		return rho;
	}

	double rx2(double alpha) {
		double rho = 0.;
#pragma omp parallel for reduction (+ : rho)
		for (int i = 0; i < slae.N; i += 2) {
			x[i] += alpha * z[i], \
			x[i + 1] += alpha * z[i + 1];
			double rk1 = r[i] -= mask[i] * alpha * s[i];
			double rk2 = r[i + 1] -= mask[i + 1] * alpha * s[i + 1];
			double qk1 = q[i] = DR[i] * rk1, \
				qk2 = q[i + 1] = DR[i + 1] * rk2;
			rho += qk1 * rk1 + qk2 * rk2;
		}
		return rho;
	}

	double rx2Lines(double alpha) {
#pragma omp parallel for
		for (int i = 0; i < slae.N; i += 2) {
			x[i] += alpha * z[i], \
			x[i + 1] += alpha * z[i + 1];
			double rk1 = r[i] -= mask[i] * alpha * s[i];
			double rk2 = r[i + 1] -= mask[i + 1] * alpha * s[i + 1];
			q[i] = DR[i] * rk1, \
			q[i + 1] = DR[i + 1] * rk2;
		}
		for (int k = 0; k < lineCount; ++k) {  // учёт ограничений вдоль прямых
			unsigned i = lineRows[k];
			double len = r[i] * lines[2 * k] + r[i + 1] * lines[2 * k + 1];
			r[i] = lines[2 * k] * len, \
			r[i + 1] = lines[2 * k + 1] * len;
			len = q[i] * lines[2 * k] + q[i + 1] * lines[2 * k + 1];
			q[i] = lines[2 * k] * len, \
			q[i + 1] = lines[2 * k + 1] * len;
		}
		double rho = 0.;
#pragma omp parallel for reduction (+ : rho)
		for (int i = 0; i < slae.N; i += 2)
			rho += q[i] * r[i] + q[i + 1] * r[i + 1];
		return rho;
	}

	double rx3(double alpha) {
		double rho = 0.;
#pragma omp parallel for reduction (+ : rho)
		for (int i = 0; i < slae.N; i += 3) {
			x[i] += alpha * z[i], \
			x[i + 1] += alpha * z[i + 1], \
			x[i + 2] += alpha * z[i + 2];
			double rk1 = r[i] -= mask[i] * alpha * s[i];
			double rk2 = r[i + 1] -= mask[i + 1] * alpha * s[i + 1];
			double rk3 = r[i + 2] -= mask[i + 2] * alpha * s[i + 2];
			double qk1 = q[i] = DR[i] * rk1, \
				qk2 = q[i + 1] = DR[i + 1] * rk2, \
				qk3 = q[i + 2] = DR[i + 2] * rk3;
			rho += qk1 * rk1 + qk2 * rk2 + qk3 * rk3;
		}
		return rho;
	}

	double rx3Lines(double alpha) {
#pragma omp parallel for
		for (int i = 0; i < slae.N; i += 3) {
			x[i] += alpha * z[i], \
			x[i + 1] += alpha * z[i + 1], \
			x[i + 2] += alpha * z[i + 2];
			double rk1 = r[i] -= mask[i] * alpha * s[i];
			double rk2 = r[i + 1] -= mask[i + 1] * alpha * s[i + 1];
			double rk3 = r[i + 2] -= mask[i + 2] * alpha * s[i + 2];
			q[i] = DR[i] * rk1, \
			q[i + 1] = DR[i + 1] * rk2, \
			q[i + 2] = DR[i + 2] * rk3;
		}
		for (int k = 0; k < lineCount; ++k) {  // учёт ограничений вдоль прямых
			unsigned i = lineRows[k];
			double len = r[i] * lines[3 * k] + r[i + 1] * lines[3 * k + 1] + r[i + 2] * lines[3 * k + 2];
			r[i] = lines[3 * k] * len, \
			r[i + 1] = lines[3 * k + 1] * len, \
			r[i + 2] = lines[3 * k + 2] * len;
			len = q[i] * lines[3 * k] + q[i + 1] * lines[3 * k + 1] + q[i + 2] * lines[3 * k + 2];
			q[i] = lines[3 * k] * len, \
			q[i + 1] = lines[3 * k + 1] * len, \
			q[i + 2] = lines[3 * k + 2] * len;
		}
		double rho = 0.;
#pragma omp parallel for reduction (+ : rho)
		for (int i = 0; i < slae.N; i += 3)
			rho += q[i] * r[i] + q[i + 1] * r[i + 1] + q[i + 2] * r[i + 2];
		return rho;
	}

	// Solve without Vectorization
	unsigned solve1(double eps) {
		double rhoNext = 0., rhoPrev = 1.;
		unsigned iterNum = 0;

#pragma omp parallel for
		for (int i = 0; i < slae.N; ++i) {
			double sum = 0.;
			for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
				sum += slae.data[j] * x[slae.cols[j]];
			double rk = r[i] = mask[i] * (slae.rp[i] - sum);
			q[i] = DR[i] * rk;
			z[i] = 0.;
		}
		for (int k = 0; k < lineCount; ++k) {  // учёт ограничений вдоль прямых
			unsigned i = lineRows[k];
			double len = 0.;
			for (int j = 0; j < dim; ++j)
				len += r[i + j] * lines[dim * k + j];
			for (int j = 0; j < dim; ++j)
				r[i + j] = lines[dim * k + j] * len;
			len = 0.;
			for (int j = 0; j < dim; ++j)
				len += q[i + j] * lines[dim * k + j];
			for (int j = 0; j < dim; ++j)
				q[i + j] = lines[dim * k + j] * len;
		}
#pragma omp parallel for reduction (+ : rhoNext)
		for (int i = 0; i < slae.N; ++i)
			rhoNext += q[i] * r[i];

		double eps2b = eps * eps * normB;

		while (rhoNext > eps2b) {    //основной цикл

			double beta = rhoNext / rhoPrev;

#pragma omp parallel for
			for (int i = 0; i < slae.N; ++i)
				z[i] = q[i] + beta * z[i];

			double omega = 0.;
#pragma omp parallel for reduction (+ : omega)
			for (int i = 0; i < slae.N; ++i) {
				double sum = 0.;
				for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
					sum += slae.data[j] * z[slae.cols[j]];
				s[i] = sum;
				omega += sum * z[i];
			}

			double alpha = rhoNext / omega;
			rhoPrev = rhoNext;

			rhoNext = (this->*rxPtr)(alpha);

			if (iterNum > slae.N) break;
			++iterNum;
		}
		return iterNum;
	}

	// Solve with Vectorization 2x2
	unsigned solve2(double eps) {
		double rhoNext = 0., rhoPrev = 1.;
		unsigned iterNum = 0;

#pragma omp parallel for
		for (int i = 0; i < slae.N; i += 2) {
			register double sum1 = 0., sum2 = 0.;
			int shift = slae.rows[i + 1] - slae.rows[i];
			for (int j = slae.rows[i]; j < slae.rows[i + 1]; j += 2) {
				int col = slae.cols[j];
				register double xe1 = x[col], xe2 = x[col + 1];
				sum1 += slae.data[j] * xe1 + slae.data[j + 1] * xe2;
				sum2 += slae.data[j + shift] * xe1 + slae.data[j + shift + 1] * xe2;
			}
			double rk1 = r[i] = mask[i] * (slae.rp[i] - sum1);
			double rk2 = r[i + 1] = mask[i + 1] * (slae.rp[i + 1] - sum2);
			q[i] = DR[i] * rk1, \
			q[i + 1] = DR[i + 1] * rk2;
			z[i + 1] = z[i] = 0.;
		}
		for (int k = 0; k < lineCount; ++k) {  // учёт ограничений вдоль прямых
			unsigned i = lineRows[k];
			double len = r[i] * lines[2 * k] + r[i + 1] * lines[2 * k + 1];
			r[i] = lines[2 * k] * len, \
			r[i + 1] = lines[2 * k + 1] * len;
			len = q[i] * lines[2 * k] + q[i + 1] * lines[2 * k + 1];
			q[i] = lines[2 * k] * len, \
			q[i + 1] = lines[2 * k + 1] * len;
		}
#pragma omp parallel for reduction (+ : rhoNext)
		for (int i = 0; i < slae.N; i += 2)
			rhoNext += q[i] * r[i] + q[i + 1] * r[i + 1];

		double eps2b = eps * eps * normB;

		while (rhoNext > eps2b) {    //основной цикл

			double beta = rhoNext / rhoPrev;

#pragma omp parallel for
			for (int i = 0; i < slae.N; i += 2) {
				z[i] = q[i] + beta * z[i], \
				z[i + 1] = q[i + 1] + beta * z[i + 1];
			}

			double omega = 0.;
#pragma omp parallel for reduction (+ : omega)
			for (int i = 0; i < slae.N; i += 2) {
				register double sum1 = 0., sum2 = 0.;
				int shift = slae.rows[i + 1] - slae.rows[i];
				for (int j = slae.rows[i]; j < slae.rows[i + 1]; j += 2) {
					int col = slae.cols[j];
					register double ze1 = z[col], ze2 = z[col + 1];
					sum1 += slae.data[j] * ze1 + slae.data[j + 1] * ze2;
					sum2 += slae.data[j + shift] * ze1 + slae.data[j + shift + 1] * ze2;
				}
				s[i] = sum1, s[i + 1] = sum2;
				omega += sum1 * z[i] + sum2 * z[i + 1];
			}

			double alpha = rhoNext / omega;
			rhoPrev = rhoNext;

			rhoNext = (this->*rxPtr)(alpha);

			if (iterNum > slae.N) break;
			++iterNum;
		}
		return iterNum;
	}

	// Solve with Vectorization 3x3
	unsigned solve3(double eps) {
		double rhoNext = 0., rhoPrev = 1.;
		unsigned iterNum = 0;

#pragma omp parallel for
		for (int i = 0; i < slae.N; i += 3) {
			register double sum1 = 0., sum2 = 0., sum3 = 0.;
			int shift = slae.rows[i + 1] - slae.rows[i], shift2 = shift * 2;
			for (int j = slae.rows[i]; j < slae.rows[i + 1]; j += 3) {
				int col = slae.cols[j];
				register double xe1 = x[col], xe2 = x[col + 1], xe3 = x[col + 2];
				sum1 += slae.data[j] * xe1 + slae.data[j + 1] * xe2 + slae.data[j + 3] * xe3;
				sum2 += slae.data[j + shift] * xe1 + slae.data[j + shift + 1] * xe2 + slae.data[j + shift + 2] * xe3;
				sum3 += slae.data[j + shift2] * xe1 + slae.data[j + shift2 + 1] * xe2 + slae.data[j + shift2 + 2] * xe3;
			}
			double rk1 = r[i] = mask[i] * (slae.rp[i] - sum1);
			double rk2 = r[i + 1] = mask[i + 1] * (slae.rp[i + 1] - sum2);
			double rk3 = r[i + 2] = mask[i + 2] * (slae.rp[i + 2] - sum3);
			q[i] = DR[i] * rk1, \
			q[i + 1] = DR[i + 1] * rk2, \
			q[i + 2] = DR[i + 2] * rk3;
			z[i + 2] = z[i + 1] = z[i] = 0.;
		}
		for (int k = 0; k < lineCount; ++k) {  // учёт ограничений вдоль прямых
			unsigned i = lineRows[k];
			double len = r[i] * lines[3 * k] + r[i + 1] * lines[3 * k + 1] + r[i + 2] * lines[3 * k + 2];
			r[i] = lines[3 * k] * len, \
			r[i + 1] = lines[3 * k + 1] * len, \
			r[i + 2] = lines[3 * k + 2] * len;
			len = q[i] * lines[3 * k] + q[i + 1] * lines[3 * k + 1] + q[i + 2] * lines[3 * k + 2];
			q[i] = lines[3 * k] * len, \
			q[i + 1] = lines[3 * k + 1] * len, \
			q[i + 2] = lines[3 * k + 2] * len;
		}
#pragma omp parallel for reduction (+ : rhoNext)
		for (int i = 0; i < slae.N; i += 3)
			rhoNext += q[i] * r[i] + q[i + 1] * r[i + 1] + q[i + 2] * r[i + 2];

		double eps2b = eps * eps * normB;

		while (rhoNext > eps2b) {    //основной цикл

			double beta = rhoNext / rhoPrev;

#pragma omp parallel for
			for (int i = 0; i < slae.N; i += 3) {
				z[i] = q[i] + beta * z[i], \
				z[i + 1] = q[i + 1] + beta * z[i + 1], \
				z[i + 2] = q[i + 2] + beta * z[i + 2];
			}

			double omega = 0.;
#pragma omp parallel for reduction (+ : omega)
			for (int i = 0; i < slae.N; i += 3) {
				register double sum1 = 0., sum2 = 0., sum3 = 0.;
				int shift = slae.rows[i + 1] - slae.rows[i], shift2 = shift * 2;
				for (int j = slae.rows[i]; j < slae.rows[i + 1]; j += 3) {
					int col = slae.cols[j];
					register double ze1 = z[col], ze2 = z[col + 1], ze3 = z[col + 2];
					sum1 += slae.data[j] * ze1 + slae.data[j + 1] * ze2 + slae.data[j + 3] * ze3;
					sum2 += slae.data[j + shift] * ze1 + slae.data[j + shift + 1] * ze2 + slae.data[j + shift + 2] * ze3;
					sum3 += slae.data[j + shift2] * ze1 + slae.data[j + shift2 + 1] * ze2 + slae.data[j + shift2 + 2] * ze3;
				}
				s[i] = sum1, s[i + 1] = sum2, s[i + 2] = sum3;
				omega += sum1 * z[i] + sum2 * z[i + 1] + sum3 * z[i + 2];
			}

			double alpha = rhoNext / omega;
			rhoPrev = rhoNext;

			rhoNext = (this->*rxPtr)(alpha);

			if (iterNum > slae.N) break;
			++iterNum;
		}
		return iterNum;
	}

	unsigned (JCGV::* solveMethod)(double);
	double (JCGV::* rxPtr)(double);

public:

	JCGV(SparseSLAE& slae, double* solVector, bool* mask, unsigned dim = 1, \
		double* lines = nullptr, unsigned* lineRows = nullptr, unsigned lineCount = 0)
		: slae(slae), x(solVector), mask(mask), dim(dim), \
		lines(lines), lineRows(lineRows), lineCount(lineCount) {

		r.malloc(slae.N);
		z.malloc(slae.N);
		s.malloc(slae.N);
		q.malloc(slae.N);
		DR.malloc(slae.N);

		switch (dim) {
		case 2:
			solveMethod = &JCGV::solve2;
			rxPtr = lineCount ? &JCGV::rx2Lines : &JCGV::rx2;
			break;
		case 3:
			solveMethod = &JCGV::solve3;
			rxPtr = lineCount ? &JCGV::rx3Lines : &JCGV::rx3;
			break;
		default:
			solveMethod = &JCGV::solve1;
			rxPtr = lineCount ? &JCGV::rx1Lines : &JCGV::rx1;
		}
	}

	~JCGV() = default;

	void updateNorm() {
		double sum = 0.;
#pragma omp parallel for reduction(+ : sum)
		for (int i = 0; i < slae.N; ++i)
			sum += DR[i] * slae.rp[i] * slae.rp[i];
		if (sum < 1e-200)
#pragma omp parallel for reduction(+ : sum)
			for (int i = 0; i < slae.N; ++i)
				sum += DR[i] * x[i] * x[i];
		normB = sum;
	}

	void precond() {
#pragma omp parallel for
		for (int i = 0; i < slae.N; ++i) {
			double sum = 0.;
			for (unsigned j = slae.rows[i]; j < slae.rows[i + 1]; ++j) {
				/*if (slae.cols[j] == i) {
					sum = slae.data[j];
					break;
				}*/
				double el = slae.data[j];
				sum += el * el;
			}
			DR[i] = 1. / sqrt(sum);
		}
		updateNorm();
	}

	unsigned solve(double eps) {
		return (this->*solveMethod)(eps);
	}

	void setLinesDim(unsigned dim) {
		if (JCGV::dim == 1) JCGV::dim = dim;
	}

};