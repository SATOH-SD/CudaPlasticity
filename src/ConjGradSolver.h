// LEGACY

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

			if (iterNum > slae.N) break;
			++iterNum;
		} while (rNextScal * _normB > eps * eps);
		solution = xNext;
	}

};



class ConjGradSolver3 {

private:

	SparseSLAE& slae;

	double* xNext = nullptr,
		* xPrev = nullptr,
		* rNext = nullptr,
		* rPrev = nullptr,
		* zNext = nullptr,
		* zPrev = nullptr,
		* s = nullptr;

	unsigned* displs = nullptr;

	double _normB = 0.;


public:

	ConjGradSolver3(SparseSLAE& sparseSlae, double* solVector)
		: slae(sparseSlae) {
		xPrev = new double[slae.N],
			rNext = new double[slae.N],
			rPrev = new double[slae.N],
			zNext = new double[slae.N],
			zPrev = new double[slae.N],
			s = new double[slae.N];
		_normB = 0.;
		for (int i = 0; i < slae.N; ++i) {
			/*double value = fabs(slae.rp[i]);
			if (value > _normB)
				_normB = value;*/
			_normB += slae.rp[i] * slae.rp[i];
		}
		if (_normB < 1e-200)
			for (int i = 0; i < slae.N; ++i)
				_normB += solVector[i] * solVector[i];
		_normB = 1. / _normB;

		unsigned threads = omp_get_max_threads();
		displs = new unsigned[threads + 1];
		displs[0] = 0;
		for (unsigned i = 0; i < threads; ++i) {
			unsigned size = slae.N / threads + (i < slae.N % threads);
			displs[i + 1] = displs[i] + size;
		}
	}

	~ConjGradSolver3() {
		//delete[] xPrev;  //TODO: FIX
		delete[] rNext;
		delete[] rPrev;
		delete[] zNext;
		delete[] zPrev;
		delete[] s;
		delete[] displs;
	}

	void solve(double* solution, bool* mask, size_t& iterNum, double eps) {
		xNext = solution;
		double max = 0., rhoNext = 0., rhoPrev = 0., omega;

#pragma omp parallel for reduction (+ : rhoNext)
		for (int i = 0; i < slae.N; ++i) {
			double sum = 0.;
			for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
				sum += slae.data[j] * solution[slae.cols[j]];
			double rk = mask[i] * (slae.rp[i] - sum);
			zNext[i] = rNext[i] = rk;
			rhoNext += rk * rk;
		}


		do {    //основной цикл
			//std::cout << "help\n";
			std::swap(xNext, xPrev);
			std::swap(rNext, rPrev);
			std::swap(zNext, zPrev);

			rhoPrev = rhoNext, omega = 0.,
				rhoNext = 0.;
#pragma omp parallel for reduction (+ : omega)
			for (int i = 0; i < slae.N; ++i) {
				double sum = 0.;
				for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
					sum += slae.data[j] * zPrev[slae.cols[j]];
				s[i] = sum;
				omega += sum * zPrev[i];
			}
			double alpha = rhoPrev / omega;
#pragma omp parallel for reduction (+ : rhoNext)
			for (int i = 0; i < slae.N; ++i) {
				xNext[i] = xPrev[i] + alpha * zPrev[i];
				double rk = rPrev[i] - mask[i] * alpha * s[i];
				rNext[i] = rk;
				rhoNext += rk * rk;
			}
			double beta = rhoNext / rhoPrev;

#pragma omp parallel for
			for (int i = 0; i < slae.N; ++i)
				zNext[i] = rNext[i] + beta * zPrev[i];

			if (iterNum > slae.N) break;
			++iterNum;
		} while (rhoNext * _normB > eps * eps);
		solution = xNext;
	}


	void solve2(double* solution, bool* mask, size_t& iterNum, double eps) {
		xNext = solution;
		double max = 0., rhoNext = 0., rhoPrev = 1., omega;

#pragma omp parallel for reduction (+ : rhoNext)
		for (int i = 0; i < slae.N; ++i) {
			double sum = 0.;
			for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
				sum += slae.data[j] * solution[slae.cols[j]];
			double rk = mask[i] * (slae.rp[i] - sum);
			rNext[i] = rk;
			zNext[i] = 0.;
			rhoNext += rk * rk;
		}

		//std::cout << rhoPrev << " " << rhoNext << "\n";

		while (rhoNext * _normB > eps * eps) {    //основной цикл
			//std::cout << "help\n";
			//std::swap(xNext, xPrev);
			//std::swap(rNext, rPrev);
			//std::swap(zNext, zPrev);

			double beta = rhoNext / rhoPrev;

#pragma omp parallel for
			for (int i = 0; i < slae.N; ++i)
				//zNext[i] = rPrev[i] + beta * zPrev[i];
				zNext[i] = rNext[i] + beta * zNext[i];
			
			omega = 0.;
#pragma omp parallel for reduction (+ : omega)
			for (int i = 0; i < slae.N; ++i) {
				double sum = 0.;
				for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
					sum += slae.data[j] * zNext[slae.cols[j]];
				s[i] = sum;
				omega += sum * zNext[i];
			}

			double alpha = rhoNext / omega;
			rhoPrev = rhoNext, rhoNext = 0.;

			//std::cout << "omega " << omega << "\n";
			//std::cout << "alpha " << alpha << "\n";

#pragma omp parallel for reduction (+ : rhoNext)
			for (int i = 0; i < slae.N; ++i) {
				//xNext[i] = xPrev[i] + alpha * zNext[i];
				xNext[i] += alpha * zNext[i];
				double rk = rNext[i] -= mask[i] * alpha * s[i];
				//double rk = rPrev[i] - mask[i] * alpha * s[i];
				//rNext[i] = rk;
				rhoNext += rk * rk;
			}

			//std::cout << rhoPrev << " " << rhoNext << "\n";

			if (iterNum > slae.N) break;
			++iterNum;
		}
		//solution = xNext;
	}

	void solve3(double* solution, bool* mask, size_t& iterNum, double eps) {
		xNext = solution;
		double max = 0., rhoNext = 0., rhoPrev = 1., omega;

#pragma omp parallel reduction (+ : rhoNext)
		{
			unsigned tid = omp_get_thread_num();
			for (int i = displs[tid]; i < displs[tid + 1]; ++i) {
				double sum = 0.;
				for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
					sum += slae.data[j] * solution[slae.cols[j]];
				double rk = mask[i] * (slae.rp[i] - sum);
				rNext[i] = rk;
				zNext[i] = 0.;
				rhoNext += rk * rk;
			}
		}

		while (rhoNext * _normB > eps * eps) {    //основной цикл
			//std::cout << "help\n";
			std::swap(xNext, xPrev);
			std::swap(rNext, rPrev);
			std::swap(zNext, zPrev);

			double beta = rhoNext / rhoPrev;

#pragma omp parallel
			{
				unsigned tid = omp_get_thread_num();
				for (int i = displs[tid]; i < displs[tid + 1]; ++i)
					zNext[i] = rPrev[i] + beta * zPrev[i];
					//zNext[i] = rNext[i] + beta * zNext[i];
			}

			omega = 0.;
#pragma omp parallel reduction (+ : omega)
			{
				unsigned tid = omp_get_thread_num();
				for (int i = displs[tid]; i < displs[tid + 1]; ++i) {
					double sum = 0.;
					for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
						sum += slae.data[j] * zNext[slae.cols[j]];
					s[i] = sum;
					omega += sum * zNext[i];
				}
			}

			double alpha = rhoNext / omega;
			rhoPrev = rhoNext, rhoNext = 0.;

#pragma omp parallel reduction (+ : rhoNext)
			{
				unsigned tid = omp_get_thread_num();
				for (int i = displs[tid]; i < displs[tid + 1]; ++i) {
					xNext[i] = xPrev[i] + alpha * zNext[i];
					//xNext[i] += alpha * zNext[i];
					double rk = rPrev[i] - mask[i] * alpha * s[i];
					//double rk = rNext[i] - mask[i] * alpha * s[i];
					rNext[i] = rk;
					rhoNext += rk * rk;
				}
			}

			if (iterNum > slae.N) break;
			++iterNum;
		}
		solution = xNext;
	}

};


class PCG {

private:

	SparseSLAE& slae;

	double* x = nullptr,
		* r = nullptr,
		* z = nullptr,
		* s = nullptr,
		* q = nullptr,
		* D = nullptr,
		* DR = nullptr;

	bool* mask = nullptr;

	double normB = 0., normDR = 0.;

public:

	PCG(SparseSLAE& slae, double* solVector, bool* mask) 
		: slae(slae), x(solVector), mask(mask) {

		r = new double[slae.N];
		z = new double[slae.N];
		s = new double[slae.N];
		q = new double[slae.N];
		D = new double[slae.N];
		DR = new double[slae.N];

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

	~PCG() {
		delete[] r;
		delete[] z;
		delete[] s;
		delete[] q;
		delete[] D;
		delete[] DR;
	}

	void checkSpector() {
		double minA = 1e300, minR = 1e300, minE = 1e300,
			maxA = 0., maxR = 0., maxE = 0.,
			avrA = 0., avrR = 0.;
		for (int i = 0; i < slae.N; ++i) {
			double A = 0., R = 0.;
			for (unsigned j = slae.rows[i]; j < slae.rows[i + 1]; ++j) {
				if (slae.cols[j] == i)
					A = slae.data[j];
				else
					R += fabs(slae.data[j]);
			}
			if (A < minA) minA = A;
			if (A > maxA) maxA = A;
			if (R < minR) minR = R;
			if (R > maxR) maxR = R;
			if (A - R < minE) minE = A - R;
			if (A + R > maxE) maxE = A + R;
			avrA += A; avrR += R;// avrE += E;
		}
		std::cout << "\nminA = " << minA << "   maxA = " << maxA << "  avr = " << avrA / slae.N;
		std::cout << "\nminR = " << minR << "   maxR = " << maxR << "  avr = " << avrR / slae.N;
		std::cout << "\nminE = " << minE << "   maxE = " << maxE << "\n";
	}

	void precond() {
		double norm = 0.;
#pragma omp parallel for reduction (+ : norm)
		for (int i = 0; i < slae.N; ++i) {
			double sum = 0.;
			for (unsigned j = slae.rows[i]; j < slae.rows[i + 1]; ++j) {
				/*if (slae.cols[j] == i)
					D[i] = slae.data[j];*/
				double el = slae.data[j];
				sum += el * el;
			}
			double d = D[i] = sqrt(sum);
			double dr = DR[i] = 1. / d;
			//double dr = DR[i] = 1. / sqrt(sum);
			norm += dr * dr;
			//std::cout << D[i] << " " << DR[i] << "\n";
			//printf("%f %e\n", D[i], DR[i]);
		}
		normDR = norm;

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


	void solve(size_t& iterNum, double eps) {
		double rhoNext = 0., rhoPrev = 1., omega;

#pragma omp parallel for reduction (+ : rhoNext)
		for (int i = 0; i < slae.N; ++i) {
			double sum = 0.;
			for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
				sum += slae.data[j] * x[slae.cols[j]];
			double rk = mask[i] * (slae.rp[i] - sum);
			r[i] = rk;
			z[i] = 0.;
			rhoNext += rk * rk;
		}

		double eps2b = eps * eps * normB;

		while (rhoNext > eps2b) {    //основной цикл

			double beta = rhoNext / rhoPrev;

#pragma omp parallel for
			for (int i = 0; i < slae.N; ++i)
				z[i] = r[i] + beta * z[i];

			omega = 0.;
#pragma omp parallel for reduction (+ : omega)
			for (int i = 0; i < slae.N; ++i) {
				double sum = 0.;
				for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
					sum += slae.data[j] * z[slae.cols[j]];
				s[i] = sum;
				omega += sum * z[i];
			}

			double alpha = rhoNext / omega;
			rhoPrev = rhoNext, rhoNext = 0.;

#pragma omp parallel for reduction (+ : rhoNext)
			for (int i = 0; i < slae.N; ++i) {
				x[i] += alpha * z[i];
				double rk = r[i] -= mask[i] * alpha * s[i];
				rhoNext += rk * rk;
			}

			if (iterNum > slae.N) break;
			++iterNum;
		}
//#pragma omp parallel for reduction (+ : rhoNext)
//		for (int i = 0; i < slae.N; ++i) {
//			double sum = 0.;
//			for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
//				sum += slae.data[j] * x[slae.cols[j]];
//			double rk = mask[i] * (slae.rp[i] - sum);
//			r[i] = rk;
//			z[i] = 0.;
//			rhoNext += rk * rk;
//		}
//		std::cout << "\nrho = " << sqrt(rhoNext / normB) << "  ";
	}


	void solve2(size_t& iterNum, double eps) {
		double rhoNext = 0., rhoPrev = 1., omega;

#pragma omp parallel for reduction (+ : rhoNext)
		for (int i = 0; i < slae.N; ++i) {
			double sum = 0.;
			for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
				sum += slae.data[j] * x[slae.cols[j]];
			double rk = mask[i] * (slae.rp[i] - sum);
			r[i] = rk;
			z[i] = 0.;
			double qk = q[i] = DR[i] * rk;
			rhoNext += qk * rk;
		}

		double eps2b = eps * eps * normB;// *sqrt(normDR);

		while (rhoNext > eps2b) {    //основной цикл

			double beta = rhoNext / rhoPrev;

#pragma omp parallel for
			for (int i = 0; i < slae.N; ++i)
				z[i] = q[i] + beta * z[i];

			omega = 0.;
#pragma omp parallel for reduction (+ : omega)
			for (int i = 0; i < slae.N; ++i) {
				double sum = 0.;
				for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
					sum += slae.data[j] * z[slae.cols[j]];
				s[i] = sum;
				omega += sum * z[i];
			}

			double alpha = rhoNext / omega;
			rhoPrev = rhoNext, rhoNext = 0.;

#pragma omp parallel for reduction (+ : rhoNext)
			for (int i = 0; i < slae.N; ++i) {
				x[i] += alpha * z[i];
				double rk = r[i] -= mask[i] * alpha * s[i];
				double qk = q[i] = DR[i] * rk;
				rhoNext += qk * rk;
			}

			if (iterNum > slae.N) break;
			++iterNum;
		}

#pragma omp parallel for reduction (+ : rhoNext)
		for (int i = 0; i < slae.N; ++i) {
			double sum = 0.;
			for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
				sum += slae.data[j] * x[slae.cols[j]];
			double rk = mask[i] * (slae.rp[i] - sum);
			r[i] = rk;
			z[i] = 0.;
			rhoNext += rk * rk;
			//rhoNext += DR[i] * rk * rk;
		}
		//std::cout << "\nrho = " << sqrt(rhoNext / normB) << "  ";
		//std::cout << "\nrho = " << sqrt(rhoNext / normB / sqrt(normDR)) << "  ";
	}


	void solve3(size_t& iterNum, double eps) {
		double rhoNext = 0., rhoPrev = 1., omega;


		
#pragma omp parallel for reduction (+ : rhoNext)
		for (int i = 0; i < slae.N; ++i) {
			double sum = 0.;
			for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
				sum += slae.data[j] * x[slae.cols[j]];
			//double rk = mask[i] * D[i] * (slae.rp[i] - sum);
			double rk = mask[i] * (slae.rp[i] - sum);
			r[i] = rk;
			z[i] = 0.;
			rhoNext += rk * rk;
		}
		for (int i = 0; i < slae.N; ++i) {
			for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
				slae.data[j] *= DR[slae.cols[j]];
			x[i] *= D[i];
		}
		

		double eps2b = eps * eps * normB / (normDR);

		while (rhoNext > eps2b) {    //основной цикл

			double beta = rhoNext / rhoPrev;

#pragma omp parallel for
			for (int i = 0; i < slae.N; ++i)
				z[i] = r[i] + beta * z[i];

			omega = 0.;
#pragma omp parallel for reduction (+ : omega)
			for (int i = 0; i < slae.N; ++i) {
				double sum = 0.;
				for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
					sum += slae.data[j] * z[slae.cols[j]];
				s[i] = sum;
				omega += sum * z[i];
			}

			double alpha = rhoNext / omega;
			rhoPrev = rhoNext, rhoNext = 0.;

#pragma omp parallel for reduction (+ : rhoNext)
			for (int i = 0; i < slae.N; ++i) {
				x[i] += alpha * z[i];
				double rk = r[i] -= mask[i] * alpha * s[i];
				rhoNext += rk * rk;
			}

			if (iterNum > slae.N) break;
			++iterNum;
		}
		//#pragma omp parallel for reduction (+ : rhoNext)
		//		for (int i = 0; i < slae.N; ++i) {
		//			double sum = 0.;
		//			for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
		//				sum += slae.data[j] * x[slae.cols[j]];
		//			double rk = mask[i] * (slae.rp[i] - sum);
		//			r[i] = rk;
		//			z[i] = 0.;
		//			rhoNext += rk * rk;
		//		}
		//		std::cout << "\nrho = " << sqrt(rhoNext / normB) << "\n";

		for (int i = 0; i < slae.N; ++i)
			x[i] *= DR[i];
	}

	void solve4(size_t& iterNum, double eps) {
		double rhoNext = 0., rhoPrev = 1., omega;

		for (int i = 0; i < slae.N; ++i) {
			for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
				slae.data[j] *= DR[i];
		}
#pragma omp parallel for reduction (+ : rhoNext)
		for (int i = 0; i < slae.N; ++i) {
			double sum = 0.;
			for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
				sum += slae.data[j] * x[slae.cols[j]];
			double rk = mask[i] * (DR[i] * slae.rp[i] - sum);
			//double rk = mask[i] * (slae.rp[i] - sum);
			r[i] = rk;
			z[i] = 0.;
			rhoNext += rk * rk;
		}

		double eps2b = eps * eps * normB * sqrt(normDR);

		while (rhoNext > eps2b) {    //основной цикл

			double beta = rhoNext / rhoPrev;

#pragma omp parallel for
			for (int i = 0; i < slae.N; ++i)
				z[i] = r[i] + beta * z[i];

			omega = 0.;
#pragma omp parallel for reduction (+ : omega)
			for (int i = 0; i < slae.N; ++i) {
				double sum = 0.;
				for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
					sum += slae.data[j] * z[slae.cols[j]];
				s[i] = sum;
				omega += sum * z[i];
			}

			double alpha = rhoNext / omega;
			rhoPrev = rhoNext, rhoNext = 0.;

#pragma omp parallel for reduction (+ : rhoNext)
			for (int i = 0; i < slae.N; ++i) {
				x[i] += alpha * z[i];
				double rk = r[i] -= mask[i] * alpha * s[i];
				rhoNext += rk * rk;
			}

			if (iterNum > slae.N) break;
			++iterNum;
		}
		//#pragma omp parallel for reduction (+ : rhoNext)
		//		for (int i = 0; i < slae.N; ++i) {
		//			double sum = 0.;
		//			for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
		//				sum += slae.data[j] * x[slae.cols[j]];
		//			double rk = mask[i] * (slae.rp[i] - sum);
		//			r[i] = rk;
		//			z[i] = 0.;
		//			rhoNext += rk * rk;
		//		}
		//		std::cout << "\nrho = " << sqrt(rhoNext / normB) << "\n";
	}

};


class CGV2 {

private:

	SparseSLAE& slae;

	double* x = nullptr,
		* r = nullptr,
		* z = nullptr,
		* s = nullptr;

	bool* mask = nullptr;

	double normB = 0., normDR = 0.;

public:

	CGV2(SparseSLAE& slae, double* solVector, bool* mask)
		: slae(slae), x(solVector), mask(mask) {

		r = new double[slae.N];
		z = new double[slae.N];
		s = new double[slae.N];

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

	~CGV2() {
		delete[] r;
		delete[] z;
		delete[] s;
	}

	void solve(size_t& iterNum, double eps) {
		double rhoNext = 0., rhoPrev = 1., omega;

#pragma omp parallel for reduction (+ : rhoNext)
		for (int i = 0; i < slae.N; ++i) {
			double sum = 0.;
			for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
				sum += slae.data[j] * x[slae.cols[j]];
			double rk = mask[i] * (slae.rp[i] - sum);
			r[i] = rk;
			z[i] = 0.;
			rhoNext += rk * rk;
		}

		double eps2b = eps * eps * normB;

		while (rhoNext > eps2b) {    //основной цикл

			double beta = rhoNext / rhoPrev;

#pragma omp parallel for
			for (int i = 0; i < slae.N; ++i)
				z[i] = r[i] + beta * z[i];

			omega = 0.;
#pragma omp parallel for reduction (+ : omega)
			for (int i = 0; i < slae.N; ++i) {
				double sum = 0.;
				for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
					sum += slae.data[j] * z[slae.cols[j]];
				s[i] = sum;
				omega += sum * z[i];
			}

			double alpha = rhoNext / omega;
			rhoPrev = rhoNext, rhoNext = 0.;

#pragma omp parallel for reduction (+ : rhoNext)
			for (int i = 0; i < slae.N; ++i) {
				x[i] += alpha * z[i];
				double rk = r[i] -= mask[i] * alpha * s[i];
				rhoNext += rk * rk;
			}

			if (iterNum > slae.N) break;
			++iterNum;
		}
	}


	void solve2(size_t& iterNum, double eps) {
		double rhoNext = 0., rhoPrev = 1., omega;

#pragma omp parallel for reduction (+ : rhoNext)
		for (int i = 0; i < slae.N; ++i) {
			double sum = 0.;
			for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
				sum += slae.data[j] * x[slae.cols[j]];
			double rk = mask[i] * (slae.rp[i] - sum);
			r[i] = rk;
			z[i] = 0.;
			rhoNext += rk * rk;
		}

		double eps2b = eps * eps * normB;

		while (rhoNext > eps2b) {    //основной цикл

			double beta = rhoNext / rhoPrev;

#pragma omp parallel for
			for (int i = 0; i < slae.N; ++i)
				z[i] = r[i] + beta * z[i];

			omega = 0.;
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
				s[i] = sum1;
				s[i + 1] = sum2;
				omega += sum1 * z[i] + sum2 * z[i + 1];
			}
			/*for (int i = 0; i < slae.N; i += 2) {
				register double sum1 = 0., sum2 = 0.;
				int shift = slae.rows[i + 1] - slae.rows[i];
				for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j) {
					register double ze = z[slae.cols[j]];
					sum1 += slae.data[j] * ze;
					sum2 += slae.data[j + shift] * ze;
				}
				s[i] = sum1;
				s[i + 1] = sum2;
				omega += sum1 * z[i] + sum2 * z[i + 1];
			}*/

			double alpha = rhoNext / omega;
			rhoPrev = rhoNext, rhoNext = 0.;

#pragma omp parallel for reduction (+ : rhoNext)
			for (int i = 0; i < slae.N; ++i) {
				x[i] += alpha * z[i];
				double rk = r[i] -= mask[i] * alpha * s[i];
				rhoNext += rk * rk;
			}

			if (iterNum > slae.N) break;
			++iterNum;
		}
	}

};



// Block Jacobi Preconditioned Conjugate Gradient Method Vectorized by 2x2
class BJCG2 {

private:

	SparseSLAE& slae;

	double* x = nullptr,
		* r = nullptr,
		* z = nullptr,
		* s = nullptr,
		* dr = nullptr,
		* D = nullptr,
		* DR = nullptr;

	bool* mask = nullptr;

	double normB = 0., normDR = 0.;

public:

	BJCG2(SparseSLAE& slae, double* solVector, bool* mask)
		: slae(slae), x(solVector), mask(mask) {

		r = new double[slae.N];
		z = new double[slae.N];
		s = new double[slae.N];
		dr = new double[slae.N];
		D = new double[2 * slae.N];
		DR = new double[2 * slae.N];

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

	~BJCG2() {
		delete[] r;
		delete[] z;
		delete[] s;
		delete[] dr;
		delete[] D;
		delete[] DR;
	}

	void precond() {
		double norm = 0.;
		for (int i = 0; i < slae.N; i += 2) {
			int i2 = 2 * i;
			double sum = 0.;
			for (unsigned j = slae.rows[i]; j < slae.rows[i + 1]; j += 2) {
				if (slae.cols[j] == i) {
					D[i2] = slae.data[j];
					D[i2 + 1] = slae.data[j + 1];
					int shift = slae.rows[i + 1] - slae.rows[i];
					D[i2 + 2] = slae.data[j + shift];
					D[i2 + 3] = slae.data[j + shift + 1];
					break;
				}
				/*double el = slae.data[j];
				sum += el * el;*/
			}
			
			//double dr = DR[i] = 1. / d;
			double invDet = 1. / (D[i2] * D[i2 + 3] - D[i2 + 1] * D[i2 + 2]);
			DR[i2] = D[i2 + 3] * invDet;
			DR[i2 + 1] = D[i2 + 1] * invDet;
			DR[i2 + 2] = D[i2 + 2] * invDet;
			DR[i2 + 3] = D[i2] * invDet;
			//printf("%e %e %e %e\n", DR[i2], DR[i2 + 1], DR[i2 + 2], DR[i2 + 3]);
			
			norm += DR[i2] * DR[i2] + DR[i2 + 1] * DR[i2 + 1] + DR[i2 + 2] * DR[i2 + 2] + DR[i2 + 3] * DR[i2 + 3];
			//std::cout << D[i] << " " << DR[i] << "\n";
			//printf("%f %e\n", D[i], DR[i]);
		}
		normDR = norm;
	}

	void solve(size_t& iterNum, double eps) {
		double rhoNext = 0., rhoPrev = 1., omega;

#pragma omp parallel for reduction (+ : rhoNext)
		for (int i = 0; i < slae.N; ++i) {
			double sum = 0.;
			for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
				sum += slae.data[j] * x[slae.cols[j]];
			double rk = mask[i] * (slae.rp[i] - sum);
			r[i] = rk;
			z[i] = 0.;
			rhoNext += rk * rk;
		}

		double eps2b = eps * eps * normB;// *(normDR * normDR);

		while (rhoNext > eps2b) {    //основной цикл

			double beta = rhoNext / rhoPrev;

#pragma omp parallel for
			for (int i = 0; i < slae.N; ++i)
				z[i] = r[i] + beta * z[i];

			omega = 0.;
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
				s[i] = sum1;
				s[i + 1] = sum2;
				omega += sum1 * z[i] + sum2 * z[i + 1];
			}
			/*for (int i = 0; i < slae.N; i += 2) {
				register double sum1 = 0., sum2 = 0.;
				int shift = slae.rows[i + 1] - slae.rows[i];
				for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j) {
					register double ze = z[slae.cols[j]];
					sum1 += slae.data[j] * ze;
					sum2 += slae.data[j + shift] * ze;
				}
				s[i] = sum1;
				s[i + 1] = sum2;
				omega += sum1 * z[i] + sum2 * z[i + 1];
			}*/

			double alpha = rhoNext / omega;
			rhoPrev = rhoNext, rhoNext = 0.;

#pragma omp parallel for reduction (+ : rhoNext)
			for (int i = 0; i < slae.N; ++i) {
				x[i] += alpha * z[i];
				double rk = r[i] -= mask[i] * alpha * s[i];
				rhoNext += rk * rk;
			}

			if (iterNum > slae.N) break;
			++iterNum;
		}
	}
	
	void solve2(size_t& iterNum, double eps) {
		double rhoNext = 0., rhoPrev = 1., omega;

#pragma omp parallel for reduction (+ : rhoNext)
		for (int i = 0; i < slae.N; i += 2) {
			double sum1 = 0., sum2 = 0.;
			int shift = slae.rows[i + 1] - slae.rows[i];
			for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j) {
				sum1 += slae.data[j] * x[slae.cols[j]];
				sum2 += slae.data[j + shift] * x[slae.cols[j + shift]];
				/*int col = slae.cols[j];
				register double xe1 = x[col], xe2 = x[col + 1];
				sum1 += slae.data[j] * xe1 + slae.data[j + 1] * xe2;
				sum2 += slae.data[j + shift] * xe1 + slae.data[j + shift + 1] * xe2;*/
			}
			double rk1 = mask[i] * (slae.rp[i] - sum1);
			double rk2 = mask[i + 1] * (slae.rp[i + 1] - sum2);
			r[i] = rk1;
			r[i + 1] = rk2;
			z[i + 1] = z[i] = 0.;

			dr[i] = DR[2 * i] * rk1 + DR[2 * i + 1] * rk2;
			dr[i + 1] = DR[2 * i + 2] * rk1 + DR[2 * i + 3] * rk2;
			//rhoNext += rk1 * rk1 + rk2 * rk2;
			rhoNext += dr[i] * rk1 + dr[i + 1] * rk2;
		}

		double eps2b = eps * eps * normB * sqrt(normDR);

		while (rhoNext > eps2b) {    //основной цикл

			double beta = rhoNext / rhoPrev;

#pragma omp parallel for
			for (int i = 0; i < slae.N; ++i)
				//z[i] = r[i] + beta * z[i];
				z[i] = dr[i] + beta * z[i];

			omega = 0.;
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
				s[i] = sum1;
				s[i + 1] = sum2;
				omega += sum1 * z[i] + sum2 * z[i + 1];
			}

			double alpha = rhoNext / omega;
			rhoPrev = rhoNext, rhoNext = 0.;

#pragma omp parallel for reduction (+ : rhoNext)
			for (int i = 0; i < slae.N; i += 2) {
				x[i] += alpha * z[i];
				x[i + 1] += alpha * z[i + 1];
				double rk1 = r[i] -= mask[i] * alpha * s[i];
				double rk2 = r[i + 1] -= mask[i + 1] * alpha * s[i + 1];

				dr[i] = DR[2 * i] * rk1 + DR[2 * i + 1] * rk2;
				dr[i + 1] = DR[2 * i + 2] * rk1 + DR[2 * i + 3] * rk2;
				//rhoNext += rk1 * rk1 + rk2 * rk2;
				rhoNext += dr[i] * rk1 + dr[i + 1] * rk2;
			}

			if (iterNum > slae.N) break;
			++iterNum;
		}
	}

};