#include "SLAE_Solvers.cuh"


//const int BS = 1024;


std::vector<double> Gauss(StripSLAE& slae) {
	//прямой ход
	for (int k = 0; k < slae.size(); ++k) {
		double leadEl = slae(k, k);
		size_t lineEnd = std::min(slae.size(), k + slae.width() + 1);
		//деление на ведущий элемент
#pragma omp parallel for
		for (int j = k + 1; j < lineEnd; ++j)
			slae(k, j) /= leadEl;
		slae.b(k) /= leadEl;
		//if (k == slae.size() - 1) slae.print();
#pragma omp parallel for
		for (int i = k + 1; i < lineEnd; ++i) {
			for (int j = k + 1; j < lineEnd; ++j)
				slae(i, j) -= slae(k, j) * slae(i, k);
			slae.b(i) -= slae.b(k) * slae(i, k);
		}
	}
	//slae.print();
	//обратный ход
	for (int i = (int)slae.size() - 2; i >= 0; --i) {
		size_t lineEnd = std::min(slae.size(), i + slae.width() + 1);
		for (int j = (int)lineEnd - 1; j > i; --j)
			slae.b(i) -= slae.b(j) * slae(i, j);
	}

	std::vector<double> solution(slae.size());
	for (size_t i = 0; i < slae.size(); ++i)
		solution[i] = slae.b(i);
	return solution;
}


//Вычитание векторов
std::vector<double> operator-(const std::vector<double>& vec1, const std::vector<double>& vec2) {
	//assert(vec1.size() == vec2.size() && "Vector size mismatch!");
	std::vector<double> result(vec1);
	for (size_t i = 0; i < result.size(); ++i)
		result[i] -= vec2[i];
	return result;
}

//Кубическая норма
double normInf(const std::vector<double>& vec) {
	double max = 0;
	for (size_t i = 0; i < vec.size(); ++i)
		if (fabs(vec[i]) > max)
			max = fabs(vec[i]);
	return max;
}

//Евклидова норма
double norm2(const std::vector<double>& vec) {
	double sum = 0;
	for (size_t i = 0; i < vec.size(); ++i)
		sum += vec[i] * vec[i];
	return sum;
}


//Метод сопряжённых градиентов
void conjugateGradient(StripSLAE& slae, std::vector<double>& solution, size_t& iterNum, double eps) {
	std::vector<double> xNext(solution), xPrev(solution),
		rNext(slae.size()), rPrev(slae.size()),
		zNext(slae.size()), zPrev(slae.size()),
		Az(slae.size());

	std::function<bool()> exitCondition; //условие завершения цикла
	if (iterNum == 0)
		exitCondition = [&]() { return sqrt(norm2(xNext - xPrev)) > eps; };
	else {
		size_t maxIter = iterNum;
		exitCondition = [maxIter, &iterNum]() { return iterNum < maxIter; };
		iterNum = 0;
	}
	//size_t W = slae.width(), N = slae.size();
	size_t W = slae.width(), N = slae.factN;
#pragma omp parallel for
	for (int i = 0; i < N; ++i) {
		double sum = 0.;
		size_t lineBegin = W > i ? W - i : 0;
		size_t lineEnd = 2 * W < W + N - i - 1 ? 2 * W : W + N - i - 1;
		for (size_t j = lineBegin; j <= lineEnd; ++j)
			sum += slae.data()[i * (2 * W + 1) + j] * solution[i - W + j];
		zNext[i] = rNext[i] = slae.b(i) - sum;
	}

	do {    //основной цикл
		xPrev.swap(xNext);
		rPrev.swap(rNext);
		zPrev.swap(zNext);
		
		double rPrevScal = 0., rNextScal = 0., Az_z = 0.;
#pragma omp parallel for reduction (+ : rPrevScal, Az_z)
		for (int i = 0; i < N; ++i) {
			rPrevScal += rPrev[i] * rPrev[i];
			double sum = 0.;
			size_t lineBegin = W > i ? W - i : 0;
			size_t lineEnd = 2 * W < W + N - i - 1 ? 2 * W : W + N - i - 1;
			for (size_t j = lineBegin; j <= lineEnd; ++j)
				sum += slae.data()[i * (2 * W + 1) + j] * zPrev[i - W + j];
			Az[i] = sum;
			Az_z += sum * zPrev[i];
		}
		double alpha = rPrevScal / Az_z;
#pragma omp parallel for reduction (+ : rNextScal)
		for (int i = 0; i < N; ++i) {
			xNext[i] = xPrev[i] + alpha * zPrev[i];
			double rk = rPrev[i] - alpha * Az[i];
			rNext[i] = rk;
			rNextScal += rk * rk;
		}
		double beta = rNextScal / rPrevScal;
		//std::cout << iterNum << "   " << alpha << "    " << beta << "\n";
#pragma omp parallel for
		for (int i = 0; i < N; ++i)
			zNext[i] = rNext[i] + beta * zPrev[i];
		/*for (int i = 0; i < slae.size(); ++i)
			std::cout << xNext[i] << " ";
		std::cout << "\n";*/
		//std::cout << iterNum << "   " << norm2(xNext - xPrev) << "\n";

		++iterNum;
	} while (exitCondition());
	//std::cout << iterNum << "\n";
	solution = xNext;
}

//Метод сопряжённых градиентов для разреженной СЛАУ
void conjugateGradient(SparseSLAE& slae, double* solution, bool* mask, size_t& iterNum, double eps) {
	double* xNext = solution,
		* xPrev = new double[slae.N],
		* rNext = new double[slae.N],
		* rPrev = new double[slae.N],
		* zNext = new double[slae.N],
		* zPrev = new double[slae.N],
		* Az = new double[slae.N];
	double _normB = 0.;
	for (int i = 0; i < slae.N; ++i) {
		double value = fabs(slae.rp[i]);
		if (value > _normB)
			_normB = value;
	}
	_normB = 1. / _normB;
	std::function<bool()> exitCondition; //условие завершения цикла
	if (iterNum == 0)
		exitCondition = [&]() {
			double max = 0.;
			for (int i = 0; i < slae.N; ++i) {
				double value = fabs(rNext[i]);
				//double value = fabs(xNext[i] - xPrev[i]) / _normB;
				if (value > max)
					max = value;
			}
			return max * _normB > eps;
		};
	else {
		size_t maxIter = iterNum;
		exitCondition = [maxIter, &iterNum]() { return iterNum < maxIter; };
		iterNum = 0;
	}

#pragma omp parallel for
	for (int i = 0; i < slae.N; ++i) {
		double sum = 0.;
		for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
			sum += slae.data[j] * solution[slae.cols[j]];
		zNext[i] = rNext[i] = slae.rp[i] - sum;
	}

	do {    //основной цикл
		//std::clog << "help\n";
		std::swap(xNext, xPrev);
		std::swap(rNext, rPrev);
		std::swap(zNext, zPrev);

		double rPrevScal = 0., rNextScal = 0., Az_z = 0.;
#pragma omp parallel for reduction (+ : rPrevScal, Az_z)
		for (int i = 0; i < slae.N; ++i) {
			rPrevScal += rPrev[i] * rPrev[i];
			double sum = 0.;
			//for (int j = slae.rows[i]; j < mask[i] * slae.rows[i + 1]; ++j)
			for (int j = slae.rows[i]; j < slae.rows[i + 1]; ++j)
				sum += slae.data[j] * zPrev[slae.cols[j]];
			Az[i] = sum;
			Az_z += sum * zPrev[i];
		}
		double alpha = rPrevScal / Az_z;
#pragma omp parallel for reduction (+ : rNextScal)
		for (int i = 0; i < slae.N; ++i) {
			xNext[i] = xPrev[i] + mask[i] * alpha * zPrev[i];
			//xNext[i] = xPrev[i] + alpha * zPrev[i];
			double rk = rPrev[i] - alpha * Az[i];
			rNext[i] = rk;
			rNextScal += rk * rk;
		}
		double beta = rNextScal / rPrevScal;
		
#pragma omp parallel for
		for (int i = 0; i < slae.N; ++i)
			zNext[i] = rNext[i] + beta * zPrev[i];

		//std::cout << iterNum << "   " << rNextScal << "\n";
		//if (iterNum > 500)
			//break;
		++iterNum;
	} while (exitCondition());
	//delete[] xPrev;
	delete[] rNext;
	delete[] rPrev;
	delete[] zNext;
	delete[] zPrev;
	delete[] Az;
}

template <typename fp>
__global__ void cgInit(fp* matrix, fp* b, fp* xNext, fp* rNext, fp* zNext, int N, int W) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	fp sum = 0.;
	int lineBegin = W > i ? W - i : 0;
	int lineEnd = 2 * W < W + N - i - 1 ? 2 * W : W + N - i - 1;
	for (int j = lineBegin; j <= lineEnd; ++j)
		sum += matrix[i * (2 * W + 1) + j] * xNext[i - W + j];
	zNext[i] = rNext[i] = b[i] - sum;
}

template <typename fp>
__global__ void cg1(fp* matrix, fp* rPrev, fp* zPrev, fp* Az, fp* rPrevScal, fp* Az_z, int N, int W) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	atomicAdd(rPrevScal, rPrev[i] * rPrev[i]);
	fp sum = 0.;
	int lineBegin = W > i ? W - i : 0;
	int lineEnd = 2 * W < W + N - i - 1 ? 2 * W : W + N - i - 1;
	for (int j = lineBegin; j <= lineEnd; ++j)
		sum += matrix[i * (2 * W + 1) + j] * zPrev[i - W + j];
	Az[i] = sum;
	//printf("%d\n", i);
	atomicAdd(Az_z, sum * zPrev[i]);
}

template <typename fp>
__global__ void cg2(fp* xNext, fp* xPrev, fp* rNext, fp* rPrev, fp* zPrev, fp* Az, fp alpha, fp* rNextScal) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	xNext[i] = xPrev[i] + alpha * zPrev[i];
	fp rk = rPrev[i] - alpha * Az[i];
	rNext[i] = rk;
	atomicAdd(rNextScal, rk * rk);
	//printf("%d\n", i);
}

template <typename fp>
__global__ void cg3(fp* rNext, fp* zNext, fp* zPrev, fp beta) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	zNext[i] = rNext[i] + beta * zPrev[i];
	//printf("%d\n", i);
}

template <typename fp>
__global__ void exitNorm(fp* xNext, fp* xPrev, fp* norm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	fp sub = xNext[i] - xPrev[i];
	atomicAdd(norm, sub * sub);
	//norm = fmax(norm, fabs(sub));
}

template <typename fp>
__global__ void exitNormR(fp* rNext, fp* norm) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	atomicAdd(norm, rNext[i] * rNext[i]);
}

void conjugateGradientCuda(StripSLAE& slae, std::vector<double>& solution, size_t& iterNum, double eps) {
	cudaError_t cudaStatus;
	eps = eps * eps;

	double* dev_matrix = nullptr, * dev_b = nullptr, \
		* xNext = nullptr, * xPrev = nullptr, \
		* rNext = nullptr, * rPrev = nullptr,
		* zNext = nullptr, * zPrev = nullptr,
		* Az = nullptr;
	double rPrevScal = 0., rNextScal = 0., Az_z = 0., norm = 0.;
	double* dev_rPrevScal = nullptr, * dev_rNextScal = nullptr, * dev_Az_z = nullptr, * dev_norm = nullptr;

	/*double bNorm = 0.;
	for (size_t i = 0; i < slae.size(); ++i)
		bNorm += slae.b(i) * slae.b(i);*/

	std::function<bool()> exitCondition; //условие завершения цикла
	if (iterNum == 0)
		exitCondition = [&]() { return norm > eps; };
	else {
		size_t maxIter = iterNum;
		exitCondition = [maxIter, &iterNum]() { return iterNum < maxIter; };
		iterNum = 0;
	}

	int blockSize = BS; std::min(BS, (int)slae.size());
	dim3 block(blockSize);
	dim3 grid((int)slae.size() / BS);
	//dim3 grid(((int)slae.size() + blockSize - 1) / blockSize);

	//std::cout << grid.x << "\n";

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		throw("no device");
	}

	int lineWidth = slae.width() * 2 + 1;
	cudaMalloc((void**)&dev_matrix, slae.size() * lineWidth * sizeof(double));
	cudaMalloc((void**)&dev_b, slae.size() * sizeof(double));
	cudaMalloc((void**)&xNext, slae.size() * sizeof(double));
	cudaMalloc((void**)&xPrev, slae.size() * sizeof(double));
	cudaMalloc((void**)&rNext, slae.size() * sizeof(double));
	cudaMalloc((void**)&rPrev, slae.size() * sizeof(double));
	cudaMalloc((void**)&zNext, slae.size() * sizeof(double));
	cudaMalloc((void**)&zPrev, slae.size() * sizeof(double));
	cudaMalloc((void**)&Az, slae.size() * sizeof(double));
	cudaMalloc((void**)&dev_rPrevScal, sizeof(double));
	cudaMalloc((void**)&dev_rNextScal, sizeof(double));
	cudaMalloc((void**)&dev_Az_z, sizeof(double));
	cudaMalloc((void**)&dev_norm, sizeof(double));

	cudaMemcpy(dev_matrix, slae.data(), slae.size() * lineWidth * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, slae.bData(), slae.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(xNext, solution.data(), slae.size() * sizeof(double), cudaMemcpyHostToDevice);

	//std::cout << "cudaStatus " << cudaStatus << "\n";

	cgInit<double><<<grid, block>>>(dev_matrix, dev_b, xNext, rNext, zNext, (int)slae.size(), (int)slae.width());

	//iterNum = 0;
	//for (size_t i = 0; i < 354; ++i) {
	do {
		std::swap(xNext, xPrev);
		std::swap(rNext, rPrev);
		std::swap(zNext, zPrev);

		rPrevScal = 0., rNextScal = 0., Az_z = 0.;
		cudaMemcpy(dev_rNextScal, &rNextScal, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_rPrevScal, &rPrevScal, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Az_z, &Az_z, sizeof(double), cudaMemcpyHostToDevice);
		norm = 0.;
		cudaMemcpy(dev_norm, &norm, sizeof(double), cudaMemcpyHostToDevice);
		cg1<double><<<grid, block>>>(dev_matrix, rPrev, zPrev, Az, dev_rPrevScal, dev_Az_z, (int)slae.factN, (int)slae.width());
		cudaDeviceSynchronize();
		cudaMemcpy(&rPrevScal, dev_rPrevScal, sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(&Az_z, dev_Az_z, sizeof(double), cudaMemcpyDeviceToHost);
		double alpha = rPrevScal / Az_z;
		cg2<double><<<grid, block>>>(xNext, xPrev, rNext, rPrev, zPrev, Az, alpha, dev_rNextScal);
		cudaDeviceSynchronize();
		cudaMemcpy(&rNextScal, dev_rNextScal, sizeof(double), cudaMemcpyDeviceToHost);
		double beta = rNextScal / rPrevScal;
		//std::cout << iterNum << "   " << alpha << "    " << beta << "\n";
		cg3<double><<<grid, block>>>(rNext, zNext, zPrev, beta);
		cudaDeviceSynchronize();

		exitNorm<double><<<grid, block>>>(xNext, xPrev, dev_norm);
		//exitNormR<double><<<grid, block>>>(rNext, dev_norm);
		cudaDeviceSynchronize();
		cudaMemcpy(&norm, dev_norm, sizeof(double), cudaMemcpyDeviceToHost);
		//cudaMemcpy(solution.data(), rPrev, slae.size() * sizeof(double), cudaMemcpyDeviceToHost);
		/*for (int i = 0; i < slae.size(); ++i)
			std::cout << solution[i] << " ";
		std::cout << "\n";*/
		//std::cout << iterNum << "   " << sqrt(norm) << "\n";
		++iterNum;
	} while (exitCondition());

	cudaMemcpy(solution.data(), xNext, slae.size() * sizeof(double), cudaMemcpyDeviceToHost);
	cudaFree(dev_matrix);
	cudaFree(dev_b);
	cudaFree(xNext); cudaFree(xPrev);
	cudaFree(rNext); cudaFree(rPrev);
	cudaFree(zNext); cudaFree(zPrev);
	cudaFree(Az);
	cudaFree(dev_rNextScal);
	cudaFree(dev_rPrevScal);
	cudaFree(dev_Az_z);
}

template<typename fp>
void conjugateGradientBase(CudaSLAE<fp>& slae, fp* solution, size_t& iterNum, fp eps) {
	//std::clog << "hello there\n";
	eps = eps * eps;

	fp
		* xNext = nullptr, * xPrev = nullptr, \
		* rNext = nullptr, * rPrev = nullptr,
		* zNext = nullptr, * zPrev = nullptr,
		* Az = nullptr;
	fp rPrevScal = 0., rNextScal = 0., Az_z = 0., norm = 0.;
	fp* dev_rPrevScal = nullptr, * dev_rNextScal = nullptr, * dev_Az_z = nullptr, * dev_norm = nullptr;

	/*fp bNorm = 0.;
	for (size_t i = 0; i < slae.size(); ++i)
		bNorm += slae.b(i) * slae.b(i);*/

	std::function<bool()> exitCondition; //условие завершения цикла
	if (iterNum == 0)
		exitCondition = [&]() { return norm > eps; };
	else {
		size_t maxIter = iterNum;
		exitCondition = [maxIter, &iterNum]() { return iterNum < maxIter; };
		iterNum = 0;
	}

	int blockSize = BS; std::min(BS, (int)slae.size());
	dim3 block(blockSize);
	//dim3 grid((int)slae.size() / BS);
	dim3 grid((int)slae.memLen / BS);
	//dim3 grid(((int)slae.size() + blockSize - 1) / blockSize);

	//std::cout << grid.x << "\n";

	int lineWidth = slae.width() * 2 + 1;
	cudaMalloc((void**)&xNext, slae.memLen * sizeof(fp));
	cudaMalloc((void**)&xPrev, slae.memLen * sizeof(fp));
	cudaMalloc((void**)&rNext, slae.memLen * sizeof(fp));
	cudaMalloc((void**)&rPrev, slae.memLen * sizeof(fp));
	cudaMalloc((void**)&zNext, slae.memLen * sizeof(fp));
	cudaMalloc((void**)&zPrev, slae.memLen * sizeof(fp));
	cudaMalloc((void**)&Az, slae.memLen * sizeof(fp));
	cudaMalloc((void**)&dev_rPrevScal, sizeof(fp));
	cudaMalloc((void**)&dev_rNextScal, sizeof(fp));
	cudaMalloc((void**)&dev_Az_z, sizeof(fp));
	cudaMalloc((void**)&dev_norm, sizeof(fp));

	cudaMemcpy(xNext, solution, slae.size() * sizeof(fp), cudaMemcpyDeviceToDevice);
	//xNext = solution;

	//std::cout << "cudaStatus " << cudaStatus << "\n";

	cgInit<fp><<<grid, block>>>(slae.matrix, slae.rp, xNext, rNext, zNext, slae.size(), slae.width());

	//iterNum = 0;
	//for (size_t i = 0; i < 354; ++i) {
	do {
		std::swap(xNext, xPrev);
		std::swap(rNext, rPrev);
		std::swap(zNext, zPrev);

		rPrevScal = 0., rNextScal = 0., Az_z = 0.;
		cudaMemcpy(dev_rNextScal, &rNextScal, sizeof(fp), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_rPrevScal, &rPrevScal, sizeof(fp), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Az_z, &Az_z, sizeof(fp), cudaMemcpyHostToDevice);
		norm = 0.;
		cudaMemcpy(dev_norm, &norm, sizeof(fp), cudaMemcpyHostToDevice);
		cg1<fp><<<grid, block>>>(slae.matrix, rPrev, zPrev, Az, dev_rPrevScal, dev_Az_z, slae.size(), slae.width());
		cudaDeviceSynchronize();
		cudaMemcpy(&rPrevScal, dev_rPrevScal, sizeof(fp), cudaMemcpyDeviceToHost);
		cudaMemcpy(&Az_z, dev_Az_z, sizeof(fp), cudaMemcpyDeviceToHost);
		fp alpha = rPrevScal / Az_z;
		cg2<fp><<<grid, block>>>(xNext, xPrev, rNext, rPrev, zPrev, Az, alpha, dev_rNextScal);
		cudaDeviceSynchronize();
		cudaMemcpy(&rNextScal, dev_rNextScal, sizeof(fp), cudaMemcpyDeviceToHost);
		fp beta = rNextScal / rPrevScal;
		//std::cout << iterNum << "   " << alpha << "    " << beta << "\n";
		cg3<fp><<<grid, block>>>(rNext, zNext, zPrev, beta);
		cudaDeviceSynchronize();

		exitNorm<fp><<<grid, block>>>(xNext, xPrev, dev_norm);
		//exitNormR<double><<<grid, block>>>(rNext, dev_norm);
		cudaDeviceSynchronize();
		cudaMemcpy(&norm, dev_norm, sizeof(fp), cudaMemcpyDeviceToHost);
		//cudaMemcpy(solution.data(), rPrev, slae.size() * sizeof(fp), cudaMemcpyDeviceToHost);
		/*for (int i = 0; i < slae.size(); ++i)
			std::cout << solution[i] << " ";
		std::cout << "\n";*/
		//std::cout << iterNum << "   " << sqrt(norm) << "\n";
		++iterNum;
	} while (exitCondition());

	cudaMemcpy(solution, xNext, slae.size() * sizeof(fp), cudaMemcpyDeviceToDevice);
	cudaFree(xNext); cudaFree(xPrev);
	cudaFree(rNext); cudaFree(rPrev);
	cudaFree(zNext); cudaFree(zPrev);
	cudaFree(Az);
	cudaFree(dev_rNextScal);
	cudaFree(dev_rPrevScal);
	cudaFree(dev_Az_z);
}

void conjugateGradient(CudaSLAE<double>& slae, double* solution, size_t& iterNum, double eps) {
	conjugateGradientBase(slae, solution, iterNum, eps);
}

void conjugateGradient(CudaSLAE<float>& slae, float* solution, size_t& iterNum, float eps) {
	conjugateGradientBase(slae, solution, iterNum, eps);
}


//template <typename fp>
//__global__ void cgInit(fp* matrix, fp* b, fp* xNext, fp* rNext, fp* zNext, int N, int W) {
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	fp sum = 0.;
//	int lineBegin = W > i ? W - i : 0;
//	int lineEnd = 2 * W < W + N - i - 1 ? 2 * W : W + N - i - 1;
//	for (int j = lineBegin; j <= lineEnd; ++j)
//		sum += matrix[i * (2 * W + 1) + j] * xNext[i - W + j];
//	zNext[i] = rNext[i] = b[i] - sum;
//}

//template <typename fp>
//__global__ void cg1(fp* matrix, fp* rPrev, fp* zPrev, fp* Az, fp* rPrevScal, fp* Az_z, int N, int W) {
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	atomicAdd(rPrevScal, rPrev[i] * rPrev[i]);
//	fp sum = 0.;
//	int lineBegin = W > i ? W - i : 0;
//	int lineEnd = 2 * W < W + N - i - 1 ? 2 * W : W + N - i - 1;
//	for (int j = lineBegin; j <= lineEnd; ++j)
//		sum += matrix[i * (2 * W + 1) + j] * zPrev[i - W + j];
//	Az[i] = sum;
//	//printf("%d\n", i);
//	atomicAdd(Az_z, sum * zPrev[i]);
//}

template <typename fp>
__global__ void cg2mask(fp* xNext, fp* xPrev, fp* rNext, fp* rPrev, fp* zPrev, fp* Az, fp alpha, fp* rNextScal, bool* mask) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	xNext[i] = xPrev[i] + mask[i] * alpha * zPrev[i];
	fp rk = rPrev[i] - alpha * Az[i];
	rNext[i] = rk;
	atomicAdd(rNextScal, rk * rk);
	//printf("%d\n", i);
}

//template <typename fp>
//__global__ void cg3(fp* rNext, fp* zNext, fp* zPrev, fp beta) {
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	zNext[i] = rNext[i] + beta * zPrev[i];
//	//printf("%d\n", i);
//}

//template <typename fp>
//__global__ void exitNorm(fp* xNext, fp* xPrev, fp* norm) {
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	fp sub = xNext[i] - xPrev[i];
//	atomicAdd(norm, sub * sub);
//	//norm = fmax(norm, fabs(sub));
//}

template<typename fp>
void conjugateGradientBase(CudaSLAE<fp>& slae, fp* solution, bool* mask, size_t& iterNum, fp eps) {
	//std::clog << "hello there\n";
	eps = eps * eps;

	fp
		* xNext = nullptr, * xPrev = nullptr, \
		* rNext = nullptr, * rPrev = nullptr,
		* zNext = nullptr, * zPrev = nullptr,
		* Az = nullptr;
	fp rPrevScal = 0., rNextScal = 0., Az_z = 0., norm = 0.;
	fp* dev_rPrevScal = nullptr, * dev_rNextScal = nullptr, * dev_Az_z = nullptr, * dev_norm = nullptr;

	/*fp bNorm = 0.;
	for (size_t i = 0; i < slae.size(); ++i)
		bNorm += slae.b(i) * slae.b(i);*/

	std::function<bool()> exitCondition; //условие завершения цикла
	if (iterNum == 0)
		exitCondition = [&]() { return norm > eps; };
	else {
		size_t maxIter = iterNum;
		exitCondition = [maxIter, &iterNum]() { return iterNum < maxIter; };
		iterNum = 0;
	}

	int blockSize = BS; std::min(BS, (int)slae.size());
	dim3 block(blockSize);
	//dim3 grid((int)slae.size() / BS);
	dim3 grid((int)slae.memLen / BS);
	//dim3 grid(((int)slae.size() + blockSize - 1) / blockSize);

	//std::cout << grid.x << "\n";

	int lineWidth = slae.width() * 2 + 1;
	cudaMalloc((void**)&xNext, slae.memLen * sizeof(fp));
	cudaMalloc((void**)&xPrev, slae.memLen * sizeof(fp));
	cudaMalloc((void**)&rNext, slae.memLen * sizeof(fp));
	cudaMalloc((void**)&rPrev, slae.memLen * sizeof(fp));
	cudaMalloc((void**)&zNext, slae.memLen * sizeof(fp));
	cudaMalloc((void**)&zPrev, slae.memLen * sizeof(fp));
	cudaMalloc((void**)&Az, slae.memLen * sizeof(fp));
	cudaMalloc((void**)&dev_rPrevScal, sizeof(fp));
	cudaMalloc((void**)&dev_rNextScal, sizeof(fp));
	cudaMalloc((void**)&dev_Az_z, sizeof(fp));
	cudaMalloc((void**)&dev_norm, sizeof(fp));

	cudaMemcpy(xNext, solution, slae.size() * sizeof(fp), cudaMemcpyDeviceToDevice);
	//xNext = solution;

	//std::cout << "cudaStatus " << cudaStatus << "\n";

	cgInit<fp> << <grid, block >> > (slae.matrix, slae.rp, xNext, rNext, zNext, slae.size(), slae.width());

	//iterNum = 0;
	//for (size_t i = 0; i < 354; ++i) {
	do {
		std::swap(xNext, xPrev);
		std::swap(rNext, rPrev);
		std::swap(zNext, zPrev);

		rPrevScal = 0., rNextScal = 0., Az_z = 0.;
		cudaMemcpy(dev_rNextScal, &rNextScal, sizeof(fp), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_rPrevScal, &rPrevScal, sizeof(fp), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Az_z, &Az_z, sizeof(fp), cudaMemcpyHostToDevice);
		norm = 0.;
		cudaMemcpy(dev_norm, &norm, sizeof(fp), cudaMemcpyHostToDevice);
		cg1<fp><<<grid, block>>>(slae.matrix, rPrev, zPrev, Az, dev_rPrevScal, dev_Az_z, slae.size(), slae.width());
		cudaDeviceSynchronize();
		cudaMemcpy(&rPrevScal, dev_rPrevScal, sizeof(fp), cudaMemcpyDeviceToHost);
		cudaMemcpy(&Az_z, dev_Az_z, sizeof(fp), cudaMemcpyDeviceToHost);
		fp alpha = rPrevScal / Az_z;
		cg2mask<fp><<<grid, block>>>(xNext, xPrev, rNext, rPrev, zPrev, Az, alpha, dev_rNextScal, mask);
		//cg2<fp> << <grid, block >> > (xNext, xPrev, rNext, rPrev, zPrev, Az, alpha, dev_rNextScal);
		cudaDeviceSynchronize();
		cudaMemcpy(&rNextScal, dev_rNextScal, sizeof(fp), cudaMemcpyDeviceToHost);
		fp beta = rNextScal / rPrevScal;
		//std::cout << iterNum << "   " << alpha << "    " << beta << "\n";
		cg3<fp><<<grid, block>>>(rNext, zNext, zPrev, beta);
		cudaDeviceSynchronize();

		exitNorm<fp><<<grid, block>>>(xNext, xPrev, dev_norm);
		//exitNormR<double><<<grid, block>>>(rNext, dev_norm);
		cudaDeviceSynchronize();
		cudaMemcpy(&norm, dev_norm, sizeof(fp), cudaMemcpyDeviceToHost);
		//cudaMemcpy(solution.data(), rPrev, slae.size() * sizeof(fp), cudaMemcpyDeviceToHost);
		/*for (int i = 0; i < slae.size(); ++i)
			std::cout << solution[i] << " ";
		std::cout << "\n";*/
		//std::cout << iterNum << "   " << sqrt(norm) << "\n";
		++iterNum;
		//if (iterNum > 500)
			//break;
	} while (exitCondition());

	cudaMemcpy(solution, xNext, slae.size() * sizeof(fp), cudaMemcpyDeviceToDevice);
	cudaFree(xNext); cudaFree(xPrev);
	cudaFree(rNext); cudaFree(rPrev);
	cudaFree(zNext); cudaFree(zPrev);
	cudaFree(Az);
	cudaFree(dev_rNextScal);
	cudaFree(dev_rPrevScal);
	cudaFree(dev_Az_z);
}


void conjugateGradient(CudaSLAE<double>& slae, double* solution, bool* mask, size_t& iterNum, double eps) {
	conjugateGradientBase(slae, solution, mask, iterNum, eps);
}

void conjugateGradient(CudaSLAE<float>& slae, float* solution, bool* mask, size_t& iterNum, float eps) {
	conjugateGradientBase(slae, solution, mask, iterNum, eps);
}