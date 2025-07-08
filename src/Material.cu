#include "Material.cuh"

__device__ double dev_linearDiagram(double eps, double E, double sigmaT, double K_T, double, double) {
	double epsT = sigmaT / E;
	return eps < epsT ? eps * E : sigmaT + (eps - epsT) * K_T;
}

__device__ double dev_powerDiagram(double eps, double E, double sigmaT, double, double A, double m) {
	double epsT = sigmaT / E;
	return eps < epsT ? eps * E : \
		sigmaT + A * pow(eps - epsT + pow(E / (A * m), 1. / (m - 1.)), m) - A * pow(E / (A * m), m / (m - 1.));
}

__host__ double linearDiagram(double eps, double E, double sigmaT, double K_T, double, double) {
	double epsT = sigmaT / E;
	return eps < epsT ? eps * E : sigmaT + (eps - epsT) * K_T;
}

__host__ double powerDiagram(double eps, double E, double sigmaT, double, double A, double m) {
	double epsT = sigmaT / E;
	return eps < epsT ? eps * E : \
		sigmaT + A * pow(eps - epsT + pow(E / (A * m), 1. / (m - 1.)), m) - A * pow(E / (A * m), m / (m - 1.));
}

__device__ double (*ptr_dev_lin)(double, double, double, double, double, double) = dev_linearDiagram;
__device__ double (*ptr_dev_pow)(double, double, double, double, double, double) = dev_powerDiagram;


void Material::setLinearPlast(double E, double sigmaT, double K_T) {
	Material::E = E;
	Material::sigmaT = sigmaT;
	Material::K_T = K_T;
	host_f = linearDiagram;
	cudaMemcpyFromSymbol(&device_f, ptr_dev_lin, sizeof(ptr_dev_lin));
}

void Material::setPowerPlast(double E, double sigmaT, double A, double m) {
	Material::E = E;
	Material::sigmaT = sigmaT;
	Material::A = A;
	Material::m = m;
	host_f = powerDiagram;
	cudaMemcpyFromSymbol(&device_f, ptr_dev_pow, sizeof(ptr_dev_pow));
}