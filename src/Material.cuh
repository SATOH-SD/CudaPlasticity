#pragma once
#include <functional>
#include <nvfunctional>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


class Material {

private:

	double A;
	double m;
	double K_T;

	double (*device_f)(double, double, double, double, double, double) = nullptr;
	double (*host_f)(double, double, double, double, double, double) = nullptr;

public:

	double E;         //модуль упругости
	double nu;        //коэффициент Пуассона
	double h = 1.;    //толщина конечного элемента
	double rho = 1.;  //плотность вещества

	double sigmaT;    //предел текучести

	void setLinearPlast(double E, double sigmaT, double K_T);

	void setPowerPlast(double E, double sigmaT, double A, double m);

	__host__ double f(double eps) const {
		return (*host_f)(eps, E, sigmaT, K_T, A, m);
	}

	__device__ double dev_f(double eps) const {
		return (*device_f)(eps, E, sigmaT, K_T, A, m);
	}

};