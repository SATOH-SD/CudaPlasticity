#pragma once
#include <math.h>

#include "cuda_runtime.h"

//Схема интегрирования по Гауссу
//template<int size>
struct GaussScheme {

	double point[5] = {};
	double coef[5] = {};

	int size = 1;

	//template<int size>
	__host__ __device__
	constexpr GaussScheme(int n) {
		size = n;
		switch (size) {
		case 1:
			point[0] = 0.;
			coef[0] = 2.;
			break;
		case 2:
			point[0] = -1. / sqrt(3.);
			point[1] = 1. / sqrt(3.);
			coef[0] = coef[1] = 1.;
			break;
		case 3:
			point[0] = -sqrt(0.6);
			point[1] = 0.;
			point[2] = sqrt(0.6);
			coef[0] = coef[2] = 5. / 9.;
			coef[1] = 8. / 9.;
			break;
		case 4:
			point[0] = -sqrt(3. / 7. + 2. / 7. * sqrt(1.2));
			point[1] = -sqrt(3. / 7. - 2. / 7. * sqrt(1.2));
			point[2] = sqrt(3. / 7. - 2. / 7. * sqrt(1.2));
			point[3] = sqrt(3. / 7. - 2. / 7. * sqrt(1.2));
			coef[0] = coef[3] = (18. - sqrt(30.)) / 36.;
			coef[1] = coef[2] = (18. + sqrt(30.)) / 36.;
			break;
		case 5:
			point[0] = -sqrt(5. + 2. * sqrt(10. / 7.)) / 3.;
			point[1] = -sqrt(5. - 2. * sqrt(10. / 7.)) / 3.;
			point[2] = 0.;
			point[3] = sqrt(5. - 2. * sqrt(10. / 7.)) / 3.;
			point[4] = sqrt(5. + 2. * sqrt(10. / 7.)) / 3.;
			coef[0] = coef[4] = (322. - 13. * sqrt(70.)) / 900.;
			coef[1] = coef[3] = (322. + 13. * sqrt(70.)) / 900.;
			coef[2] = 128. / 225.;
			break;
		}
	}
};



//__constant__ double GS3_point[3] = { -sqrt(0.6), 0., sqrt(0.6) };
//__constant__ double GS3_coef[3] = { 5. / 9., 8. / 9., 5. / 9. };