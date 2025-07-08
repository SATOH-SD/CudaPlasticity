#pragma once

#include <cmath>

#include "cuda_runtime.h"

struct vec2 {
	double x = 0., y = 0.;

	__device__ __host__
	vec2() = default;
	
	__device__ __host__
	vec2(const vec2& vec)
		: x(vec.x), y(vec.y) {
	};

	__device__ __host__
	vec2(double _x, double _y)
		: x(_x), y(_y) {
	};

	__device__ __host__
	double operator*(const vec2& vec) const {
		return x * vec.x + y * vec.y;
	}

	__device__ __host__
	vec2 operator*(double k) const {
		return { x * k, y * k };
	}

	__device__ __host__
	vec2 operator/(double k) const {
		return { x / k, y / k };
	}

	__device__ __host__
	vec2 operator+(const vec2& vec) const {
		return { x + vec.x, y + vec.y };
	}

	__device__ __host__
	vec2 operator-() const {
		return { -x, -y };
	}

	__device__ __host__
	vec2 operator-(const vec2& vec) const {
		return { x - vec.x, y - vec.y };
	}

	__device__ __host__
	vec2 operator+=(const vec2& vec) {
		x += vec.x;
		y += vec.y;
		return *this;
	}

	__device__ __host__
	vec2 operator-=(const vec2& vec) {
		x -= vec.x;
		y -= vec.y;
		return *this;
	}

	__device__ __host__
	vec2 operator*=(double k) {
		x *= k;
		y *= k;
		return *this;
	}

	__device__ __host__
	vec2 operator/=(double k) {
		x /= k;
		y /= k;
		return *this;
	}

	//Компонента z векторного произведения
	__device__ __host__
	double crossZ(const vec2& vec) {
		return x * vec.y - y * vec.x;
	}

	__device__ __host__
	double norm() const {
		return sqrt(x * x + y * y);
	}

	__device__ __host__
	vec2 normalize() const {
		return *this * (1. / norm());
	}
};

//template<typename fp>
//vec2<fp> operator*(fp k, const vec2<fp>& v) {
//	return v * k;
//}

__device__ __host__
vec2 operator*(double k, const vec2& v);