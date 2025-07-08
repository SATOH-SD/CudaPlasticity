#include "vec2.cuh"

__device__ __host__
vec2 operator*(double k, const vec2& v) {
	return v * k;
}