#pragma once

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template<int s1, int s2, typename fp>
class StaticMatrix {

public:

	fp data[s1 * s2] = {};

	//Вернуть размер матрицы по первому индексу
	__host__ __device__
	int size1() const {
		return s1;
	}

	//Вернуть размер матрицы по второму индексу
	__host__ __device__
	int size2() const {
		return s2;
	}

	__host__ __device__
	int dataSize() const {
		return s1 * s2;
	}

	__host__ __device__
	inline fp& operator()(int i, int j) {
		return data[i * s2 + j];
	}

	//Константное обращение к элементу матрицы
	__host__ __device__
	inline const fp& operator()(int i, int j) const {
		return data[i * s2 + j];
	}

	//Вывод на экран для отладки
	__host__ __device__
	void print() const;
};


//Вывод на экран для отладки
template<int s1, int s2, typename fp>
__host__ __device__
void StaticMatrix<s1, s2, fp>::print() const {
	printf("\n");
	for (int i = 0; i < s1; ++i) {
		for (int j = 0; j < s2; ++j) {
			//std::cout.width(width);
			printf("%e ", data[i * s2 + j]);
		}
		printf("\n");
	}
	//std::cout << std::endl;
}