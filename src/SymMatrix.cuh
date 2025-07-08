#pragma once

#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template<int s, typename fp>
class SymMatrix {

public:

	fp data[(1 + s) * s / 2] = {};

	__host__ __device__
	int size() const {
		return s;
	}

	__host__ __device__
	int dataSize() const {
		return (1 + s) * s / 2;
	}

	__host__ __device__
	inline fp& operator()(int i, int j) {
		return data[i * (i + 1) / 2 + j];
	}

	//Константное обращение к элементу матрицы
	__host__ __device__
	inline const fp& operator()(int i, int j) const {
		return data[i * (i + 1) / 2 + j];
	}

	//Вывод на экран для отладки
	__host__ __device__
	void print() const;

	void sPrint(int width = 0) const;

};

//Вывод на экран для отладки
template<int s, typename fp>
__host__ __device__
void SymMatrix<s, fp>::print() const {
	printf("\n");
	for (int i = 0; i < s; ++i) {
		for (int j = 0; j <= i; ++j) {
			//std::cout.width(width);
			printf("%e ", data[i * (i + 1) / 2 + j]);//wrong
		}
		printf("\n");
	}
}

template<int s, typename fp>
void SymMatrix<s, fp>::sPrint(int width) const {
	std::cout << "\n";
	for (int i = 0; i < s; ++i) {
		for (int j = 0; j <= i; ++j) {
			std::cout.width(width);
			std::cout << data[i * (i + 1) / 2 + j] << " ";
		}
		std::cout << "\n";
	}
}