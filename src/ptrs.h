#pragma once

#include <cstring>

#include "cuda_runtime.h"


// Host selfdestructive pointer
// (не хранит информацию о размере выделенной памяти!)
template<typename T>
class hptr {

private:

	T* ptr = nullptr;

public:

	hptr() = default;

	hptr(const hptr<T>&) = delete;

	hptr(hptr<T>&& p) {
		ptr = p.ptr;
		p.ptr = nullptr;
	}

	hptr(unsigned size) {
		ptr = new T[size];
	}

	~hptr() {
		delete[] ptr;
		ptr = nullptr;
	}

	hptr<T>& operator=(const hptr<T>&) = delete;

	hptr<T>& operator=(hptr<T>&& p) {
		if (p != *this) {
			delete[] ptr;
			ptr = p.ptr;
			p.ptr = nullptr;
		}
		return *this;
	}

	operator T*() const {
		return ptr;
	}

	T* raw() {
		return ptr;
	}

	const T* raw() const {
		return ptr;
	}

	T& operator[](unsigned i) {
		return ptr[i];
	}

	const T& operator[](unsigned i) const {
		return ptr[i];
	}

	bool isEmpty() const {
		return ptr == nullptr;
	}

	void malloc(unsigned size) {
		ptr = new T[size];
	}

	// Перевыделить память без сохранения данных
	void realloc(unsigned size) {
		delete[] ptr;
		ptr = new T[size];
	}

	// Изменить размер выделенной памяти с сохранением данных
	void expand(unsigned oldSize, unsigned newSize) {
		T* newptr = std::malloc(newSize * sizeof(T));
		memcpy(newptr, ptr, std::min(oldSize, newSize) * sizeof(T));
		std::swap(ptr, newptr);
		std::free(newptr);
	}

	void swap(hptr<T>& ptr) {
		std::swap(hptr<T>::ptr, ptr.ptr);
	}

	void free() {
		delete[] ptr;
		ptr = nullptr;
	}

	void setZero(unsigned size) {
		memset(ptr, 0, size * sizeof(T));
	}
};


// Device selfdestructive pointer
// (не хранит информацию о размере выделенной памяти!)
template<typename T>
class dptr {

private:

	T* ptr = nullptr;

public:

	dptr() = default;

	dptr(const dptr<T>&) = delete;

	dptr(dptr<T>&& p) {
		ptr = p.ptr;
		p.ptr = nullptr;
	}

	dptr(unsigned size) {
		cudaMalloc(&ptr, size * sizeof(T));
	}

	~dptr() {
		cudaFree(ptr);
		ptr = nullptr;
	}

	dptr<T>& operator=(const dptr<T>&) = delete;

	dptr<T>& operator=(dptr<T>&& p) {
		if (p != *this) {
			cudaFree(ptr);
			ptr = p.ptr;
			p.ptr = nullptr;
		}
		return *this;
	}

	operator T*() const {
		return ptr;
	}

	T* raw() {
		return ptr;
	}

	const T* raw() const {
		return ptr;
	}

	T getItem(unsigned i) const {
		T item;
		cudaMemcpy(&item, ptr + i, sizeof(T), cudaMemcpyDeviceToHost);
		return item;
	}

	void setItem(unsigned i, const T& item) {
		cudaMemcpy(ptr + i, &item, sizeof(T), cudaMemcpyHostToDevice);
	}

	bool isEmpty() const {
		return ptr == nullptr;
	}

	void malloc(unsigned size) {
		cudaMalloc(&ptr, size * sizeof(T));
	}

	// Изменить размер выделенной памяти с сохранением данных
	void expand(unsigned oldSize, unsigned newSize);

	// Перевыделить память без сохранения данных
	void realloc(unsigned size) {
		cudaFree(ptr);
		cudaMalloc(&ptr, size * sizeof(T));
	}

	void swap(dptr<T>& ptr) {
		std::swap(dptr<T>::ptr, ptr);
	}

	void free() {
		cudaFree(ptr);
		ptr = nullptr;
	}

	void setZero(unsigned size) {
		cudaMemset(ptr, 0, size * sizeof(T));
	}

};


// Page-pinned host selfdestructive pointer
// (не хранит информацию о размере выделенной памяти!)
template<typename T>
class pptr {

private:

	T* ptr = nullptr;

public:

	pptr() = default;

	//...

};


template<typename T>
inline void hostToDevice(dptr<T>& dst, const hptr<T>& src, unsigned size) {
	cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
inline void deviceToHost(hptr<T>& dst, const dptr<T>& src, unsigned size) {
	cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToHost);
}