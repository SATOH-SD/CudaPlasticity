#pragma once

#include "ptrs.h"


class FiniteElement;


class ColorMap {

public:

	unsigned elemTypes = 0;

	unsigned colorCount = 0;

	hptr<unsigned char> elemColor;

	hptr<unsigned> colorMap;

	hptr<unsigned> colorMapIdx;

	ColorMap() = default;

	ColorMap(unsigned nodeCount, unsigned elemTypes, const FiniteElement* fe, const unsigned* elem);

	void init(unsigned nodeCount, unsigned elemTypes, const FiniteElement* fe, const unsigned* elem);

	void clear();

	bool empty() const {
		return colorCount == 0;
	}

};