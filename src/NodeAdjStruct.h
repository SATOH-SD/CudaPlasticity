#pragma once

#include "ptrs.h"


class FiniteElement;


class NodeAdjStruct {
	
public:

	unsigned N = 0;

	hptr<unsigned> adj;

	hptr<unsigned> adjIdx;

	NodeAdjStruct() = default;

	NodeAdjStruct(const NodeAdjStruct& nodeAdjStruct);

	NodeAdjStruct(unsigned nodeCount, unsigned elemTypes, const FiniteElement* elemInfo, const unsigned* elem);

	~NodeAdjStruct() = default;

	void init(unsigned nodeCount, unsigned elemTypes, const FiniteElement* elemInfo, const unsigned* elem);

	void print() const;

	void clear();

	bool NodeAdjStruct::empty() const {
		return N == 0;
	}

};