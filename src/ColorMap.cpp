#include "ColorMap.h"

#include <iostream>

#include "FiniteElement.h"


ColorMap::ColorMap(unsigned nodeCount, unsigned elemTypes, const FiniteElement* fe, const unsigned* elem) {
	init(nodeCount, elemTypes, fe, elem);
}


void ColorMap::init(unsigned nodeCount, unsigned elemTypes, const FiniteElement* fe, const unsigned* elem) {
	ColorMap::elemTypes = elemTypes;

	unsigned elemCount = fe[elemTypes - 1].poolIdx + fe[elemTypes - 1].elemCount;
	elemColor.realloc(elemCount);
	elemColor.setZero(elemCount);

	colorCount = 1;
	elemColor[0] = 1;

	hptr<unsigned char> nodeElemCount(nodeCount);
	const FiniteElement& lastElem = fe[elemTypes - 1];
	for (unsigned i = 0; i < lastElem.memIdx + lastElem.elemCount * lastElem.nodeCount; ++i)
		++nodeElemCount[elem[i]];  // считаем, что в каждом элементе нет повторения узлов

	hptr<unsigned> nodeElemIdx(nodeCount + 1);
	nodeElemIdx[0] = 0;
	for (unsigned i = 0; i < nodeCount; ++i)
		nodeElemIdx[i + 1] = nodeElemIdx[i] + nodeElemCount[i];

	hptr<unsigned> nodeElemMap(nodeElemIdx[nodeCount]);
	nodeElemCount.setZero(nodeCount);

	for (unsigned t = 0; t < elemTypes; ++t) {
		const FiniteElement& type = fe[t];
		const unsigned* locELem = elem + type.memIdx;

		for (unsigned e = 0; e < type.elemCount; ++e) {
			unsigned ge = type.poolIdx + e;
			bool colors[256] = {};
			for (unsigned i = 0; i < type.nodeCount; ++i) {
				unsigned node = locELem[e * type.nodeCount + i];
				for (unsigned j = 0; j < nodeElemCount[node]; ++j)
					colors[elemColor[nodeElemMap[nodeElemIdx[node] + j]]] = true;
				nodeElemMap[nodeElemIdx[node] + nodeElemCount[node]] = ge;
				++nodeElemCount[node];
			}
			unsigned char newColor = 1;
			for (; newColor <= colorCount; ++newColor)
				if (!colors[newColor])
					break;
			if (newColor > colorCount)
				++colorCount;
			elemColor[ge] = newColor;
		}
	}
	//std::cout << "color count: " << colorCount << "\n"; // DEBUG

	colorMapIdx.realloc(elemTypes * (colorCount + 1));
	colorMapIdx.setZero(elemTypes * (colorCount + 1));
	for (unsigned t = 0; t < elemTypes; ++t) {
		const FiniteElement& type = fe[t];
		unsigned* locColorMapIdx = colorMapIdx + (colorCount + 1) * t;
		unsigned char* locElemColor = elemColor + type.poolIdx;
		for (unsigned e = 0; e < type.elemCount; ++e)
			++locColorMapIdx[locElemColor[e]];
		--locColorMapIdx;
		for (unsigned c = 1; c <= colorCount; ++c)
			locColorMapIdx[c + 1] += locColorMapIdx[c];
	}
	colorMap.realloc(lastElem.poolIdx + lastElem.elemCount);
	hptr<unsigned> elemColorCounts(colorCount);
	//fill colorMap...
	for (unsigned t = 0; t < elemTypes; ++t) {
		const FiniteElement& type = fe[t];
		unsigned* locColorMapIdx = colorMapIdx + (colorCount + 1) * t;
		unsigned char* locElemColor = elemColor + type.poolIdx;
		unsigned* locColorMap = colorMap + type.poolIdx;
		elemColorCounts.setZero(colorCount);
		for (unsigned e = 0; e < type.elemCount; ++e) {
			unsigned char color = locElemColor[e] - 1;
			locColorMap[locColorMapIdx[color] + elemColorCounts[color]] = e;
			++elemColorCounts[color];
		}
	}
	// DEBUG
	//for (unsigned t = 0; t < elemTypes; ++t) {
	//	const FiniteElement& type = fe[t];
	//	unsigned* locColorMapIdx = colorMapIdx + (colorCount + 1) * t;
	//	unsigned* locColorMap = colorMap + type.poolIdx;
	//	std::cout << "type " << t << ":\nidx: ";
	//	for (unsigned c = 0; c <= colorCount; ++c)
	//		std::cout << locColorMapIdx[c] << " ";
	//	std::cout << "\n";
	//	/*for (unsigned c = 0; c < colorCount; ++c) {
	//		std::cout << "color " << c + 1 << ":  ";
	//		for (unsigned i = locColorMapIdx[c]; i < locColorMapIdx[c + 1]; ++i)
	//			std::cout << locColorMap[i] << " ";
	//		std::cout << "\n";
	//	}*/
	//}
	// DEBUG
}


void ColorMap::clear() {
	elemTypes = 0;
	colorCount = 0;
	elemColor.free();
	colorMap.free();
	colorMapIdx.free();
}