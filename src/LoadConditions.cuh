#pragma once

#include <vector>
#include <map>
#include <list>
#include <functional>
#include <nvfunctional>

#include "vec2.cuh"


struct ForceCondition {

	nvstd::function<vec2(vec2)> forceFunc;
	vec2 value;

	bool normOrient = false;
	//если сила ориентирована по нормали, то
	//x - нормальная компонента, y - касательная компонента

};

struct DisplacementCondition {

	nvstd::function<vec2(vec2)> displFunc;

	bool fixMidX = false;
	bool fixMidY = false;

};

struct FixedAxis {
	bool vertical = true;
	double coord = 0.;
};

struct FixedBorder {
	bool vertical = true;
};

struct PointCondition {
	vec2 point;
	vec2 value;
};

//Условия нагружения
class LoadConditions {

public:

	bool Rset = false;

	//Объёмная сила
	nvstd::function<vec2(vec2)> R = [] __host__ __device__(vec2) { return vec2(0., 0.); };

	std::map<int, DisplacementCondition> displCond;

	std::map<int, ForceCondition> forceCond;

	std::map<int, FixedBorder> fixedBorder;
	
	std::list<FixedAxis> fixedAxis;
	
	std::list<PointCondition> forcePoint;
	
	std::list<PointCondition> displPoint;


	//Задать постоянное перемещение
	void setDisplacement(size_t borderId, vec2 u, bool fixOnlyMiddleX = false, bool fixOnlyMiddleY = false) {
		auto& displ = displCond[borderId];
		//displ.value = u;
		displ.displFunc = [=] __host__ __device__ (vec2) { return u; };
		displ.fixMidX = fixOnlyMiddleX;
		displ.fixMidY = fixOnlyMiddleY;
	}

	//Задать переменное перемещение
	void setDisplacement(size_t borderId, nvstd::function<vec2(vec2)> displFunc) {
		auto& displ = displCond[borderId];
		displ.displFunc = displFunc;
		displ.fixMidX = false;
		displ.fixMidY = false;
	}

	//Задать постоянную силу
	void setForce(size_t borderId, vec2 f, bool normalOriented = false) {
		auto& force = forceCond[borderId];
		force.forceFunc = [=] __host__ __device__ (vec2) { return f; };
		force.value = f;
		force.normOrient = normalOriented;
	}

	//Задать переменную силу
	void setForce(size_t borderId, nvstd::function<vec2(vec2)> forceFunc, bool normalOriented = false) {
		auto& force = forceCond[borderId];
		force.forceFunc = forceFunc;
		force.normOrient = normalOriented;
	}

	//Зафиксировать вертикальную ось
	void fixVertAxis(double x) {
		FixedAxis sa;
		sa.coord = x;
		sa.vertical = true;
		fixedAxis.push_back(sa);
	}

	//Зафиксировать горизонтальную ось
	void fixHorAxis(double y) {
		FixedAxis sa;
		sa.coord = y;
		sa.vertical = false;
		fixedAxis.push_back(sa);
	}

	//Зафиксировать границу
	void fixBorder(size_t borderId) {
		auto& displ = displCond[borderId];
		displ.displFunc = [](vec2) { return vec2(0., 0.); };
		displ.fixMidX = false;
		displ.fixMidY = false;
	}

	//Зафиксировать перемещение границы вдоль вертикали
	void fixBorderVert(size_t borderId) {
		auto& border = fixedBorder[borderId];
		border.vertical = true;
	}

	//Зафиксировать перемещение границы вдоль горизонтали
	void fixBorderHor(size_t borderId) {
		auto& border = fixedBorder[borderId];
		border.vertical = false;
	}

	//Зафиксировать точку
	void fixPoint(vec2 point) {
		PointCondition fixed;
		fixed.point = point;
		fixed.value = vec2(0., 0.);
		displPoint.push_back(fixed);
	}

	//Задать перемещение в точке
	void setDisplPoint(vec2 point, vec2 displ) {
		PointCondition displaced;
		displaced.point = point;
		displaced.value = displ;
		displPoint.push_back(displaced);
	}

	//Задать силу в точке
	void setForcePoint(vec2 point, vec2 force) {
		PointCondition forced;
		forced.point = point;
		forced.value = force;
		forcePoint.push_back(forced);
	}

	//Задать поле гравитационных сил
	void setGravity(double g = 9.81, vec2 direction = vec2(0., -1.)) {
		vec2 force = g * direction.normalize();
		R = [=](vec2) { return force; };
		Rset = true;
	}

	//Задать поле центробежных сил
	void setRotation(double omega, vec2 rotationAxis = vec2(0., 0.)) {
		double omega2 = omega * omega;
		R = [=](vec2 r) { return omega2 * (r - rotationAxis); };
		Rset = true;
	}

	//Задать поле объёмных сил
	void setVolumeForce(nvstd::function<vec2(vec2)> R) {
		LoadConditions::R = R;
		Rset = true;
	}

};