#include <iostream>
#include <vector>
#include <thread>

#include "PlasticitySolver.cuh"


const double E_304L = 193e9;
const double s_T_304L = 170e6;
const double nu_304L = 0.29;


//Тест решения упругости в прямоугольнике
void solverTest() {

	Mesh mesh;
	mesh.useCuda = false;
	mesh.genRectangle(-1., 1., -1., 1., 4, 4);

	LoadConditions lc;
	//lc.setDisplacement(2, vec2(0, 0));
	lc.setDisplacement(2, vec2(0, 0.), false, true);
	lc.setForce(0, vec2(1, 0));
	//lc.setForce(2, vec2(-1e6, 0));
	//lc.setDisplacement(0, vec2(1e-4, 3e-5));
	//lc.fixVertAxis(0.);
	//lc.fixHorAxis(0.);

	Material m;
	m.setLinearPlast(E_304L * 1e-6, s_T_304L, E_304L * 1e-6);
	m.E = E_304L * 1e-6;//220e9;
	m.nu = nu_304L * 1e-6;//0.3;
	m.h = 1.;

	PlasticitySolver ps(m, mesh, lc);
	ps.solveElast();
	ps.saveAsVtk("../data/solverTest.vtk");
}

//Тест решения упругости в пластине с отверстием
void holePlateTest() {
	Mesh mesh;
	//mesh.useCuda = false;
	mesh.genRectWithHole(-3., 3., -2., 2., 0.5, 8 * 48, 8 * 32);

	LoadConditions lc;
	//lc.setDisplacement(3, vec2(0, 0), false, true);
	lc.setForce(1, vec2(1e6, 0));
	lc.setForce(3, vec2(-1e6, 0));
	//lc.setForce(4, { 0, 1e6 }, true);
	lc.fixVertAxis(0.);
	lc.fixHorAxis(0.);

	Material m;
	m.E = E_304L;//220e9;
	m.nu = nu_304L;//0.3;
	m.h = 1.;

	PlasticitySolver ps(m, mesh, lc);
	ps.polarCoord = true;
	ps.solveElast();
	ps.saveDisplsToFile("../data/holePlateTest.txt");
	ps.saveAsVtk("../data/holePlate.vtk");

	double S = 1e6, a = 0.5;
	auto s_r = [&](vec2 r) { return 0.5 * S * (1. - a * a / (r.x * r.x)) + 0.5 * S * (1. + 3. * pow(a, 4.) / pow(r.x, 4.) - 4. * a * a / (r.x * r.x)) * cos(2. * r.y); };
	auto s_phi = [&](vec2 r) { return 0.5 * S * (1. + a * a / (r.x * r.x)) - 0.5 * S * (1. + 3. * pow(a, 4.) / pow(r.x, 4.)) * cos(2. * r.y); };
	auto tau = [&](vec2 r) { return -0.5 * S * (1. - 3. * pow(a, 4.) / pow(r.x, 4.) + 2. * a * a / (r.x * r.x)) * sin(2. * r.y); };

	std::cout << "Error: ";
	ps.polarSigmaError(s_r, s_phi, tau, 1.);
}

//Тест в кольце
void testRing() {
	Mesh mesh;
	mesh.genRing(0.1, 0.2, 40, 5);
	//mesh.print();
	//mesh.printAnalysis();
	//mesh.saveAsVtk("../data/ring.vtk");
	//mesh.useCuda = false;

	double U = 0.00001;
	double phi = 1e-4;
	LoadConditions lc;
	//lc.setDisplacement(0, { 0, 0 });
	//lc.setDisplacement(1, [=](vec2 r) { return vec2(U, 0.) + vec2(-r.y, r.x).normalize() * phi * 0.2; });

	lc.setForce(0, [=](vec2 r) { return vec2(1e6 * fabs(r.x), 0); });
	//lc.setForce(0, vec2(1e6, 0));
	//lc.setForce(3, vec2(-1e6, 0));
	//lc.setForce(4, { 0, 1e6 }, true);
	lc.fixVertAxis(0.);
	lc.fixHorAxis(0.);

	Material m;
	m.E = 1.1e10;
	m.nu = 0.3;
	m.h = 0.1;

	PlasticitySolver ps(m, mesh, lc);
	ps.polarCoord = true;
	//ps.solveElast();
	ps.solve();
	//ps.saveDisplsToFile("../data/ringTest.txt");
	ps.saveAsVtk("../data/ring.vtk");
}

//Тест сходимости по энергии деформации
void energyConvTest() {
	LoadConditions lc;
	double S = 170e6;
	//lc.setDisplacement(3, vec2(0, 0), false, true);
	lc.setForce(1, vec2(S, 0));
	lc.setForce(3, vec2(-S, 0));
	//lc.setForce(4, { 0, 1e6 }, true);
	lc.fixVertAxis(0.);
	lc.fixHorAxis(0.);

	Material m;
	//m.setLinearPlast(E_304L, s_T_304L, E_304L * 0.1);
	m.setPowerPlast(E_304L, s_T_304L, E_304L * 0.01, 0.015);
	//m.E = E_304L;//220e9;
	m.nu = nu_304L;//0.3;
	m.h = 1.;

	const int base[2] = { 9, 6 };
	const int scales[9] = { 1, 2, 4, 8, 16 };
	//const int scales[9] = { 3, 6, 12, 24, 48 };
	int elemCount[9] = { 1280, 2880, 5120, 9460, 20480, 41860, 81920, 161100, 327680 };

	const double q = 0.5;
	size_t segments1 = 8, gridNum = 5;
	size_t segments = segments1;
	std::vector<double> sizes(gridNum), energy(gridNum);
	std::vector<int> elems(gridNum);
	
	Mesh mesh;
	//mesh.useCuda = false;

	for (size_t i = 0; i < gridNum; ++i) {
		
		//mesh.genRectWithHole(-3., 3., -2., 2., 0.5, segments, segments, 1);
		mesh.genRectWithHole(-3., 3., -2., 2., 0.5, base[0] * scales[i], base[1] * scales[i], 2);
		//mesh.genRectangle(-2., 2., -2., 2., segments, segments);
		sizes[i] = mesh.avrElemSize();
		elems[i] = mesh.elemCount();

		PlasticitySolver ps(m, mesh, lc);
		//ps.floatBoost = true;
		ps.solve();
		energy[i] = ps.deformEnergy();

		segments *= (int)round(1. / q);
	}
	std::cout << "\n\nelem size    |    energy\n";
	for (size_t i = 0; i < gridNum; ++i)
		std::cout << sizes[i] << "           " << energy[i] << "\n";

	std::ofstream file("../data/aitken.txt", std::ios_base::out);
	double f = energy[0] + (energy[0] - energy[1]) * (energy[0] - energy[1]) / \
		(2. * energy[1] - energy[0] - energy[2]);  //формула для уточнённого значения
	file << "elems        energy       errEstim            p\n";
	file << elems[0] << "  " << energy[1] << "   " << fabs(f - energy[0]) / energy[0] << "\n";
	file << elems[1] << "  " << energy[1] << "   " << fabs(f - energy[1]) / energy[1] << "\n";

	for (size_t j = 2; j < gridNum; ++j) {
		f = energy[j - 2] + (energy[j - 2] - energy[j - 1]) * (energy[j - 2] - energy[j - 1]) / \
			(2. * energy[j - 1] - energy[j - 2] - energy[j]);
		file << elems[j] << "  " << energy[j] << "    " << fabs(f - energy[j]) / energy[j] << "   " \
			<< log(fabs((energy[j] - energy[j - 1]) / (energy[j - 1] - energy[j - 2]))) / log(sizes[j] / sizes[j - 1]) << "\n";
		std::cout << "f = " << f << "\n";
	}
	file.close();
}

//Тест сходимости для задачи Кирша
void kirschConvTest() {
	Material m;
	m.E = E_304L;//220e9;
	m.nu = nu_304L;//0.3;
	m.h = 1.;
	double a = 0.5,  //радиус отверсия в пластине
		S = 1e6;   //сила, приложенная к сторонам

	LoadConditions lc;
	lc.setForce(1, vec2(S, 0));
	lc.setForce(3, vec2(-S, 0));
	lc.fixVertAxis(0.);
	lc.fixHorAxis(0.);

	auto u_r_ZeroPi = [&](double r) { return (-pow(a, 4.) * (1 + m.nu) + a * a * (5. + m.nu) * r * r + 2. * pow(r, 4.)) * S * 0.5 / (m.E * r * r * r); };
	auto u_r_HalfPi = [&](double r) { return (pow(a, 4.) * (1 + m.nu) + a * a * (-3. + m.nu) * r * r - 2. * m.nu * pow(r, 4.)) * S * 0.5 / (m.E * r * r * r); };

	auto s_r = [&](vec2 r) { return 0.5 * S * (1. - a * a / (r.x * r.x)) + 0.5 * S * (1. + 3. * pow(a, 4.) / pow(r.x, 4.) - 4. * a * a / (r.x * r.x)) * cos(2. * r.y); };
	auto s_phi = [&](vec2 r) { return 0.5 * S * (1. + a * a / (r.x * r.x)) - 0.5 * S * (1. + 3. * pow(a, 4.) / pow(r.x, 4.)) * cos(2. * r.y); };
	auto tau = [&](vec2 r) { return -0.5 * S * (1. - 3. * pow(a, 4.) / pow(r.x, 4.) + 2. * a * a / (r.x * r.x)) * sin(2. * r.y); };

	const double segments1 = 4, q = 0.5;
	std::ofstream file("../data/conv.txt", std::ios_base::out);

	size_t gridNum = 5;
	std::vector<double> errors(gridNum), sizes(gridNum);

	file << "steps         error        delta        log_q(delta)";
	double segments = segments1;
	for (size_t i = 0; i < gridNum; ++i) {
		Mesh mesh;
		mesh.genRectWithHole(-3., 3., -2., 2., a, segments, segments);
		sizes[i] = mesh.avrElemSize();

		PlasticitySolver ps(m, mesh, lc);
		ps.solveElast();
		errors[i] = ps.polarSigmaError(s_r, s_phi, tau, 2.);

		//errors[i] = fabs(S - ps.getSxx0()) / S;
		//errors[i] = ps.absErrorDisplZeroPiLine(u_r_ZeroPi);
		//errors[i] = ps.absErrorDisplHalfPiLine(u_r_HalfPi);
		file << "\nh = " << sizes[i] << "     " << errors[i];
		if (i > 0) {
			double delta = errors[i] / errors[i - 1];
			double p = log(delta) / log(q);
			std::cout << p << "\n";
			file << "     " << delta << "    " << p;
		}
		segments *= (int)round(1. / q);
	}
	file.close();
}

void meshTransferTest() {
	Mesh mesh;
	//mesh.useCuda = false;
	mesh.genRectWithHole(-3., 3., -2., 2., 0.5, 48, 32);
	//mesh.genRectWithHole(-3., 3., -2., 2., 0.5, 2, 2);
	//mesh.genRectangle(-2, 2, -2, 2, 2, 2);
	mesh.printAnalysis();
	mesh.ramSaved = false;
	//mesh.meshToRAM();
	//mesh.meshToRAM();

	Mesh mesh2(mesh);
	std::clog << "mesh copied\n";
	mesh2.meshToRAM();
	mesh2.meshToRAM();
	std::clog << "mesh moved\n";
	mesh2.saveAsVtk("../data/newMesh.vtk");
	std::clog << "mesh saved\n";
}

//Тест пластичности
void plasticityTest() {

	Mesh mesh;
	//mesh.useCuda = false;
	mesh.genRectWithHole(-3., 3., -2., 2., 0.5, 48, 32, 1);
	//mesh.genRectWithHole(-3., 3., -2., 2., 0.5, 96 * 2, 64 * 2, 1);
	//mesh.genRectWithHole(-3., 3., -2., 2., 0.5, 144, 96, 2);
	//mesh.genRectWithHole(-3., 3., -2., 2., 0.5, 6 * 4, 4 * 4, 1);
	//mesh.genRectangle(-2, 2, -2, 2, 2, 2);
	//mesh.printAnalysis();
	//mesh.renumByDirection();
	std::cout << "max diff " << mesh.findMaxIndexDiff() << "\n\n";
	//mesh.saveAsVtk("../data/renumMesh.vtk");

	double S = 170; //растягивающая сила
	LoadConditions lc;
	lc.setForce(1, vec2(S, 0.));
	lc.setForce(3, vec2(-S, 0.));
	//lc.setDisplacement(3, vec2(0, 0));
	lc.fixVertAxis(0.);
	lc.fixHorAxis(0.);
	
	Material m;
	//m.nu = 0.3;
	m.h = 1.;
	//m.setLinearPlast(193e9, 170e6, 110e9);
	//m.setLinearPlast(220e9, 170e6, 22e9);
	//m.setLinearPlast(E_304L * 1e-6, s_T_304L * 1e-6, E_304L * 0.1 * 1e-6);
	m.setPowerPlast(E_304L * 1e-6, s_T_304L * 1e-6, E_304L * 0.01 * 1e-6, 0.015);
	//m.E = E_304L;//220e9;
	m.nu = nu_304L;//0.3;

	//PlasticitySolver ps(m, grid, lc);
	PlasticitySolver ps(m, mesh, lc);
	//ps.setPlainCondition(strain);
	//ps.floatBoost = true;
	//ps.iterOutput = false;
	//ps.solve();
	//ps.floatBoost = false;
	ps.solve();
	//ps.polarCoord = true;
	ps.saveAsVtk("../data/holePlatePlastPow.vtk");
	//std::cout << "saved\n";
	//std::cin.get();
}

void test1() {
	Mesh mesh;
	mesh.genRectWithHole(-3., 3., -2., 2., 0.5, 96, 64);
	
	double S = 170.; //растягивающая сила
	LoadConditions lc;
	lc.fixBorder(3);
	lc.setForce(1, vec2(S, 0.));

	Material m;
	m.setLinearPlast(193e3, 170, 193e2);
	m.nu = 0.29;

	PlasticitySolver ps(m, mesh, lc);
	ps.solve();
	ps.saveAsVtk("../data/test1.vtk");
}

void speedTestAlpha() {
	double S = 170e6; //растягивающая сила
	LoadConditions lc;
	lc.setForce(1, vec2(S, 0.));
	lc.setForce(3, vec2(-S, 0.));
	lc.fixVertAxis(0.);
	lc.fixHorAxis(0.);

	Material m;
	//m.nu = 0.3;
	m.h = 1.;
	//m.setLinearPlast(193e9, 170e6, 110e9);
	//m.setLinearPlast(220e9, 170e6, 22e9);
	m.setLinearPlast(E_304L, s_T_304L, E_304L * 0.1);
	//m.E = E_304L;//220e9;
	m.nu = nu_304L;//0.3;

	const int tests = 4;
	double cpuTime[tests] = {};
	double gpuTime[tests] = {};
	int elemCount[tests] = {};
	Mesh mesh;
	bool fullTest = mesh.useCuda;
	//GPU
	if (fullTest)
		for (int i = 0; i < tests; ++i) {
			std::cout << "\n\nChilling...";
			std::this_thread::sleep_for(std::chrono::milliseconds(1000 * (i + 1)));
			int scale = 1 << i;
			std::cout << "\rGPU test " << double(scale * scale) / 4. << "x\n";
			mesh.genRectWithHole(-3., 3., -2., 2., 0.5, 24 * scale, 16 * scale);
			//mesh.meshToGPU();

			PlasticitySolver ps(m, mesh, lc);
			switch (i) {
			case 0: case 1: ps.iterOutput = false; break;
			default: ps.iterOutput = true; break;
			}
			gpuTime[i] = ps.solve();
		}

	//CPU
	mesh.useCuda = false;
	for (int i = 0; i < tests; ++i) {
		std::cout << "\n\nChilling...";
		std::this_thread::sleep_for(std::chrono::milliseconds(1000 * (i + 1)));
		int scale = 1 << i;
		std::cout << "\rCPU test " << double(scale * scale) / 4. << "x\n";
		mesh.genRectWithHole(-3., 3., -2., 2., 0.5, 24 * scale, 16 * scale);
		elemCount[i] = mesh.elemCount();

		PlasticitySolver ps(m, mesh, lc);
		switch (i) {
		case 0: case 1: ps.iterOutput = false; break;
		default: ps.iterOutput = true; break;
		}
		cpuTime[i] = ps.solve();
	}
	std::cout << "\n\n\nResults\n\nCPU only\n Elements        Time\n  ";
	for (int i = 0; i < tests; ++i)
		std::cout << elemCount[i] << "          " << cpuTime[i] << "\n  ";
	if (fullTest) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		std::cout << "\n" << deviceProp.name << " " \
			<< round((double)deviceProp.totalGlobalMem / 1'048'576.) << " MB\n";
		std::cout << " Elements        Time\n  ";
		for (int i = 0; i < tests; ++i)
			std::cout << elemCount[i] << "          " << gpuTime[i] << "\n  ";
	}
	std::cout << std::endl;

#ifdef _WIN64
	std::cout << "Press ENTER to exit..." << std::flush;
	std::cin.get();
#endif
}

#include <array>

std::string getProcessorName() {
#ifdef _WIN64
#include <intrin.h>

	std::array<int, 4> integerBuffer = {};
	constexpr size_t sizeofIntegerBuffer = sizeof(int) * integerBuffer.size();

	std::array<char, 64> charBuffer = {};

	// The information you wanna query __cpuid for.
	// https://learn.microsoft.com/en-us/cpp/intrinsics/cpuid-cpuidex?view=vs-2019
	constexpr std::array<int, 3> functionIds = {
		// Manufacturer
		//  EX: "Intel(R) Core(TM"
		0x8000'0002,
		// Model
		//  EX: ") i7-8700K CPU @"
		0x8000'0003,
		// Clockspeed
		//  EX: " 3.70GHz"
		0x8000'0004
	};
	std::string cpu;
	for (int id : functionIds) {
		// Get the data for the current ID.
		__cpuid(integerBuffer.data(), id);

		// Copy the raw data from the integer buffer into the character buffer
		std::memcpy(charBuffer.data(), integerBuffer.data(), sizeofIntegerBuffer);

		// Copy that data into a std::string
		cpu += std::string(charBuffer.data());
	}
	return cpu;
#else
//#include <cpuid.h>
//
//	char CPUBrandString[0x40];
//	unsigned int CPUInfo[4] = { 0,0,0,0 };
//
//	__cpuid(0x80000000, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
//	unsigned int nExIds = CPUInfo[0];
//
//	memset(CPUBrandString, 0, sizeof(CPUBrandString));
//
//	for (unsigned int i = 0x80000000; i <= nExIds; ++i)
//	{
//		__cpuid(i, CPUInfo[0], CPUInfo[1], CPUInfo[2], CPUInfo[3]);
//
//		if (i == 0x80000002)
//			memcpy(CPUBrandString, CPUInfo, sizeof(CPUInfo));
//		else if (i == 0x80000003)
//			memcpy(CPUBrandString + 16, CPUInfo, sizeof(CPUInfo));
//		else if (i == 0x80000004)
//			memcpy(CPUBrandString + 32, CPUInfo, sizeof(CPUInfo));
//	}
	//std::string cpu = CPUBrandString;
	std::string cpu = "";
	return cpu;
#endif
}

void speedTest2() {
	double S = 170e6; //растягивающая сила
	LoadConditions lc;
	lc.setForce(1, vec2(S, 0.));
	lc.setForce(3, vec2(-S, 0.));
	lc.fixVertAxis(0.);
	lc.fixHorAxis(0.);

	double Se = 1e6; //растягивающая сила
	LoadConditions lce;
	lce.setForce(1, vec2(Se, 0.));
	lce.setForce(3, vec2(-Se, 0.));
	lce.fixVertAxis(0.);
	lce.fixHorAxis(0.);

	Material m;
	m.h = 1.;
	m.setLinearPlast(E_304L, s_T_304L, E_304L * 0.1);
	m.nu = nu_304L;

	const int base[2] = { 6, 4 };
	const int scales[9] = { 4, 6, 8, 11, 16, 23, 32, 45, 64 };
	int elemCount[9] = { 1280, 2880, 5120, 9460, 20480, 41860, 81920, 161100, 327680 };
	//1280, 2880, 5120, 9460, 20480, 41860, 161100, 327680

	//const int elastCpu1 = 8, elastGpu1 = 9, elastCpu2 = 6, elastGpu2 = 6, \
		plastCpu1 = 8, plastGpu1 = 9, plastCpu2 = 6, plastGpu2 = 6, \
		plastGpuF1 = 9, plastGpuF2 = 6; //default
	const int elastCpu1 = 8, elastGpu1 = 8, elastCpu2 = 6, elastGpu2 = 6, \
		plastCpu1 = 8, plastGpu1 = 8, plastCpu2 = 6, plastGpu2 = 6, \
		plastGpuF1 = 8, plastGpuF2 = 6; //8 GB
	//const int elastCpu1 = 9, elastGpu1 = 9, elastCpu2 = 6, elastGpu2 = 6, \
		plastCpu1 = 9, plastGpu1 = 9, plastCpu2 = 6, plastGpu2 = 6, \
		plastGpuF1 = 9, plastGpuF2 = 6; //12 GB
	double elastCpu1_time[elastCpu1] = {},
		elastGpu1_time[elastGpu1] = {},
		elastCpu2_time[elastCpu2] = {},
		elastGpu2_time[elastGpu2] = {},
		plastCpu1_time[plastCpu1] = {},
		plastGpu1_time[plastGpu1] = {},
		plastCpu2_time[plastCpu2] = {},
		plastGpu2_time[plastGpu2] = {},
		plastGpuF1_time[plastGpuF1] = {},
		plastGpuF2_time[plastGpuF2] = {};

	Mesh mesh;
	bool fullTest = 1 * mesh.useCuda;
#ifdef _WIN64
	std::cout << "CPU    Elasticity       Plasticity\n";
	std::cout << "1st    " << elemCount[elastCpu1 - 1] << "              " << elemCount[plastCpu1 - 1] << "\n";
	std::cout << "2nd    " << elemCount[elastCpu2 - 1] << "              " << elemCount[plastCpu2 - 1] << "\n\n";
	if (mesh.useCuda) {
		std::cout << "GPU    Elasticity       Plasticity         Plasticity (float boost)\n";
		std::cout << "1st    " << elemCount[elastGpu1 - 1] << "              " << elemCount[plastGpu1 - 1] << "              " << elemCount[plastGpuF1 - 1] << "\n";
		std::cout << "2nd    " << elemCount[elastGpu2 - 1] << "              " << elemCount[plastGpu2 - 1] << "              " << elemCount[plastGpuF2 - 1] << "\n\n";
	}
	std::cout << "Press ENTER to confirm..." << std::flush;
	std::cin.get();
#endif

	//GPU
	if (fullTest) {
		std::cout << "\n\n\n\nTest GPU elasticity (1st order)\n\n";
		for (int i = 0; i < elastGpu1; ++i) {
			std::cout << "\n\nChilling...";
			//std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			std::cout << "\rGPU elasticity test " << base[0] * scales[i] << "x" << base[1] * scales[i] << " (1st order)\n";
			mesh.genRectWithHole(-3., 3., -2., 2., 0.5, base[0] * scales[i], base[1] * scales[i], 1);
			PlasticitySolver ps(m, mesh, lc);
			elastGpu1_time[i] = ps.solveElast();
			std::cout << std::flush;
		}
		std::cout << "\n\n\n\nTest GPU elasticity (2nd order)\n\n";
		for (int i = 0; i < elastGpu2; ++i) {
			std::cout << "\n\nChilling...";
			//std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			std::cout << "\rGPU elasticity test " << base[0] * scales[i] << "x" << base[1] * scales[i] << " (2nd order)\n";
			mesh.genRectWithHole(-3., 3., -2., 2., 0.5, base[0] * scales[i], base[1] * scales[i], 2);
			PlasticitySolver ps(m, mesh, lce);
			elastGpu2_time[i] = ps.solveElast();
			std::cout << std::flush;
		}
		std::cout << "\n\n\n\nTest GPU plasticity float boost (1st order)\n\n";
		for (int i = 0; i < plastGpuF1; ++i) {
			std::cout << "\n\nChilling...";
			//std::this_thread::sleep_for(std::chrono::milliseconds(1000 * (i + 1)));
			std::cout << "\rGPU plasticity float boost test " << base[0] * scales[i] << "x" << base[1] * scales[i] << " (1st order)\n";
			mesh.genRectWithHole(-3., 3., -2., 2., 0.5, base[0] * scales[i], base[1] * scales[i], 1);
			PlasticitySolver ps(m, mesh, lc);
			ps.floatBoost = true;
			switch (i) {
			case 0: case 1: case 2: case 3: case 4: ps.iterOutput = false; break;
			default: ps.iterOutput = true; break;
			}
			plastGpuF1_time[i] = ps.solve();
			std::cout << std::flush;
		}
		std::cout << "\n\n\n\nTest GPU plasticity float boost (2nd order)\n\n";
		for (int i = 0; i < plastGpuF2; ++i) {
			std::cout << "\n\nChilling...";
			//std::this_thread::sleep_for(std::chrono::milliseconds(1000 * (i + 1)));
			std::cout << "\rGPU plasticity float boost test " << base[0] * scales[i] << "x" << base[1] * scales[i] << " (2nd order)\n";
			mesh.genRectWithHole(-3., 3., -2., 2., 0.5, base[0] * scales[i], base[1] * scales[i], 2);
			PlasticitySolver ps(m, mesh, lc);
			ps.floatBoost = true;
			switch (i) {
			case 0: case 1: case 2: case 3: ps.iterOutput = false; break;
			default: ps.iterOutput = true; break;
			}
			plastGpuF2_time[i] = ps.solve();
			std::cout << std::flush;
		}
		std::cout << "\n\nChilling...";
		std::this_thread::sleep_for(std::chrono::milliseconds(5000));

		std::cout << "\n\n\n\nTest GPU plasticity (1st order)\n\n";
		for (int i = 0; i < plastGpu1; ++i) {
			std::cout << "\n\nChilling...";
			//std::this_thread::sleep_for(std::chrono::milliseconds(1000 * (i + 1)));
			std::cout << "\rGPU plasticity test " << base[0] * scales[i] << "x" << base[1] * scales[i] << " (1st order)\n";
			mesh.genRectWithHole(-3., 3., -2., 2., 0.5, base[0] * scales[i], base[1] * scales[i], 1);
			PlasticitySolver ps(m, mesh, lc);
			switch (i) {
			case 0: case 1: case 2: case 3: case 4: ps.iterOutput = false; break;
			default: ps.iterOutput = true; break;
			}
			plastGpu1_time[i] = ps.solve();
			std::cout << std::flush;
		}
		std::cout << "\n\n\n\nTest GPU plasticity (2nd order)\n\n";
		for (int i = 0; i < plastGpu2; ++i) {
			std::cout << "\n\nChilling...";
			//std::this_thread::sleep_for(std::chrono::milliseconds(1000 * (i + 1)));
			std::cout << "\rGPU plasticity test " << base[0] * scales[i] << "x" << base[1] * scales[i] << " (2nd order)\n";
			mesh.genRectWithHole(-3., 3., -2., 2., 0.5, base[0] * scales[i], base[1] * scales[i], 2);
			PlasticitySolver ps(m, mesh, lc);
			switch (i) {
			case 0: case 1: case 2: ps.iterOutput = false; break;
			default: ps.iterOutput = true; break;
			}
			plastGpu2_time[i] = ps.solve();
			std::cout << std::flush;
		}
		std::cout << "\n\nChilling...";
		std::this_thread::sleep_for(std::chrono::milliseconds(5000));
	}
	mesh.useCuda = false;
	std::cout << "\n\n\n\nTest CPU elasticity (1st order)\n\n";
	for (int i = 0; i < elastCpu1; ++i) {
		std::cout << "\n\nChilling...";
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		std::cout << "\rCPU elasticity test " << base[0] * scales[i] << "x" << base[1] * scales[i] << " (1st order)\n";
		mesh.genRectWithHole(-3., 3., -2., 2., 0.5, base[0] * scales[i], base[1] * scales[i], 1);
		PlasticitySolver ps(m, mesh, lce);
		elastCpu1_time[i] = ps.solveElast();
		std::cout << std::flush;
	}
	std::cout << "\n\n\n\nTest CPU elasticity (2nd order)\n\n";
	for (int i = 0; i < elastCpu2; ++i) {
		std::cout << "\n\nChilling...";
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
		std::cout << "\rCPU elasticity test " << base[0] * scales[i] << "x" << base[1] * scales[i] << " (2nd order)\n";
		mesh.genRectWithHole(-3., 3., -2., 2., 0.5, base[0] * scales[i], base[1] * scales[i], 2);
		PlasticitySolver ps(m, mesh, lce);
		elastCpu2_time[i] = ps.solveElast();
		std::cout << std::flush;
	}
	std::cout << "\n\n\n\nTest CPU plasticity (1st order)\n\n";
	for (int i = 0; i < plastCpu1; ++i) {
		std::cout << "\n\nChilling...";
		std::this_thread::sleep_for(std::chrono::milliseconds(1000 * (i + 1)));
		std::cout << "\rCPU plasticity test " << base[0] * scales[i] << "x" << base[1] * scales[i] << " (1st order)\n";
		mesh.genRectWithHole(-3., 3., -2., 2., 0.5, base[0] * scales[i], base[1] * scales[i], 1);
		PlasticitySolver ps(m, mesh, lc);
		switch (i) {
		case 0: case 1: case 2: case 3: ps.iterOutput = false; break;
		default: ps.iterOutput = true; break;
		}
		plastCpu1_time[i] = ps.solve();
		std::cout << std::flush;
	}
	std::cout << "\n\n\nTest CPU plasticity (2nd order)\n\n";
	for (int i = 0; i < plastCpu2; ++i) {
		std::cout << "\n\nChilling...";
		std::this_thread::sleep_for(std::chrono::milliseconds(1000 * (i + 1)));
		std::cout << "\rCPU plasticity test " << base[0] * scales[i] << "x" << base[1] * scales[i] << " (2nd order)\n";
		mesh.genRectWithHole(-3., 3., -2., 2., 0.5, base[0] * scales[i], base[1] * scales[i], 2);
		PlasticitySolver ps(m, mesh, lc);
		switch (i) {
		case 0: case 1: case 2: ps.iterOutput = false; break;
		default: ps.iterOutput = true; break;
		}
		plastCpu2_time[i] = ps.solve();
		std::cout << std::flush;
	}

	/*std::cout << "\n\n\nResults\n\nCPU only\n Elements        Time\n  ";
	for (int i = 0; i < tests; ++i)
		std::cout << elemCount[i] << "          " << cpuTime[i] << "\n  ";
	if (fullTest) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		std::cout << "\n" << deviceProp.name << " " \
			<< round((double)deviceProp.totalGlobalMem / 1'048'576.) << " MB\n";
		std::cout << " Elements        Time\n  ";
		for (int i = 0; i < tests; ++i)
			std::cout << elemCount[i] << "          " << gpuTime[i] << "\n  ";
	}*/

	std::ofstream file("CudaPlasticitySpeedTest.txt", std::ios_base::out);
	file << "CPU: " << getProcessorName() << "\n";
	if (fullTest) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		file << "GPU: " << deviceProp.name << " " \
			<< round((double)deviceProp.totalGlobalMem / 1'048'576.) << " MB\n";
	}
	file << "\nElements          ";
	for (int i = 0; i < 9; ++i) {
		file.width(9); file << elemCount[i] << " ";
	}
	file << "\n\nCPU elast (1st)   ";
	for (int i = 0; i < elastCpu1; ++i) {
		file.width(9); file << elastCpu1_time[i] << " ";
	}
	for (int i = elastCpu1; i < 9; ++i) {
		file.width(9); file << "-" << " ";
	}
	file << "\nCPU elast (2nd)   ";
	for (int i = 0; i < elastCpu2; ++i) {
		file.width(9); file << elastCpu2_time[i] << " ";
	}
	for (int i = elastCpu2; i < 9; ++i) {
		file.width(9); file << "-" << " ";
	}
	file << "\nCPU plast (1st)   ";
	for (int i = 0; i < plastCpu1; ++i) {
		file.width(9); file << plastCpu1_time[i] << " ";
	}
	for (int i = plastCpu1; i < 9; ++i) {
		file.width(9); file << "-" << " ";
	}
	file << "\nCPU plast (2nd)   ";
	for (int i = 0; i < plastCpu2; ++i) {
		file.width(9); file << plastCpu2_time[i] << " ";
	}
	for (int i = plastCpu2; i < 9; ++i) {
		file.width(9); file << "-" << " ";
	}

	if (fullTest) {
		file << "\n\nGPU elast (1st)   ";
		for (int i = 0; i < elastGpu1; ++i) {
			file.width(9); file << elastGpu1_time[i] << " ";
		}
		for (int i = elastGpu1; i < 9; ++i) {
			file.width(9); file << "-" << " ";
		}
		file << "\nGPU elast (2nd)   ";
		for (int i = 0; i < elastGpu2; ++i) {
			file.width(9); file << elastGpu2_time[i] << " ";
		}
		for (int i = elastGpu2; i < 9; ++i) {
			file.width(9); file << "-" << " ";
		}
		file << "\nGPU plast (1st)   ";
		for (int i = 0; i < plastGpu1; ++i) {
			file.width(9); file << plastGpu1_time[i] << " ";
		}
		for (int i = plastGpu1; i < 9; ++i) {
			file.width(9); file << "-" << " ";
		}
		file << "\nGPU plast (2nd)   ";
		for (int i = 0; i < plastGpu2; ++i) {
			file.width(9); file << plastGpu2_time[i] << " ";
		}
		for (int i = plastGpu2; i < 9; ++i) {
			file.width(9); file << "-" << " ";
		}

		file << "\n\nGPU boost (1st)   ";
		for (int i = 0; i < plastGpuF1; ++i) {
			file.width(9); file << plastGpuF1_time[i] << " ";
		}
		for (int i = plastGpuF1; i < 9; ++i) {
			file.width(9); file << "-" << " ";
		}
		file << "\nGPU boost (2nd)   ";
		for (int i = 0; i < plastGpuF2; ++i) {
			file.width(9); file << plastGpuF2_time[i] << " ";
		}
		for (int i = plastGpuF2; i < 9; ++i) {
			file.width(9); file << "-" << " ";
		}
	}
	file << "\n\n\n\nCPU elast 1st\n{";
	for (int i = 0; i < elastCpu1; ++i) {
		if (i != 0) file << ",";
		file << "{" << elemCount[i] << "," << elastCpu1_time[i] << "}";
	}
	file << "}\nCPU elast 2nd\n{";
	for (int i = 0; i < elastCpu2; ++i) {
		if (i != 0) file << ",";
		file << "{" << elemCount[i] << "," << elastCpu2_time[i] << "}";
	}
	file << "}\nCPU plast 1st\n{";
	for (int i = 0; i < plastCpu1; ++i) {
		if (i != 0) file << ",";
		file << "{" << elemCount[i] << "," << plastCpu1_time[i] << "}";
	}
	file << "}\nCPU plast 2nd\n{";
	for (int i = 0; i < plastCpu2; ++i) {
		if (i != 0) file << ",";
		file << "{" << elemCount[i] << "," << plastCpu2_time[i] << "}";
	}
	file << "}";
	if (fullTest) {
		file << "\nGPU elast 1st\n{";
		for (int i = 0; i < elastGpu1; ++i) {
			if (i != 0) file << ",";
			file << "{" << elemCount[i] << "," << elastGpu1_time[i] << "}";
		}
		file << "}\nGPU elast 2nd\n{";
		for (int i = 0; i < elastGpu2; ++i) {
			if (i != 0) file << ",";
			file << "{" << elemCount[i] << "," << elastGpu2_time[i] << "}";
		}
		file << "}\nGPU plast 1st\n{";
		for (int i = 0; i < plastGpu1; ++i) {
			if (i != 0) file << ",";
			file << "{" << elemCount[i] << "," << plastGpu1_time[i] << "}";
		}
		file << "}\nGPU plast 2nd\n{";
		for (int i = 0; i < plastGpu2; ++i) {
			if (i != 0) file << ",";
			file << "{" << elemCount[i] << "," << plastGpu2_time[i] << "}";
		}
		file << "}\nGPU boost 1st\n{";
		for (int i = 0; i < plastGpuF1; ++i) {
			if (i != 0) file << ",";
			file << "{" << elemCount[i] << "," << plastGpuF1_time[i] << "}";
		}
		file << "}\nGPU boost 2nd\n{";
		for (int i = 0; i < plastGpuF2; ++i) {
			if (i != 0) file << ",";
			file << "{" << elemCount[i] << "," << plastGpuF2_time[i] << "}";
		}
		file << "}";
	}
	file.close();

	std::cout << "\n\n\nResults\n\n";
	std::cout << "CPU: " << getProcessorName() << "\n";
	if (fullTest) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, 0);
		std::cout << "GPU: " << deviceProp.name << " " \
			<< round((double)deviceProp.totalGlobalMem / 1'048'576.) << " MB\n";
	}
	std::cout << "\nElements          ";
	for (int i = 0; i < 9; ++i) {
		std::cout.width(9); std::cout << elemCount[i] << " ";
	}
	std::cout << "\n\nCPU elast (1st)   ";
	for (int i = 0; i < elastCpu1; ++i) {
		std::cout.width(9); std::cout << elastCpu1_time[i] << " ";
	}
	for (int i = elastCpu1; i < 9; ++i) {
		std::cout.width(9); std::cout << "-" << " ";
	}
	std::cout << "\nCPU elast (2nd)   ";
	for (int i = 0; i < elastCpu2; ++i) {
		std::cout.width(9); std::cout << elastCpu2_time[i] << " ";
	}
	for (int i = elastCpu2; i < 9; ++i) {
		std::cout.width(9); std::cout << "-" << " ";
	}
	std::cout << "\nCPU plast (1st)   ";
	for (int i = 0; i < plastCpu1; ++i) {
		std::cout.width(9); std::cout << plastCpu1_time[i] << " ";
	}
	for (int i = plastCpu1; i < 9; ++i) {
		std::cout.width(9); std::cout << "-" << " ";
	}
	std::cout << "\nCPU plast (2nd)   ";
	for (int i = 0; i < plastCpu2; ++i) {
		std::cout.width(9); std::cout << plastCpu2_time[i] << " ";
	}
	for (int i = plastCpu2; i < 9; ++i) {
		std::cout.width(9); std::cout << "-" << " ";
	}

	if (fullTest) {
		std::cout << "\n\nGPU elast (1st)   ";
		for (int i = 0; i < elastGpu1; ++i) {
			std::cout.width(9); std::cout << elastGpu1_time[i] << " ";
		}
		for (int i = elastGpu1; i < 9; ++i) {
			std::cout.width(9); std::cout << "-" << " ";
		}
		std::cout << "\nGPU elast (2nd)   ";
		for (int i = 0; i < elastGpu2; ++i) {
			std::cout.width(9); std::cout << elastGpu2_time[i] << " ";
		}
		for (int i = elastGpu2; i < 9; ++i) {
			std::cout.width(9); std::cout << "-" << " ";
		}
		std::cout << "\nGPU plast (1st)   ";
		for (int i = 0; i < plastGpu1; ++i) {
			std::cout.width(9); std::cout << plastGpu1_time[i] << " ";
		}
		for (int i = plastGpu1; i < 9; ++i) {
			std::cout.width(9); std::cout << "-" << " ";
		}
		std::cout << "\nGPU plast (2nd)   ";
		for (int i = 0; i < plastGpu2; ++i) {
			std::cout.width(9); std::cout << plastGpu2_time[i] << " ";
		}
		for (int i = plastGpu2; i < 9; ++i) {
			std::cout.width(9); std::cout << "-" << " ";
		}

		std::cout << "\n\nGPU boost (1st)   ";
		for (int i = 0; i < plastGpuF1; ++i) {
			std::cout.width(9); std::cout << plastGpuF1_time[i] << " ";
		}
		for (int i = plastGpuF1; i < 9; ++i) {
			std::cout.width(9); std::cout << "-" << " ";
		}
		std::cout << "\nGPU boost (2nd)   ";
		for (int i = 0; i < plastGpuF2; ++i) {
			std::cout.width(9); std::cout << plastGpuF2_time[i] << " ";
		}
		for (int i = plastGpuF2; i < 9; ++i) {
			std::cout.width(9); std::cout << "-" << " ";
		}
	}

	std::cout << "\n\n" << std::endl;

#ifdef _WIN64
	std::cout << "Press ENTER to exit..." << std::flush;
	std::cin.get();
#endif
}

//Тест переменной силы на границе
void forceFuncTest() {
	double leftBorder = -1., rightBorder = 1.;
	Mesh mesh;
	mesh.genRectangle(leftBorder, rightBorder, 0, 1, 20, 10);

	LoadConditions lc;
	lc.setDisplacement(3, { 0., 0. });
	lc.fixVertAxis(leftBorder);
	lc.fixVertAxis(rightBorder);
	//lc.setForce(1, { 0., -1e6 });
	lc.setForce(1, [](vec2 r) { return vec2(0., 1e6 * r.x); });

	Material m;
	m.nu = 0.3;
	m.h = 1.;
	m.E = 220e9; //сталь

	PlasticitySolver ps(m, mesh, lc);
	ps.solveElast();
	ps.saveDisplsToFile("../data/forceTest.txt");
	ps.saveAsVtk("../data/forceTest.vtk");
}

//Тест воспроизведения диаграммы деформирования
void diagramTest() {
	Mesh mesh;
	//mesh.useCuda = false;
	mesh.genRectangle(-1., 1., -1., 1., 2, 2);

	LoadConditions lc;
	lc.setGravity();
	//lc.fixVertAxis(0.);
	//lc.fixHorAxis(0.);

	Material m;
	m.setLinearPlast(E_304L, s_T_304L, E_304L * 0.1);
	//m.E = E_304L;//220e9;
	m.nu = nu_304L;//0.3;
	m.h = 1.;
	//m.setLinearPlast(220e9, 100e6, 10e9);

	Material m2;
	m2.setPowerPlast(E_304L, s_T_304L, E_304L * 0.01, 0.015);
	m2.nu = nu_304L;
	m2.h = 1.;

	std::list<double> sigma, eps;

	for (double displ = 0; displ <= 0.02; displ += 0.0005) {
		//lc.setForce(0, vec2(force, 0.));
		//lc.setForce(2, vec2(-force, 0.));
		lc.setDisplacement(2, {}, 0, 1);
		lc.setDisplacement(0, vec2(displ, 0.), 0, 1);

		PlasticitySolver ps(m, mesh, lc);
		ps.solve();

		sigma.push_back(ps.getSxx0());
		eps.push_back(ps.getExx0());
		std::cout << "E_c = " << ps.getSxx0() / ps.getExx0() << "\n\n";
	}

	std::ofstream file("../data/diagramLin.txt", std::ios_base::out);
	for (auto it1 = sigma.begin(), it2 = eps.begin(); it1 != sigma.end(); ++it1, ++it2)
		file << *it2 << " " << *it1 * 1e-6 << "\n";
	file.close();

	sigma.clear();
	eps.clear();

	//for (double force = 0; force <= 340e6; force += 10e6) {
	for (double displ = 0; displ <= 0.02; displ += 0.0005) {
		//lc.setForce(0, vec2(force, 0.));
		//lc.setForce(2, vec2(-force, 0.));
		lc.setDisplacement(2, {}, 0, 1);
		lc.setDisplacement(0, vec2(displ, 0.), 0, 1);

		PlasticitySolver ps(m2, mesh, lc);
		ps.solve();

		sigma.push_back(ps.getSxx0());
		eps.push_back(ps.getExx0());
		std::cout << "E_c = " << ps.getSxx0() / ps.getExx0() << "\n\n";
	}

	std::ofstream file2("../data/diagramPow.txt", std::ios_base::out);
	for (auto it1 = sigma.begin(), it2 = eps.begin(); it1 != sigma.end(); ++it1, ++it2)
		file2 << *it2 << " " << *it1 * 1e-6 << "\n";
	file2.close();
}

//Тест одноосного пластического сжатия
void plastPressTest() {
	Mesh mesh;
	//mesh.genRectWithHole(-3., 3., -2., 2., 0.5, 48, 32, 2);
	mesh.genRectangle(-3., 3., -2., 2., 4 * 60, 4 * 40, 1);

	LoadConditions lc;
	//lc.setForce(1, vec2(0., -180e6));
	//lc.fixBorderVert(1);
	//lc.setForce(3, vec2(0., 180e6));
	lc.setDisplacement(1, vec2(0., -1e-2));
	lc.setDisplacement(3, vec2(0., 0.));
	//lc.fixVertAxis(0.);
	//lc.fixHorAxis(0.);
	lc.setGravity();
	lc.fixVertAxis(0.);

	Material m;
	m.h = 1.;
	m.rho = 7950;
	//m.setLinearPlast(E_304L, s_T_304L, E_304L * 0.1);
	m.setPowerPlast(E_304L, s_T_304L, E_304L * 0.01, 0.015);
	//m.E = E_304L;//220e9;
	m.nu = nu_304L;//0.3;

	PlasticitySolver ps(m, mesh, lc);
	ps.solve();
	ps.saveAsVtk("../data/plastPress.vtk");
}

void contactTest() {
	Mesh mesh;
	mesh.loadFromFile("ANSA/Contact.inp");
	mesh.renumByDirection();

	LoadConditions lc;
	lc.fixBorder(0);
	lc.fixVertAxis(-6.);
	lc.fixVertAxis(6.);
	lc.fixBorderVert(1);
	lc.setForce(1, vec2(0, -300));

	Material m;
	m.setPowerPlast(193'000, 170, 1930, 0.02);
	m.nu = 0.29;
	m.h = 1.;
	
	PlasticitySolver ps(m, mesh, lc);
	ps.setPlainCondition(strain);
	ps.solve();
	ps.saveAsVtk("../data/contact.vtk");
}

//Записать точное решение задачи Кирша в файл
void writeKirschExact() {
	Mesh mesh;
	mesh.useCuda = false;
	mesh.genRectWithHole(-3., 3., -2., 2., 0.5, 48 * 4, 32 * 4);

	std::ofstream file("../data/Kirsch.vtk", std::ios_base::out);
	file << "# vtk DataFile Version 2.0\n";
	file << "Kirsch\n";
	file << "ASCII\n";
	file << "DATASET POLYDATA\n";
	file << "POINTS " << mesh.nodeCount << " float\n";
	for (int i = 0; i < mesh.nodeCount; ++i)
		file << mesh.node[i].x << " " << mesh.node[i].y << " " << 0 << "\n";
	file << "POLYGONS " << mesh.elemPos[2] << " " << 4 * mesh.count3 + 5 * mesh.count4 + 9 * mesh.count8;
	for (size_t i = 0; i < mesh.count3; ++i) {
		file << "\n3 ";
		for (size_t j = 3 * i; j < 3 * (i + 1); ++j)
			file << mesh.elem3[j] << " ";
	}
	for (size_t i = 0; i < mesh.count4; ++i) {
		file << "\n4 ";
		for (size_t j = 4 * i; j < 4 * (i + 1); ++j)
			file << mesh.elem4[j] << " ";
	}
	for (size_t i = 0; i < mesh.count8; ++i) {
		file << "\n8 ";
		for (size_t j = 8 * i; j < 8 * i + 4; ++j)
			file << mesh.elem8[j] << " " << mesh.elem8[j + 4] << " ";
	}

	double a = 0.5,  //радиус отверсия в пластине
		S = 1;   //сила, приложенная к сторонам
	auto s_r = [&](vec2 r) { return 0.5 * S * (1. - a * a / (r.x * r.x)) + 0.5 * S * (1. + 3. * pow(a, 4.) / pow(r.x, 4.) - 4. * a * a / (r.x * r.x)) * cos(2. * r.y); };
	auto s_phi = [&](vec2 r) { return 0.5 * S * (1. + a * a / (r.x * r.x)) - 0.5 * S * (1. + 3. * pow(a, 4.) / pow(r.x, 4.)) * cos(2. * r.y); };
	auto tau = [&](vec2 r) { return -0.5 * S * (1. - 3. * pow(a, 4.) / pow(r.x, 4.) + 2. * a * a / (r.x * r.x)) * sin(2. * r.y); };
	const double pi = 3.141'592'653'589'793;
	auto sqr = [](double x) { return x * x; };

	file << "\nPOINT_DATA " << mesh.nodeCount << \
		"\nSCALARS NodeID int 1\nLOOKUP_TABLE my_table";
	for (size_t i = 0; i < mesh.nodeCount; ++i)
		file << "\n" << i;
	file << "\nFIELD FieldData2 4\nSigma_r 1 " << mesh.nodeCount << " float\n";
	for (size_t i = 0; i < mesh.nodeCount; ++i) {
		vec2 ec = mesh.node[i];
		vec2 r(sqrt(ec.x * ec.x + ec.y * ec.y), atan(ec.y / ec.x) + (ec.x < 0. ? pi : 0.));
		file << s_r(r) << " ";
	}
	file << "\nSigma_phi 1 " << mesh.nodeCount << " float\n";
	for (size_t i = 0; i < mesh.nodeCount; ++i) {
		vec2 ec = mesh.node[i];
		vec2 r(sqrt(ec.x * ec.x + ec.y * ec.y), atan(ec.y / ec.x) + (ec.x < 0. ? pi : 0.));
		file << s_phi(r) << " ";
	}
	file << "\nTau_rphi 1 " << mesh.nodeCount << " float\n";
	for (size_t i = 0; i < mesh.nodeCount; ++i) {
		vec2 ec = mesh.node[i];
		vec2 r(sqrt(ec.x * ec.x + ec.y * ec.y), atan(ec.y / ec.x) + (ec.x < 0. ? pi : 0.));
		file << tau(r) << " ";
	}
	file << "\nSigma_intensity 1 " << mesh.nodeCount << " float\n";
	for (size_t i = 0; i < mesh.nodeCount; ++i) {
		vec2 ec = mesh.node[i];
		vec2 r(sqrt(ec.x * ec.x + ec.y * ec.y), atan(ec.y / ec.x) + (ec.x < 0. ? pi : 0.));
		file << sqrt(0.5 * (sqr(s_r(r)) + sqr(s_phi(r)) + sqr(s_r(r) - s_phi(r)) + 6. * sqr(tau(r)))) << " ";
	}

	file.close();
}

void meshTest() {
	double S = 170e6; //растягивающая сила
	LoadConditions lc;
	lc.setForce(1, vec2(S, 0.));
	lc.setForce(3, vec2(-S, 0.));
	//lc.setForce(0, vec2(S, 0.));
	//lc.setForce(2, vec2(-S, 0.));
	//lc.setDisplacement(0, { 0., 0. });
	lc.fixVertAxis(0.);
	lc.fixHorAxis(0.);

	Material m;
	m.h = 1.;
	m.setLinearPlast(193e3 * 1e6, 170 * 1e6, 193e2 * 1e6);
	m.nu = 0.29;

	Mesh mesh;
	//mesh.useCuda = false;
	//mesh.loadFromFile("ANSA/HolePlate2.inp");
	//mesh.loadFromFile("ANSA/HolePlate_1x.inp");
	//mesh.loadFromFile("ANSA/HolePlate_1p_1x.inp");
	//mesh.loadFromFile("ANSA/SimpleRect2.inp");
	//mesh.renumByDirection();
	//for (int i = 0; i < mesh.nodeCount; ++i) \
		std::cout << mesh.secOrdNodes[i] << " ";
	//mesh.meshToGPU();
	//mesh.meshToRAM();

	//mesh.genRectangle(-1, 1, -1, 1, 4, 4);
	//mesh.genRectWithHole(-3., 3., -2., 2, 0.5, 48, 32);
	const int base[2] = { 6, 4 };
	const int scales[9] = { 4, 6, 8, 11, 16, 23, 32, 45, 64 };
	int elemCount[9] = { 1280, 2880, 5120, 9460, 20480, 41860, 81920, 161100, 327680 };
	for (int i = 0; i < 4; i += 1) {
		mesh.genRectWithHole(-3., 3., -2., 2., 0.5, base[0] * scales[i], base[1] * scales[i], 1);
		PlasticitySolver ps(m, mesh, lc);
		//ps.floatBoost = true;
		ps.solve();
	}
	//mesh.meshToRAM();
	Mesh mesh2(mesh);
	
	//mesh.print();
	//mesh2.saveAsVtk("../data/TestRead.vtk");
	std::cout << "max diff " << mesh.findMaxIndexDiff();
	std::cout << "\navr size " << mesh.avrElemSize() << "\n";

	//std::cin.get();
	
	PlasticitySolver ps(m, mesh, lc);
	ps.solveElast();
	//ps.saveAsVtk("../data/test2.vtk");
}

void meshUpgradeTest() {
	Mesh mesh1;
	mesh1.genRectangle(-2, 2, -1, 1, 12, 6, 2);
	std::cout << "max diff " << mesh1.findMaxIndexDiff() << "\n\n";
	//mesh1.print();
	mesh1.saveAsVtk("../data/rect1.vtk");

	Mesh mesh2;
	mesh2.genRectWithHole(-3, 3, -2, 2, 0.5, 12, 8, 2);
	std::cout << "max diff " << mesh2.findMaxIndexDiff() << "\n\n";
	mesh2.saveAsVtk("../data/hole1.vtk");

	Mesh mesh3;
	mesh3.genRing(1, 2, 40, 6, 2);
	std::cout << "max diff " << mesh3.findMaxIndexDiff() << "\n\n";
	mesh3.saveAsVtk("../data/ring1.vtk");
}


int main() { //выбор запускаемого теста

	//meshTest();

	//meshUpgradeTest();

	//meshTransferTest();

	//solverTest();

	//holePlateTest();

	//testRing();

	//energyConvTest();

	//plasticityTest();

	//kirschConvTest();

	//forceFuncTest();

	//diagramTest();

	//writeKirschExact();
	
	//plastPressTest();

	//contactTest();

	//speedTestAlpha();

	speedTest2();

}
