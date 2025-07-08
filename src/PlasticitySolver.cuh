#pragma once

#include <omp.h>

#include "SymMatrix.cuh"
#include "StaticMatrix.cuh"
#include "Mesh.cuh"
#include "LoadConditions.cuh"
#include "SLAE_Solvers.cuh"
#include "ConjGradCuda.cu"
#include "Material.cuh"


//Плоское состояние (напряжённое, деформированное)
enum plainCondition { stress, strain };


const int secIntPs = 3; //количество точек интегрирования по одной оси для элементов 2го порядка (TEMP)
const int secIntPs2 = secIntPs * secIntPs;


//Класс решения задачи пластичности
class PlasticitySolver {

private:

	Material m;
	Material* dev_m;  //указатель на данные материала на видеокарте

	Mesh mesh;   //Сетка

	LoadConditions cond;  //Условия нагружения

	plainCondition pc = stress;

	void (*formC)(StaticMatrix<3, 3, double>&, double, double) = nullptr;
	void (*dd_formC)(StaticMatrix<3, 3, double>&, double, double) = nullptr;
	void (*df_formC)(StaticMatrix<3, 3, float>&, float, float) = nullptr;

	int slaeSize = 0;
	int memLen = 0;

	std::vector<double> uv;  //TODO: сделать указатель

	double* dd_uv = nullptr;
	float* df_uv = nullptr;

	//Механические параметры для процессора
	double* exx, * eyy, * gamma, * intensityE;
	double* exx_p, * eyy_p, * gamma_p, * intE_p;
	double* sxx, * syy, * tau, * intensityS, * tableS;
	double* psi, * E_c, * nu_c;

	//Механические параметры для видеокарты во float
	float* df_exx, * df_eyy, * df_gamma, * df_intensityE;
	float* df_sxx, * df_syy, * df_tau, * df_intensityS, * df_tableS;
	float* df_psi, * df_E_c, * df_nu_c;

	//Механические параметры для видеокарты в double
	double* dd_exx, * dd_eyy, * dd_gamma, * dd_intensityE;
	double* dd_exx_p, * dd_eyy_p, * dd_gamma_p, * dd_intE_p;
	double* dd_sxx, * dd_syy, * dd_tau, * dd_intensityS, * dd_tableS;
	double* dd_psi, * dd_E_c, * dd_nu_c;

	float* df_exit = nullptr;  //Значение для критерия останова на видеокарте
	double* dd_exit = nullptr;

	bool plastSolved = false;

	bool ramSaved = true;

	bool* kinMask = nullptr;  //Маска кинематических условий для процессора
	
	//int kinCount = 0;

	bool* dev_kinNodes;  //Маска кинематических условий для видеокарты

	StaticMatrix<3, 3, double>* C = nullptr;
	StaticMatrix<2, 3, double>* B3 = nullptr;
	StaticMatrix<2, 4, double>* B4 = nullptr;
	StaticMatrix<2, 8, double>* B8 = nullptr;

	double* detJ4 = nullptr;
	double* detJ8 = nullptr;

	StaticMatrix<3, 3, double>* dd_C = nullptr;
	StaticMatrix<2, 3, double>* dd_B3 = nullptr;
	StaticMatrix<2, 4, double>* dd_B4 = nullptr;
	StaticMatrix<2, 8, double>* dd_B8 = nullptr;

	double* dd_detJ4 = nullptr;
	double* dd_detJ8 = nullptr;

	SymMatrix<8, double>* dd_Ke4 = nullptr;
	SymMatrix<16, double>* dd_Ke8 = nullptr;

	StaticMatrix<3, 3, float>* df_C = nullptr;
	StaticMatrix<2, 3, float>* df_B3 = nullptr;
	StaticMatrix<2, 4, float>* df_B4 = nullptr;
	StaticMatrix<2, 8, float>* df_B8 = nullptr;

	float* df_detJ4 = nullptr;
	float* df_detJ8 = nullptr;

	SymMatrix<8, float>* df_Ke4 = nullptr;
	SymMatrix<16, float>* df_Ke8 = nullptr;

	void initPlastParams();
	void initPlastParamsFloat();
	
	void initConditions(SparseSLAE& K);

	template<typename fp>
	void initConditions_(CudaSparseSLAE<fp>& K, fp* uv);

	template<typename fp>
	void initConditions(CudaSparseSLAE<fp>& K, fp* uv);

	void calcBs();

	void calcBsCuda();

	void copyBs();

	void copyConditions(float* rp_f, double* rp, int memLen);

	void copyFloatToDouble(int memLen);

	void distribKe4(StripSLAE& K, const SymMatrix<8, double>& Ke, int e) const;
	void distribKe8(StripSLAE& K, const SymMatrix<16, double>& Ke, int e) const;

	void distribKe(SparseSLAE& K, const SymMatrix<8, double>& Ke, int e) const;

	void fillGlobalStiffness(StripSLAE& K, SparseSLAE& spK);

	//void fillGlobalStiffness(SparseSLAE& K);

	void fillGlobalStiffness(CudaSLAE<double>& K, CudaSparseSLAE<double>& spK);

	void fillGlobalStiffness(CudaSLAE<float>& K, CudaSparseSLAE<float>& spK);
	
	void updateParameters(int iterNum = 0);

	void updateParamDouble();

	void updateParamFloat();

	void calcPlastDeform();

	void calcPlastDeformCuda();

	void dataToRAM();

	double exitCondition() const {
		double norm = 0., denom = 0.;
#pragma omp parallel for reduction (+ : norm, denom)
		for (int i = 0; i < mesh.elemCount(); ++i) {
			double diff = intensityS[i] - tableS[i];
			norm += diff * diff;
			denom += tableS[i] * tableS[i];
		}
		return sqrt(norm / denom);
	}

	template<typename fp>
	fp exitCondition(fp* intS, fp* tableS, fp* exitNorm);

	//Преобразование тензора в полярную СК, компонента rr
	double _rr(size_t e, const double* _x, const double* _y, const double* _xy) const {
		vec2 ec;
		for (int i = e * 4; i < (e + 1) * 4; ++i)
			ec += mesh.node[mesh.elem4[i]];
		ec /= 4;
		double r = ec.norm();
		double cos_ = ec.x / r, cos12 = ec.y / r;
		return cos_ * cos_ * _x[e] + cos12 * cos12 * _y[e] + 2. * cos_ * cos12 * _xy[e];
	}

	//Преобразование тензора в полярную СК, компонента phiphi
	double _phiphi(size_t e, const double* _x, const double* _y, const double* _xy) const {
		vec2 ec;
		for (int i = e * 4; i < (e + 1) * 4; ++i)
			ec += mesh.node[mesh.elem4[i]];
		ec /= 4;
		double r = ec.norm();
		double cos_ = ec.x / r, cos12 = ec.y / r;
		return cos12 * cos12 * _x[e] + cos_ * cos_ * _y[e] - 2. * cos_ * cos12 * _xy[e];
	}

	//Преобразование тензора в полярную СК, компонента rphi
	double _rphi(size_t e, const double* _x, const double* _y, const double* _xy) const {
		vec2 ec;
		for (int i = e * 4; i < (e + 1) * 4; ++i)
			ec += mesh.node[mesh.elem4[i]];
		ec /= 4;
		double r = ec.norm();
		double cos_ = ec.x / r, cos12 = ec.y / r;
		return cos_ * cos12 * (-_x[e] + _y[e]) + _xy[e] * (cos_ * cos_ - cos12 * cos12);
	}

	std::function<void(size_t, size_t, double)> printIter = [](size_t, size_t, double) {};

public:

	//Запись деформаций и напряжений в полярных координатах
	// (можно было бы перенести в аргумент метода сохранения)
	bool polarCoord = false;

	bool iterOutput = true;

	bool floatBoost = false;

	double getSxx0() const {
		if (mesh.useCuda) {
			double sxx0 = 0;
			cudaMemcpy(&sxx0, dd_sxx, sizeof(double), cudaMemcpyDeviceToHost);
			return sxx0;
		}
		else return sxx[0];
		//return intensityS[0];
	}

	double getExx0() const {
		if (mesh.useCuda) {
			double exx0 = 0;
			cudaMemcpy(&exx0, dd_exx, sizeof(double), cudaMemcpyDeviceToHost);
			return exx0;
		}
		else return exx[0];
	}

	void setPlainCondition(plainCondition _pc);

	PlasticitySolver() = default;

	PlasticitySolver(const Material& material, Mesh& _mesh, const LoadConditions& loadConditions)
		: m(material), mesh(_mesh), cond(loadConditions) {

		exx = new double[mesh.elemCount()];
		eyy = new double[mesh.elemCount()];
		gamma = new double[mesh.elemCount()];
		intensityE = new double[mesh.elemCount()];
		exx_p = new double[mesh.elemCount()];
		eyy_p = new double[mesh.elemCount()];
		gamma_p = new double[mesh.elemCount()];
		intE_p = new double[mesh.elemCount()];
		sxx = new double[mesh.elemCount()];
		syy = new double[mesh.elemCount()];
		tau = new double[mesh.elemCount()];
		intensityS = new double[mesh.elemCount()];
		//intS_1 = new double[mesh.elemCount()];
		//intS_2 = new double[mesh.elemCount()];
		tableS = new double[mesh.elemCount()];
		psi = new double[mesh.elemCount()];
		E_c = new double[mesh.elemCount()];
		nu_c = new double[mesh.elemCount()];
		for (int i = 0; i < mesh.elemCount(); ++i) {
			psi[i] = 1.;
			E_c[i] = m.E;
			nu_c[i] = m.nu;
		}
		C = new StaticMatrix<3, 3, double>[mesh.elemCount()];
		B3 = new StaticMatrix<2, 3, double>[mesh.count3];
		B4 = new StaticMatrix<2, 4, double>[4 * mesh.count4];
		detJ4 = new double[4 * mesh.count4];
		B8 = new StaticMatrix<2, 8, double>[secIntPs2 * mesh.count8];
		detJ8 = new double[secIntPs2 * mesh.count8];
		kinMask = new bool[2 * mesh.nodeCount];

		size_t memLen = ((mesh.nodeCount * 2) + BS - 1) / BS * BS;
		uv.resize(memLen);
		//uv.resize(grid.node.size() * 2);

		//mesh.meshToRAM();
		//mesh.print();

		setPlainCondition(pc);

		if (mesh.useCuda) {
			cudaMalloc((void**)&dev_m, sizeof(Material));
			cudaMemcpy(dev_m, &m, sizeof(Material), cudaMemcpyHostToDevice);

			int elemMemLen = (mesh.elemCount() + BS - 1) / BS * BS;//???????????
			cudaMalloc((void**)&dd_exx, elemMemLen * sizeof(double));
			cudaMalloc((void**)&dd_eyy, elemMemLen * sizeof(double));
			cudaMalloc((void**)&dd_gamma, elemMemLen * sizeof(double));
			cudaMalloc((void**)&dd_intensityE, elemMemLen * sizeof(double));
			cudaMalloc((void**)&dd_exx_p, elemMemLen * sizeof(double));
			cudaMalloc((void**)&dd_eyy_p, elemMemLen * sizeof(double));
			cudaMalloc((void**)&dd_gamma_p, elemMemLen * sizeof(double));
			cudaMalloc((void**)&dd_intE_p, elemMemLen * sizeof(double));
			cudaMalloc((void**)&dd_sxx, elemMemLen * sizeof(double));
			cudaMalloc((void**)&dd_syy, elemMemLen * sizeof(double));
			cudaMalloc((void**)&dd_tau, elemMemLen * sizeof(double));
			cudaMalloc((void**)&dd_intensityS, elemMemLen * sizeof(double));
			cudaMalloc((void**)&dd_tableS, elemMemLen * sizeof(double));
			cudaMalloc((void**)&dd_E_c, elemMemLen * sizeof(double));
			cudaMalloc((void**)&dd_nu_c, elemMemLen * sizeof(double));
			cudaMalloc((void**)&dd_psi, elemMemLen * sizeof(double));
			cudaMalloc((void**)&dd_uv, memLen * sizeof(double));
			cudaMalloc((void**)&dd_C, elemMemLen * sizeof(StaticMatrix<3, 3, double>));
			cudaMalloc((void**)&dd_exit, 2 * sizeof(double));

			cudaMalloc((void**)&df_exx, elemMemLen * sizeof(float));
			cudaMalloc((void**)&df_eyy, elemMemLen * sizeof(float));
			cudaMalloc((void**)&df_gamma, elemMemLen * sizeof(float));
			cudaMalloc((void**)&df_intensityE, elemMemLen * sizeof(float));
			cudaMalloc((void**)&df_sxx, elemMemLen * sizeof(float));
			cudaMalloc((void**)&df_syy, elemMemLen * sizeof(float));
			cudaMalloc((void**)&df_tau, elemMemLen * sizeof(float));
			cudaMalloc((void**)&df_intensityS, elemMemLen * sizeof(float));
			cudaMalloc((void**)&df_tableS, elemMemLen * sizeof(float));
			cudaMalloc((void**)&df_E_c, elemMemLen * sizeof(float));
			cudaMalloc((void**)&df_nu_c, elemMemLen * sizeof(float));
			cudaMalloc((void**)&df_psi, elemMemLen * sizeof(double));
			cudaMalloc((void**)&df_uv, memLen * sizeof(float));
			cudaMalloc((void**)&df_C, elemMemLen * sizeof(StaticMatrix<3, 3, float>));
			cudaMalloc((void**)&df_exit, 2 * sizeof(float));

			cudaMalloc((void**)&dev_kinNodes, memLen * sizeof(bool));

			cudaMalloc((void**)&dd_B4, 4 * mesh.count4 * sizeof(StaticMatrix<2, 4, double>));
			cudaMalloc((void**)&dd_detJ4, 4 * mesh.count4 * sizeof(double));
			cudaMalloc((void**)&dd_Ke4, mesh.count4 * sizeof(SymMatrix<8, double>));

			cudaMalloc((void**)&df_B4, 4 * mesh.count4 * sizeof(StaticMatrix<2, 4, float>));
			cudaMalloc((void**)&df_detJ4, 4 * mesh.count4 * sizeof(float));
			cudaMalloc((void**)&df_Ke4, mesh.count4 * sizeof(SymMatrix<8, float>));

			cudaMalloc((void**)&dd_B8, secIntPs2 * mesh.count8 * sizeof(StaticMatrix<2, 8, double>));
			cudaMalloc((void**)&dd_detJ8, secIntPs2 * mesh.count8 * sizeof(double));
			cudaMalloc((void**)&dd_Ke8, mesh.count8 * sizeof(SymMatrix<16, double>));

			cudaMalloc((void**)&df_B8, secIntPs2 * mesh.count8 * sizeof(StaticMatrix<2, 8, float>));
			cudaMalloc((void**)&df_detJ8, secIntPs2 * mesh.count8 * sizeof(float));
			cudaMalloc((void**)&df_Ke8, mesh.count8 * sizeof(SymMatrix<16, float>));
		}
	};

	~PlasticitySolver() {
		delete[] exx; delete[] eyy; delete[] gamma; delete[] intensityE;
		delete[] exx_p; delete[] eyy_p; delete[] gamma_p; delete[] intE_p;
		delete[] sxx; delete[] syy; delete[] tau; delete[] intensityS; delete[] tableS;
		//delete[] intS_1; delete[] intS_2;
		delete[] psi; delete[] E_c; delete[] nu_c;
		delete[] kinMask;
		delete[] C;
		delete[] B3; delete[] B4; delete[] B8;
		delete[] detJ4; delete detJ8;
		if (mesh.useCuda) {
			cudaFree(dd_exx); cudaFree(dd_eyy); cudaFree(dd_gamma); cudaFree(dd_intensityE);
			cudaFree(dd_exx_p); cudaFree(dd_eyy_p); cudaFree(dd_gamma_p); cudaFree(dd_intE_p);
			cudaFree(dd_sxx); cudaFree(dd_syy); cudaFree(dd_tau); cudaFree(dd_intensityS); cudaFree(dd_tableS);
			cudaFree(dd_psi); cudaFree(dd_E_c); cudaFree(dd_nu_c);
			cudaFree(dd_uv);
			cudaFree(dd_exit);

			cudaFree(df_exx); cudaFree(df_eyy); cudaFree(df_gamma); cudaFree(df_intensityE);
			cudaFree(df_sxx); cudaFree(df_syy); cudaFree(df_tau); cudaFree(df_intensityS); cudaFree(df_tableS);
			cudaFree(df_psi); cudaFree(df_E_c); cudaFree(df_nu_c);
			cudaFree(df_uv);
			cudaFree(df_exit);

			cudaFree(dev_kinNodes);

			cudaFree(dd_C);
			cudaFree(dd_B3); cudaFree(dd_B4); cudaFree(dd_B8);
			cudaFree(dd_Ke4); cudaFree(dd_Ke8);
			cudaFree(dd_detJ4); cudaFree(dd_detJ8);

			cudaFree(df_C);
			cudaFree(df_B3); cudaFree(df_B4); cudaFree(df_B8);
			cudaFree(df_Ke4); cudaFree(df_Ke8);
			cudaFree(df_detJ4); cudaFree(df_detJ8);
		}
	}

	double solveElastCPU();

	double solveElastCUDA();

	//Решить задачу упругости
	double solveElast() {
		if (mesh.useCuda)
			return solveElastCUDA();
		else
			return solveElastCPU();
	}

	double solveCPU();

	double solveCUDA();

	double solveCUDA_FD();

	//Решить задачу пластичности
	double solve() {
		if (iterOutput)
			printIter = [](size_t iterNum, size_t insideIter, double relErr) {
				std::cout << "\n\rIteration " << iterNum + 1 << ", inner iterations: " << insideIter << ", error: " << relErr << "   ";
			};
		else printIter = [](size_t, size_t, double) {};

		if (mesh.useCuda) {
			if (floatBoost)
				return solveCUDA_FD();
			else
				return solveCUDA();
		}
		else
			return solveCPU();
	}

	//Энергия деформации
	double deformEnergy() {
		if (!mesh.ramSaved)
			mesh.meshToRAM();
		if (!ramSaved)
			dataToRAM();
		double energy = 0.;
		double volume = 0.;
		for (size_t e = 0; e < mesh.count4; ++e) {
			double elemVol = m.h * mesh.elemSpace4(e);
			int i = e + mesh.elemPos[0];
			volume += elemVol;
			//energy += (sxx[i] * exx[i] + syy[i] * eyy[i] + 2. * tau[i] * gamma[i]) * 0.5 * elemVol;
			if (intensityS[i] > m.sigmaT) {
				double E_T = m.sigmaT / m.E;
				energy += m.sigmaT * E_T * 0.5 * elemVol;
				double h_eps = (intensityE[i] - E_T) * 0.01;
				for (int j = 0; j < 100; ++j)
					energy += h_eps * 0.5 * (m.f(E_T + h_eps * j) + m.f(E_T + h_eps * (j + 1))) * elemVol;
			}
			else
				energy += intensityE[i] * intensityS[i] * 0.5 * elemVol;
		}
		for (int e = 0; e < mesh.count8; ++e) {
			double elemVol = m.h * mesh.elemSpace8(e);
			int i = e + mesh.elemPos[1];
			volume += elemVol;
			//energy += (sxx[i] * exx[i] + syy[i] * eyy[i] + 2. * tau[i] * gamma[i]) * 0.5 * elemVol;
			energy += intensityE[i] * intensityS[i] * 0.5 * elemVol;
		}
		return energy;// / volume;
	}

	//Сохранить перемещения в файл
	void saveDisplsToFile(const std::string& fileName) {
		if (!mesh.ramSaved)
			mesh.meshToRAM();
		std::ofstream file(fileName, std::ios_base::out);
		for (size_t i = 0; i < mesh.nodeCount; ++i) {
			file << mesh.node[i].x << " " << mesh.node[i].y << " " \
				<< uv[2 * i] << " " << uv[2 * i + 1] << "\n";
		}
		file << std::flush;
		file.close();
	}

	//Сохранить в файл формата .vtk для отображения результатов в ParaView
	void saveAsVtk(const std::string& fileName);

	//TO DO: fix for elem8
	double sigmaError(std::function<double(vec2)> s_x, std::function<double(vec2)> s_y, std::function<double(vec2)> s_xy) const {
		double avrSxx = 0., avrSyy = 0., avrSxy = 0., vol = 0.,
			maxErrSxx = 0., maxErrSyy = 0., maxErrSxy = 0.;

		for (size_t e = 0; e < mesh.count4; ++e) {
			double loc = mesh.elemSpace4(e);
			avrSxx += sxx[e] * loc;
			avrSyy += syy[e] * loc;
			avrSxy += tau[e] * loc;
			vol += loc;

			vec2 ec;
			for (int i = e * 4; i < (e + 1) * 4; ++i)
				ec += mesh.node[mesh.elem4[i]];
			ec /= 4;

			double err = fabs(s_x(ec) - sxx[e]);
			if (err > maxErrSxx)
				maxErrSxx = err;
			err = fabs(s_y(ec) - syy[e]);
			if (err > maxErrSyy)
				maxErrSyy = err;
			err = fabs(s_xy(ec) - tau[e]);
			if (err > maxErrSxy)
				maxErrSxy = err;
		}
		maxErrSxx *= vol / avrSxx;
		maxErrSyy *= vol / avrSyy;
		maxErrSxy *= vol / avrSxy;
		return fmax(fmax(maxErrSxx, maxErrSyy), maxErrSxy);
	}

	//TO DO: fix for elem8
	double polarSigmaError(std::function<double(vec2)> s_r, std::function<double(vec2)> s_phi, std::function<double(vec2)> s_rphi, double maxR) const {
		double avrSrr = 0., avrSphiphi = 0., avrSrphi = 0., vol = 0.,
			maxErrSrr = 0., maxErrSphiphi = 0., maxErrSrphi = 0.;

		for (size_t e = 0; e < mesh.count4; ++e) {
			vec2 ec;
			for (int i = e * 4; i < (e + 1) * 4; ++i)
				ec += mesh.node[mesh.elem4[i]];
			ec /= 4;

			const double pi = 3.141'592'653'589'793;
			vec2 r(sqrt(ec.x * ec.x + ec.y * ec.y), atan(ec.y / ec.x) + (ec.x < 0. ? pi : 0.));
			/*if (e == 200) {
				std::cout << "xy " << ec.x << " " << ec.y << "\nrphi " << r.x << " " << r.y << "\n";
			}*/
			if (r.x > maxR) continue;


			double sr = _rr(e, sxx, syy, tau);
			double sphi = _phiphi(e, sxx, syy, tau);
			double srphi = _rphi(e, sxx, syy, tau);

			double loc = mesh.elemSpace4(e);
			avrSrr += fabs(sr) * loc;
			avrSphiphi += fabs(sphi) * loc;
			avrSrphi += fabs(srphi) * loc;
			vol += loc;

			double err = fabs(s_r(r) - sr);
			if (err > maxErrSrr)
				maxErrSrr = err;
			err = fabs(s_phi(r) - sphi);
			if (err > maxErrSphiphi)
				maxErrSphiphi = err;
			err = fabs(s_rphi(r) - srphi);
			if (err > maxErrSrphi)
				maxErrSrphi = err;
		}

		maxErrSrr *= vol / avrSrr;
		maxErrSphiphi *= vol / avrSphiphi;
		maxErrSrphi *= vol / avrSrphi;
		std::cout << maxErrSrr << " " << maxErrSphiphi << " " << maxErrSrphi << "\n";
		return fmax(fmax(maxErrSrr, maxErrSphiphi), maxErrSrphi);
	}

};