#pragma once

#include "ptrs.h"
#include "DeviceCommon.cuh"
#include "CGKernels.cuh"

#include <iostream> //DEBUG


template<typename fp>
class CudaJCGV {

private:

	unsigned N = 0;
	fp* data = nullptr;
	unsigned* rows = nullptr;
	unsigned* cols = nullptr;

	fp* rhs = nullptr;

	fp* x = nullptr;

	bool* mask = nullptr;

	unsigned lineCount = 0;
	fp* lines = nullptr;
	unsigned* lineRows = nullptr;

	dptr<fp> r, z, s, q, DR;

	dptr<unsigned> dev_iters;

	dptr<fp> scalars;         // 0 - omega, 1 - rhoNext, 2 - rhoPrev
	fp* omega = nullptr;      // scalars + 0
	fp* rhoNext = nullptr;    // scalars + 1
	fp* rhoPrev = nullptr;    // scalars + 2

	dptr<fp> condVals;        // 0 - eps, 1 - normB, 2 - eps2b
	fp* normB = nullptr;      // condVals + 0
	fp* dev_eps = nullptr;    // condVals + 1
	fp* eps2b = nullptr;      // condVals + 2

	cudaStream_t stream;
	cudaGraph_t bodyGraph, mainGraph, renormGraph = {};
	cudaGraphExec_t execGraph;


	//TODO: method returning kernel by vectoriztion, fp type & preconditioning
	//TODO: vectorization setup


public:

	CudaJCGV(unsigned N, fp* data, unsigned* rows, unsigned* cols, \
		fp* rhs, fp* solutionVector, bool* mask, \
		bool preconditioning, bool updatingMatrix, bool updatingRhs, \
		unsigned dim = 1, unsigned blockSize = 512,
		fp* lines = nullptr, unsigned* lineRows = nullptr, unsigned lineCount = 0)

		: N(N), data(data), rows(rows), cols(cols), rhs(rhs), x(solutionVector), \
		mask(mask), lineCount(lineCount), lines(lines), lineRows(lineRows)
	{
		unsigned memlen = (N + blockSize - 1) / blockSize * blockSize;

		r.malloc(memlen);      // Vectors memory allocation
		z.malloc(memlen);
		s.malloc(memlen);
		if (preconditioning) {
			q.malloc(memlen);
			DR.malloc(memlen);
			q.setZero(memlen);
			DR.setZero(memlen);
		}
		r.setZero(memlen);
		z.setZero(memlen);
		s.setZero(memlen);

		dev_iters.malloc(1);    // Scalar data allocation
		scalars.malloc(3);
		omega = scalars, rhoNext = scalars + 1, rhoPrev = scalars + 2;
		condVals.malloc(3);
		normB = condVals, dev_eps = condVals + 1, eps2b = condVals + 2;
		scalars.setZero(3);

		unsigned grid(memlen / blockSize), block(blockSize);

		cudaStreamCreate(&stream);
		cudaGraphCreate(&mainGraph, 0);

		cudaGraphNode_t memsetNormNode, preSolveNode, isNormZeroNode, ifNode, xNormNode, lastNode;
		if (preconditioning) {
			if (updatingMatrix) {
				cudaAddMemsetNode(normB, 0, sizeof(fp), memsetNormNode, mainGraph, {});
				void* precondArgs[] = { &N, &data, &rows, &DR, &rhs, &normB };
				cudaAddKernelNode((sizeof(fp) == 4 ? (void*)precondAndNorm_f : (void*)precondAndNorm_d), \
					grid, block, 0, precondArgs, \
					preSolveNode, mainGraph, { memsetNormNode });
			}
			else {
				cudaAddMemsetNode(normB, 0, sizeof(fp), memsetNormNode, mainGraph, {});
				void* normArgs[] = { &rhs, &DR, &normB };
				//void* normArgs[] = { &rhs, &normB };
				cudaAddKernelNode((sizeof(fp) == 4 ? (void*)norm2P_f : (void*)norm2P_d), \
					grid, block, 0, normArgs, \
					preSolveNode, mainGraph, { memsetNormNode });
				if constexpr (sizeof(fp) == 4) precond_f<<<grid, block, 0, stream>>>(N, data, rows, DR);
				else precond_d<<<grid, block, 0, stream>>>(N, data, rows, DR);
				cudaStreamSynchronize(stream);
			}
			//graph renormP
			cudaGraphConditionalHandle ifHandle;
			cudaGraphConditionalHandleCreate(&ifHandle, mainGraph);

			void* isZeroArgs[] = { &ifHandle, &normB };
			cudaAddKernelNode((sizeof(fp) == 4 ? (void*)isCloseToZero_f : (void*)isCloseToZero_d), \
				1, 1, 0, isZeroArgs, \
				isNormZeroNode, mainGraph, { preSolveNode });

			cudaGraphNodeParams ifParams = { cudaGraphNodeTypeConditional };
			ifParams.conditional.handle = ifHandle;
			ifParams.conditional.type = cudaGraphCondTypeIf;
			ifParams.conditional.size = 1;
#if CUDART_VERSION >= 13000
			cudaGraphAddNode(&ifNode, mainGraph, &isNormZeroNode, nullptr, 1, &ifParams);
#else
			cudaGraphAddNode(&ifNode, mainGraph, &isNormZeroNode, 1, &ifParams);
#endif
			renormGraph = ifParams.conditional.phGraph_out[0];
			void* xNormArgs[] = { &x, &DR, &normB };
			cudaAddKernelNode((sizeof(fp) == 4 ? (void*)norm2P_f : (void*)norm2P_d), \
				grid, block, 0, xNormArgs, \
				xNormNode, renormGraph, {});
			lastNode = ifNode;
		}
		else {
			if (updatingRhs) {
				cudaAddMemsetNode(normB, 0, sizeof(fp), memsetNormNode, mainGraph, {});
				void* normArgs[] = { &rhs, &normB };
				cudaAddKernelNode((sizeof(fp) == 4 ? (void*)norm2_f : (void*)norm2_d), \
					grid, block, 0, normArgs, \
					preSolveNode, mainGraph, { memsetNormNode });
				//graph renorm
				cudaGraphConditionalHandle ifHandle;
				cudaGraphConditionalHandleCreate(&ifHandle, mainGraph);

				void* isZeroArgs[] = { &ifHandle, &normB };
				cudaAddKernelNode((sizeof(fp) == 4 ? (void*)isCloseToZero_f : (void*)isCloseToZero_d), \
					1, 1, 0, isZeroArgs, \
					isNormZeroNode, mainGraph, { preSolveNode });

				cudaGraphNodeParams ifParams = { cudaGraphNodeTypeConditional };
				ifParams.conditional.handle = ifHandle;
				ifParams.conditional.type = cudaGraphCondTypeIf;
				ifParams.conditional.size = 1;
#if CUDART_VERSION >= 13000
				cudaGraphAddNode(&ifNode, mainGraph, &isNormZeroNode, nullptr, 1, &ifParams);
#else
				cudaGraphAddNode(&ifNode, mainGraph, &isNormZeroNode, 1, &ifParams);
#endif
				renormGraph = ifParams.conditional.phGraph_out[0];
				void* xNormArgs[] = { &x, &normB };
				cudaAddKernelNode((sizeof(fp) == 4 ? (void*)norm2_f : (void*)norm2_d), \
					grid, block, 0, xNormArgs, \
					xNormNode, renormGraph, {});
				lastNode = ifNode;
			}
			else {
				cudaGraphAddEmptyNode(&preSolveNode, mainGraph, nullptr, 0);
				cudaMemset(normB, 0, sizeof(fp));
				if constexpr (sizeof(fp) == 4) {
					norm2_f<<<grid, block, 0, stream>>>(rhs, normB);
					cudaStreamSynchronize(stream);
					if (condVals.getItem(0) < 1e-35f)
						norm2_f<<<grid, block, 0, stream>>>(x, normB);
				}
				else {
					norm2_d<<<grid, block, 0, stream>>>(rhs, normB);
					cudaStreamSynchronize(stream);
					if (condVals.getItem(0) < 1e-200)
						norm2_d<<<grid, block, 0, stream>>>(x, normB);
				}
				lastNode = preSolveNode;
			}
		}

		cudaGraphNode_t updateEpsNode;
		void* updateEpsArgs[] = { &condVals };
		cudaAddKernelNode((sizeof(fp) == 4 ? (void*)updateEps_f : (void*)updateEps_d), \
			1, 1, 0, updateEpsArgs, \
			updateEpsNode, mainGraph, { lastNode });

		const unsigned linesBlock = 32;
		unsigned linesGrid = (lineCount + linesBlock - 1) / linesBlock;

		cudaGraphNode_t initNode, initLinesNode, initScalarNode;
		if (lineCount) {
			if (preconditioning) {
				void* initArgs[] = { &data, &rows, &cols, &rhs, &x, &r, &z, &q, &DR, &mask };
				cudaAddKernelNode((sizeof(fp) == 4 ? (void*)cgInitLinesP_f : (void*)cgInitLinesP_d), \
					grid, block, 0, initArgs, \
					initNode, mainGraph, { lastNode });
				void* linesArgs[] = { &lineCount, &r, &q, &lines, &lineRows };
				cudaAddKernelNode((sizeof(fp) == 4 ? (void*)linesP2_f : (void*)linesP2_d), \
					linesGrid, linesBlock, 0, linesArgs, \
					initLinesNode, mainGraph, { initNode });
				void* scalarArgs[] = {&q, &r, &rhoNext};
				cudaAddKernelNode((sizeof(fp) == 4 ? (void*)dotProduct_f : (void*)dotProduct_d), \
					grid, block, 0, scalarArgs, \
					initScalarNode, mainGraph, { initLinesNode });
			}
			else {
				void* initArgs[] = { &data, &rows, &cols, &rhs, &x, &r, &z, &mask };
				cudaAddKernelNode((sizeof(fp) == 4 ? (void*)cgInitLines_f : (void*)cgInitLines_d), \
					grid, block, 0, initArgs, \
					initNode, mainGraph, { lastNode });
				void* linesArgs[] = { &lineCount, &r, &lines, &lineRows };
				cudaAddKernelNode((sizeof(fp) == 4 ? (void*)linesV2_f : (void*)linesV2_d), \
					linesGrid, linesBlock, 0, linesArgs, \
					initLinesNode, mainGraph, { initNode });
				void* scalarArgs[] = {&r, &rhoNext};
				cudaAddKernelNode((sizeof(fp) == 4 ? (void*)norm2_f : (void*)norm2_d), \
					grid, block, 0, scalarArgs, \
					initScalarNode, mainGraph, { initLinesNode });
			}
			lastNode = initScalarNode;
		}
		else {
			if (preconditioning) {
				void* initArgs[] = { &data, &rows, &cols, &rhs, &x, &r, &z, &q, &DR, &rhoNext, &mask };
				cudaAddKernelNode((sizeof(fp) == 4 ? (void*)cgInitP_f : (void*)cgInitP_d), \
					grid, block, 0, initArgs, \
					initNode, mainGraph, { lastNode });
			}
			else {
				void* initArgs[] = { &data, &rows, &cols, &rhs, &x, &r, &z, &rhoNext, &mask };
				cudaAddKernelNode((sizeof(fp) == 4 ? (void*)cgInit_f : (void*)cgInit_d), \
					grid, block, 0, initArgs, \
					initNode, mainGraph, { lastNode });
			}
			lastNode = initNode;
		}

		cudaGraphConditionalHandle handle;
		cudaGraphConditionalHandleCreate(&handle, mainGraph);
		cudaGraphNode_t conditionNode;
		void* condArgs[] = { &handle, &rhoNext, &eps2b, &dev_iters, &N };
		cudaAddKernelNode((sizeof(fp) == 4 ? (void*)loopCondition_f : (void*)loopCondition_d), \
			1, 1, 0, condArgs, \
			conditionNode, mainGraph, { lastNode, updateEpsNode });

		cudaGraphNode_t loopNode;
		cudaGraphNodeParams loopParams = { cudaGraphNodeTypeConditional };
		loopParams.conditional.handle = handle;
		loopParams.conditional.type = cudaGraphCondTypeWhile;
		loopParams.conditional.size = 1;
#if CUDART_VERSION >= 13000
		cudaGraphAddNode(&loopNode, mainGraph, &conditionNode, nullptr, 1, &loopParams);
#else
		cudaGraphAddNode(&loopNode, mainGraph, &conditionNode, 1, &loopParams);
#endif
		bodyGraph = loopParams.conditional.phGraph_out[0];

		cudaGraphNode_t cg1Node;
		void* cg1Args[] = { (preconditioning ? &q : &r), &z, &rhoPrev, &rhoNext };
		cudaAddKernelNode((sizeof(fp) == 4 ? (void*)cg1v2_f : (void*)cg1v2_d), \
			grid, block / 2, 0, cg1Args, \
			cg1Node, bodyGraph, {});

		cudaGraphNode_t rhoOnDevNode;
		cudaGraphAddMemcpyNode1D(&rhoOnDevNode, bodyGraph, { &cg1Node }, 1, rhoPrev, rhoNext, sizeof(fp), cudaMemcpyDeviceToDevice);

		cudaGraphNode_t memsetNode;
		cudaAddMemsetNode(scalars, 0, 2 * sizeof(fp), \
			memsetNode, bodyGraph, { rhoOnDevNode });

		cudaGraphNode_t cg2Node;
		void* cg2Args[] = { &data, &rows, &cols, &z, &s, &omega };
		cudaAddKernelNode((sizeof(fp) == 4 ? (void*)cg2_fv : (void*)cg2_dv), \
			grid, block, 0, cg2Args, \
			cg2Node, bodyGraph, { memsetNode });

		cudaGraphNode_t cg3Node, cg3LinesNode, cg3ScalarNode;
		if (lineCount) {
			if (preconditioning) {
				void* cg3Args[] = { &x, &r, &z, &s, &q, &DR, &scalars, &mask };
				cudaAddKernelNode((sizeof(fp) == 4 ? (void*)cg3LinesP_f : (void*)cg3LinesP_d), \
					grid, block, 0, cg3Args, \
					cg3Node, bodyGraph, { cg2Node });
				void* linesArgs[] = { &lineCount, &r, &q, &lines, &lineRows };
				cudaAddKernelNode((sizeof(fp) == 4 ? (void*)linesP2_f : (void*)linesP2_d), \
					linesGrid, linesBlock, 0, linesArgs, \
					cg3LinesNode, bodyGraph, { cg3Node });
				void* scalarArgs[] = {&q, &r, &rhoNext};
				cudaAddKernelNode((sizeof(fp) == 4 ? (void*)dotProduct_f : (void*)dotProduct_d), \
					grid, block, 0, scalarArgs, \
					cg3ScalarNode, bodyGraph, { cg3LinesNode });
			}
			else {
				void* cg3Args[] = { &x, &r, &z, &s, &scalars, &mask };
				cudaAddKernelNode((sizeof(fp) == 4 ? (void*)cg3Lines_f : (void*)cg3Lines_d), \
					grid, block, 0, cg3Args, \
					cg3Node, bodyGraph, { cg2Node });
				void* linesArgs[] = { &lineCount, &r, &lines, &lineRows };
				cudaAddKernelNode((sizeof(fp) == 4 ? (void*)linesV2_f : (void*)linesV2_d), \
					linesGrid, linesBlock, 0, linesArgs, \
					cg3LinesNode, bodyGraph, { cg3Node });
				void* scalarArgs[] = {&r, &rhoNext};
				cudaAddKernelNode((sizeof(fp) == 4 ? (void*)norm2_f : (void*)norm2_d), \
					grid, block, 0, scalarArgs, \
					cg3ScalarNode, bodyGraph, { cg3LinesNode });
			}
			lastNode = cg3ScalarNode;
		}
		else {
			if (preconditioning) {
				void* cg3Args[] = { &x, &r, &z, &s, &q, &DR, &scalars, &mask };
				cudaAddKernelNode((sizeof(fp) == 4 ? (void*)cg3P2_f : (void*)cg3P2_d), \
					grid, block / 2, 0, cg3Args, \
					cg3Node, bodyGraph, { cg2Node });
			}
			else {
				void* cg3Args[] = { &x, &r, &z, &s, &scalars, &mask };
				cudaAddKernelNode((sizeof(fp) == 4 ? (void*)cg3v2_f : (void*)cg3v2_d), \
					grid, block / 2, 0, cg3Args, \
					cg3Node, bodyGraph, { cg2Node });
			}
			lastNode = cg3Node;
		}

		//printf("%s\n", cudaGetErrorString(cudaGetLastError()));

		cudaAddKernelNode((sizeof(fp) == 4 ? (void*)loopCondition_f : (void*)loopCondition_d), \
			1, 1, 0, condArgs, \
			conditionNode, bodyGraph, { lastNode });
		
		cudaGraphInstantiate(&execGraph, mainGraph, nullptr, nullptr, 0);
		cudaStreamSynchronize(stream);
	}


	~CudaJCGV() {
		cudaGraphExecDestroy(execGraph);
		cudaGraphDestroy(mainGraph);
		cudaGraphDestroy(bodyGraph);
		if (renormGraph)
			cudaGraphDestroy(renormGraph);
		renormGraph = {};
		cudaStreamDestroy(stream);
	}


	// Метод решения с графом CUDA
	unsigned solve(fp eps) {
		cudaMemcpy(dev_eps, &eps, sizeof(fp), cudaMemcpyHostToDevice);
		scalars.setItem(2, fp(1.));
		dev_iters.setZero(1);

		cudaGraphLaunch(execGraph, stream);
		cudaStreamSynchronize(stream);

		unsigned host_iters;
		cudaMemcpy(&host_iters, dev_iters, sizeof(unsigned), cudaMemcpyDeviceToHost);
		return host_iters - 1;
	}


	// Метод решения без графа CUDA
	unsigned solveDebug(fp eps) {
		cudaMemcpy(dev_eps, &eps, sizeof(fp), cudaMemcpyHostToDevice);
		scalars.setItem(2, fp(1.));
		dev_iters.setZero(1);

		fp host_rho = {}, host_normB = {};
		unsigned host_iters = 0;

		const unsigned blockSize = 64;
		unsigned memlen = (N + blockSize - 1) / blockSize * blockSize;
		unsigned grid(memlen / blockSize), block(blockSize);

		cudaMemset(normB, 0, sizeof(fp));
		if constexpr (sizeof(fp) == 4) {
			norm2_f<<<grid, block, 0, stream>>>(rhs, normB);
			cudaStreamSynchronize(stream);
			if (condVals.getItem(0) < 1e-35f)
				norm2_f<<<grid, block, 0, stream>>>(x, normB);
		}
		else {
			norm2_d<<<grid, block, 0, stream>>>(rhs, normB);
			cudaStreamSynchronize(stream);
			if (condVals.getItem(0) < 1e-200)
				norm2_d<<<grid, block, 0, stream>>>(x, normB);
		}
		cudaStreamSynchronize(stream);
		cudaMemcpy(&host_normB, normB, sizeof(fp), cudaMemcpyDeviceToHost);

		eps = eps * eps * host_normB;
	
		if constexpr (sizeof(fp) == 4) updateEps_f<<<1, 1, 0, stream>>>(condVals);
		else updateEps_d<<<1, 1, 0, stream>>>(condVals);
		cudaStreamSynchronize(stream);

		if constexpr (sizeof(fp) == 4) cgInit_f<<<grid, block, 0, stream>>>(data, (int*)rows, (int*)cols, rhs, x, r, z, rhoNext, mask);
		else cgInit_d<<<grid, block, 0, stream>>>(data, (int*)rows, (int*)cols, rhs, x, r, z, rhoNext, mask);
		cudaStreamSynchronize(stream);
		cudaMemcpy(&host_rho, rhoNext, sizeof(fp), cudaMemcpyDeviceToHost);
		
		while (host_rho > eps && host_iters < N) {

			//std::cout << host_iters << ": " << host_rho << "\n";

			if constexpr (sizeof(fp) == 4) cg1_f<<<grid, block, 0, stream>>>(r, z, rhoPrev, rhoNext);
			else cg1_d<<<grid, block, 0, stream>>>(r, z, rhoPrev, rhoNext);
			cudaStreamSynchronize(stream);

			cudaMemcpy(rhoPrev, rhoNext, sizeof(fp), cudaMemcpyDeviceToDevice);
			cudaMemset(scalars, 0, 2 * sizeof(fp));

			if constexpr (sizeof(fp) == 4) cg2_f<<<grid, block, 0, stream>>>(data, (int*)rows, (int*)cols, z, s, omega);
			else cg2_d<<<grid, block, 0, stream>>>(data, (int*)rows, (int*)cols, z, s, omega);
			cudaStreamSynchronize(stream);

			if constexpr (sizeof(fp) == 4) cg3_f<<<grid, block, 0, stream>>>(x, r, z, s, scalars, mask);
			else cg3_d<<<grid, block, 0, stream>>>(x, r, z, s, scalars, mask);
			cudaStreamSynchronize(stream);

			cudaMemcpy(&host_rho, rhoNext, sizeof(fp), cudaMemcpyDeviceToHost);

			++host_iters;
		}

		//cudaMemcpy(&host_iters, dev_iters.data(), sizeof(unsigned), cudaMemcpyDeviceToHost);
		return host_iters;
	}

};