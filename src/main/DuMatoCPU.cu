#include <cuda_runtime.h>
#include <thread>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <bits/stdc++.h>
#include "DuMatoCPU.h"
#include "Timer.h"

using namespace std;

DuMatoCPU::DuMatoCPU(const char *datasetName, int k, int numberOfActiveThreads, int blockSize, int numberOfSMs, int jobsPerWarp, void (*kernel)(DataGPU*), int globalThreshold, bool relabeling, bool patternAware) {
    this->datasetName = datasetName;
    graphReader = new Graph(datasetName);
    dataCPU = new DataCPU;
    dataCPU->h_k = k;
    dataCPU->h_numberOfActiveThreads = numberOfActiveThreads;
    dataCPU->h_blockSize = blockSize;
    dataCPU->h_numberOfSMs = numberOfSMs;
    dataCPU->h_jobsPerWarp = jobsPerWarp;
    dataCPU->h_globalThreshold = globalThreshold;
    dataCPU->h_relabeling = relabeling;
    dataGPU = new DataGPU;
    this->kernel = kernel;
    this->patternAware = patternAware;

    initializeCpuDataStructures();
    initializeGpuDataStructures();
}

DuMatoCPU::DuMatoCPU(Graph* graphReader, int k, int numberOfActiveThreads, int blockSize, int numberOfSMs, int jobsPerWarp, void (*kernel)(DataGPU*), int globalThreshold, bool relabeling, bool patternAware) {
    this->graphReader = graphReader;
    dataCPU = new DataCPU;
    dataCPU->h_k = k;
    dataCPU->h_numberOfActiveThreads = numberOfActiveThreads;
    dataCPU->h_blockSize = blockSize;
    dataCPU->h_numberOfSMs = numberOfSMs;
    dataCPU->h_jobsPerWarp = jobsPerWarp;
    dataCPU->h_globalThreshold = globalThreshold;
    dataCPU->h_relabeling = relabeling;
    dataGPU = new DataGPU;
    this->kernel = kernel;
    this->patternAware = patternAware;

    initializeCpuDataStructures();
    initializeGpuDataStructures();
}

void DuMatoCPU::initializeCpuDataStructures() {
    /******************************************************************************************************/
    /*********************************************Constants************************************************/    
    dataCPU->h_maxVertexId = graphReader->getMaxVertexId();
    dataCPU->h_numberOfEdges = graphReader->getNumberOfEdges();
    dataCPU->h_maxDegree = graphReader->getMaxDegree();
    dataCPU->h_maxDegreeRounded = dataCPU->h_maxDegree < 32 ? 32 : dataCPU->h_maxDegree;

    dataCPU->h_warpSize = 32;
    dataCPU->h_numberOfBlocks = ceil(dataCPU->h_numberOfActiveThreads/(float)dataCPU->h_blockSize);
    dataCPU->h_warpsPerBlock = dataCPU->h_blockSize / dataCPU->h_warpSize;
    dataCPU->h_numberOfWarps = dataCPU->h_numberOfBlocks * dataCPU->h_warpsPerBlock;

    dataCPU->h_initialJobsPerWarp = ceil((dataCPU->h_maxVertexId+1)/(float)dataCPU->h_numberOfWarps);    
    dataCPU->h_theoreticalJobsPerWarp = std::max(dataCPU->h_initialJobsPerWarp, dataCPU->h_jobsPerWarp);

    dataCPU->h_extensionsLength = 0;
    dataCPU->h_extensionsOffset = new int[dataCPU->h_k];
    dataCPU->h_extensionsOffset[0] = 0;

    if(patternAware) {
        // Pattern-aware
        for(int k = 1, length ; k <= dataCPU->h_k - 1 ; k++) {
            length = (int)ceilf(pow(2, ceilf(log2((float)(1 * dataCPU->h_maxDegreeRounded)))));
            dataCPU->h_extensionsLength += length;
            if(k < dataCPU->h_k - 1)
                dataCPU->h_extensionsOffset[k] = dataCPU->h_extensionsLength;
        }
    }
    else {
        // Pattern-oblivious
        for(int k = 1, length ; k <= dataCPU->h_k - 1 ; k++) {
            length = (int)ceilf(pow(2, ceilf(log2((float)(k * dataCPU->h_maxDegreeRounded)))));
            dataCPU->h_extensionsLength += length;
            if(k < dataCPU->h_k - 1)
                dataCPU->h_extensionsOffset[k] = dataCPU->h_extensionsLength;
        }
    }
    
    
    /******************************************************************************************************/

    /******************************************************************************************************/
    /*******************************************Others*****************************************************/
    dataCPU->h_vertexOffset = new int[dataCPU->h_maxVertexId+2];
    dataCPU->h_adjacencyList = new int[dataCPU->h_numberOfEdges*2 + (dataCPU->h_maxVertexId+1)];
    dataCPU->h_degree = new int[dataCPU->h_maxVertexId+1];

    // Initialize CSR graph data structures for GPU
    int offset = 0;
    for(int vertexId = 0 ; vertexId <= graphReader->getMaxVertexId() ; vertexId++) {
        dataCPU->h_vertexOffset[vertexId] = offset;
        for(std::set<int>::iterator itEdges = graphReader->getNeighbours(vertexId)->begin() ; itEdges != graphReader->getNeighbours(vertexId)->end() ; itEdges++)
            dataCPU->h_adjacencyList[offset++] = *itEdges;
        dataCPU->h_adjacencyList[offset++] = -1;

        dataCPU->h_degree[vertexId] = graphReader->getNeighbours(vertexId)->size();
    }
    dataCPU->h_vertexOffset[graphReader->getMaxVertexId()+1] = dataCPU->h_vertexOffset[graphReader->getMaxVertexId()]+dataCPU->h_degree[graphReader->getMaxVertexId()]+1;

    dataCPU->h_result = new unsigned long[dataCPU->h_numberOfWarps];
    dataCPU->h_currentJob = new int[dataCPU->h_numberOfWarps];
    dataCPU->h_currentPosOfJob = new int[dataCPU->h_numberOfWarps * dataCPU->h_theoreticalJobsPerWarp];
    dataCPU->h_currentPos = new int[dataCPU->h_numberOfWarps];
    dataCPU->h_validJobs = new int[dataCPU->h_numberOfWarps];
    dataCPU->h_jobs = new int[dataCPU->h_numberOfWarps * dataCPU->h_theoreticalJobsPerWarp * dataCPU->h_warpSize];
    dataCPU->h_id = new int[dataCPU->h_numberOfWarps * dataCPU->h_warpSize];
    dataCPU->h_numberOfExtensions = new int[dataCPU->h_numberOfWarps * dataCPU->h_warpSize];
    dataCPU->h_numberOfExtensionsFixed = new int[dataCPU->h_numberOfWarps * dataCPU->h_warpSize];
    dataCPU->h_extensions = new int[dataCPU->h_numberOfWarps * dataCPU->h_extensionsLength];

    for(int i = 0 ; i < dataCPU->h_numberOfWarps ; i++) {
        dataCPU->h_currentPos[i] = -1;
        dataCPU->h_currentJob[i] = 0;
        dataCPU->h_validJobs[i] = 0;
    }
    for(int round = 0 ; round < dataCPU->h_initialJobsPerWarp ; round++) {
        for(int i = 0 ; i < dataCPU->h_numberOfWarps ; i++) {
            int jobId = round*dataCPU->h_numberOfWarps+i;
            if(jobId <= dataCPU->h_maxVertexId) {
                dataCPU->h_validJobs[i]++;
                dataCPU->h_jobs[i*dataCPU->h_theoreticalJobsPerWarp*dataCPU->h_warpSize + round*dataCPU->h_warpSize + 0] = jobId;
                dataCPU->h_currentPosOfJob[i*dataCPU->h_theoreticalJobsPerWarp+round] = 0;
            } 
        }
    }

    gpuErrorCheck(cudaMallocHost((void**)&dataCPU->h_stop, sizeof(bool)));
    gpuErrorCheck(cudaMallocHost((void**)&dataCPU->h_status, dataCPU->h_numberOfWarps * sizeof(int)));
    *(dataCPU->h_stop) = false;

    for(int i = 0 ; i < dataCPU->h_numberOfWarps ; i++)
        dataCPU->h_status[i] = 2;

    dataCPU->h_resultCounter = 0;

    if(dataCPU->h_relabeling) {
        quickMapping = new QuickMapping(dataCPU->h_k);
        dataCPU->h_hashPerWarp = (unsigned long long*)malloc(dataCPU->h_numberOfWarps*quickMapping->numberOfCgs * sizeof(unsigned long long));
        dataCPU->h_hashGlobal = (long unsigned int*)malloc(quickMapping->numberOfCgs * sizeof(long unsigned int));
        dataCPU->h_localSubgraphInduction = (long unsigned int*)malloc(dataCPU->h_numberOfWarps * dataCPU->h_warpSize * sizeof(long unsigned int));
        dataCPU->h_inductions = (int*)malloc(dataCPU->h_numberOfWarps * dataCPU->h_theoreticalJobsPerWarp * dataCPU->h_warpSize * sizeof(int));

        for(int i = 0 ; i < dataCPU->h_numberOfWarps ; i++) {
            for(int j = 0 ; j < quickMapping->numberOfCgs ; j++) {
                dataCPU->h_hashPerWarp[i*quickMapping->numberOfCgs+j] = 0;
            }
        }

        for(int i = 0 ; i < quickMapping->numberOfCgs ; i++)
            dataCPU->h_hashGlobal[i] = 0;

        memset(dataCPU->h_localSubgraphInduction, 0, dataCPU->h_numberOfWarps * dataCPU->h_warpSize * sizeof(long unsigned int));
    }
    /******************************************************************************************************/
}

void DuMatoCPU::releaseCpuDataStructures() {
    /****************************************/
    /***************Graph********************/
    // delete[] dataCPU->h_vertexOffset;
    // delete[] dataCPU->h_adjacencyList;
    // delete[] dataCPU->h_degree;
    /****************************************/

    /****************************************/
    /*************Enumeration****************/
    // delete[] dataCPU->h_result;
    // delete[] dataCPU->h_jobs;
    // delete[] dataCPU->h_currentJob;
    // delete[] dataCPU->h_currentPosOfJob;
    // delete[] dataCPU->h_validJobs;
    // delete[] dataCPU->h_currentPos;
    // delete[] dataCPU->h_extensionsOffset;
    // delete[] dataCPU->h_id;
    // delete[] dataCPU->h_extensions;
    // delete[] dataCPU->h_numberOfExtensions;
    // delete[] dataCPU->h_numberOfExtensionsFixed;
    cudaFreeHost(dataCPU->h_stop);
    cudaFreeHost(dataCPU->h_status);
    
    if(dataCPU->h_relabeling) {
        delete quickMapping;
        //free(dataCPU->h_hashPerWarp);
        //free(dataCPU->h_hashGlobal);
        //free(dataCPU->h_localSubgraphInduction);
        //free(dataCPU->h_inductions);
    }
    /****************************************/

    // delete graphReader;
    // delete dataCPU;
}

void DuMatoCPU::initializeGpuDataStructures() {
    /************ Graph related ************/
    gpuErrorCheck(cudaMalloc((void**)&dataGPU->d_vertexOffset, (dataCPU->h_maxVertexId+2)*sizeof(int)));
    gpuErrorCheck(cudaMalloc((void**)&dataGPU->d_adjacencyList, (dataCPU->h_numberOfEdges*2 + (dataCPU->h_maxVertexId+1)) * sizeof(int)));
    gpuErrorCheck(cudaMalloc((void**)&dataGPU->d_degree, (dataCPU->h_maxVertexId+1)*sizeof(int)));

    gpuErrorCheck(cudaMemcpy(dataGPU->d_vertexOffset, dataCPU->h_vertexOffset, (dataCPU->h_maxVertexId+2)*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(dataGPU->d_adjacencyList, dataCPU->h_adjacencyList, (dataCPU->h_numberOfEdges*2 + (dataCPU->h_maxVertexId+1)) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(dataGPU->d_degree, dataCPU->h_degree, (dataCPU->h_maxVertexId+1)*sizeof(int), cudaMemcpyHostToDevice));
    /***************************************/

    /***************************************/
    /******** Enumeration related **********/
    gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_k), sizeof(int)));
    gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_extensionsLength), sizeof(int)));
    gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_id), dataCPU->h_numberOfWarps * dataCPU->h_warpSize * sizeof(int)));
    gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_jobsPerWarp), sizeof(int)));
    gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_jobs), dataCPU->h_numberOfWarps * dataCPU->h_theoreticalJobsPerWarp * dataCPU->h_warpSize * sizeof(int)));
    gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_currentJob), dataCPU->h_numberOfWarps * sizeof(int)));
    gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_currentPosOfJob), dataCPU->h_numberOfWarps * dataCPU->h_theoreticalJobsPerWarp * sizeof(int)));
    gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_validJobs), dataCPU->h_numberOfWarps * sizeof(int)));
    gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_numberOfExtensions), dataCPU->h_numberOfWarps * dataCPU->h_warpSize * sizeof(int)));
    gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_numberOfExtensionsFixed), dataCPU->h_numberOfWarps * dataCPU->h_warpSize * sizeof(int)));
    gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_extensions), dataCPU->h_numberOfWarps * dataCPU->h_extensionsLength * sizeof(int)));
    gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_currentPos), dataCPU->h_numberOfWarps * sizeof(int)));
    gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_result), dataCPU->h_numberOfWarps * sizeof(unsigned long)));
    gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_warpSize), sizeof(int)));
    gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_extensionsOffset), (dataCPU->h_k-1) * sizeof(int)));
    gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_stop), sizeof(bool)));
    gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_status), dataCPU->h_numberOfWarps * sizeof(int)));
    gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_smid), dataCPU->h_numberOfWarps * sizeof(int)));
    gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_relabeling), sizeof(int)));
    if(dataCPU->h_relabeling) {
        gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_hashPerWarp), dataCPU->h_numberOfWarps * quickMapping->numberOfCgs * sizeof(unsigned long long)));
        gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_quickToCgLocal), quickMapping->numberOfQuicks * sizeof(long unsigned int)));
        gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_numberOfCgs), sizeof(int)));
        gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_extensionSources), dataCPU->h_numberOfWarps * dataCPU->h_extensionsLength * sizeof(int)));
        gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_localSubgraphInduction), dataCPU->h_numberOfWarps * dataCPU->h_warpSize * sizeof(long unsigned int)));
        gpuErrorCheck(cudaMalloc((void**)&(dataGPU->d_inductions), dataCPU->h_numberOfWarps * dataCPU->h_theoreticalJobsPerWarp * dataCPU->h_warpSize * sizeof(int)));
    }

    gpuErrorCheck(cudaMemcpy(dataGPU->d_jobsPerWarp, &(dataCPU->h_theoreticalJobsPerWarp), sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(dataGPU->d_jobs, dataCPU->h_jobs, dataCPU->h_numberOfWarps * dataCPU->h_theoreticalJobsPerWarp * dataCPU->h_warpSize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(dataGPU->d_currentJob, dataCPU->h_currentJob, dataCPU->h_numberOfWarps * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(dataGPU->d_validJobs, dataCPU->h_validJobs, dataCPU->h_numberOfWarps * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(dataGPU->d_currentPosOfJob, dataCPU->h_currentPosOfJob, dataCPU->h_numberOfWarps * dataCPU->h_theoreticalJobsPerWarp * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(dataGPU->d_warpSize, &(dataCPU->h_warpSize), sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(dataGPU->d_extensionsOffset, dataCPU->h_extensionsOffset, (dataCPU->h_k-1) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(dataGPU->d_k, &dataCPU->h_k, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(dataGPU->d_extensionsLength, &dataCPU->h_extensionsLength, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy((bool*)dataGPU->d_stop, dataCPU->h_stop, sizeof(bool), cudaMemcpyHostToDevice));    
    gpuErrorCheck(cudaMemcpy(dataGPU->d_status, dataCPU->h_status, dataCPU->h_numberOfWarps * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(dataGPU->d_currentPos, dataCPU->h_currentPos, dataCPU->h_numberOfWarps * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(dataGPU->d_relabeling, &dataCPU->h_relabeling, sizeof(int), cudaMemcpyHostToDevice));
    
    if(dataCPU->h_relabeling) {
        gpuErrorCheck(cudaMemcpy(dataGPU->d_quickToCgLocal, quickMapping->quickToCgLocal, quickMapping->numberOfQuicks * sizeof(long unsigned int), cudaMemcpyHostToDevice));
        gpuErrorCheck(cudaMemset(dataGPU->d_hashPerWarp, 0, dataCPU->h_numberOfWarps * quickMapping->numberOfCgs * sizeof(unsigned long long)));
        gpuErrorCheck(cudaMemcpy(dataGPU->d_numberOfCgs, &(quickMapping->numberOfCgs), sizeof(int), cudaMemcpyHostToDevice));
        gpuErrorCheck(cudaMemset(dataGPU->d_localSubgraphInduction, 0, dataCPU->h_numberOfWarps * dataCPU->h_warpSize * sizeof(long unsigned int)));
    }

    // Workaround to allow structs as an argument.
    gpuErrorCheck(cudaMalloc((void**)&d_dataGPU, sizeof(DataGPU)));
    gpuErrorCheck(cudaMemcpy(d_dataGPU, dataGPU, sizeof(DataGPU), cudaMemcpyHostToDevice));
    /***************************************/

    /***********/
    /* Streams */
    gpuErrorCheck(cudaStreamCreate(&main));
    gpuErrorCheck(cudaStreamCreate(&memory));
    /***********/
}

void DuMatoCPU::releaseGpuDataStructures() {
    cudaFree(dataGPU->d_vertexOffset);
    cudaFree(dataGPU->d_adjacencyList);
    cudaFree(dataGPU->d_degree); 
    cudaFree(dataGPU->d_k);
    cudaFree(dataGPU->d_extensionsLength);
    cudaFree(dataGPU->d_id);
    cudaFree(dataGPU->d_jobsPerWarp);
    cudaFree(dataGPU->d_jobs);
    cudaFree(dataGPU->d_currentJob);
    cudaFree(dataGPU->d_currentPosOfJob); 
    cudaFree(dataGPU->d_validJobs);
    cudaFree(dataGPU->d_numberOfExtensions);
    cudaFree(dataGPU->d_numberOfExtensionsFixed);
    cudaFree(dataGPU->d_extensions); 
    cudaFree(dataGPU->d_currentPos); 
    cudaFree(dataGPU->d_result);
    cudaFree(dataGPU->d_warpSize);
    cudaFree(dataGPU->d_extensionsOffset);
    cudaFree((bool*)dataGPU->d_stop);
    cudaFree(dataGPU->d_status);
    cudaFree(dataGPU->d_smid);
    cudaFree(dataGPU->d_relabeling);
    if(dataCPU->h_relabeling) {
        cudaFree(dataGPU->d_hashPerWarp);
        cudaFree(dataGPU->d_quickToCgLocal);
        cudaFree(dataGPU->d_numberOfCgs);
        cudaFree(dataGPU->d_extensionSources);
        cudaFree(dataGPU->d_localSubgraphInduction);
        cudaFree(dataGPU->d_inductions);
    }
    
    // delete dataGPU;
    // cudaFree(d_dataGPU);
    cudaStreamDestroy(main);
    cudaStreamDestroy(memory);
}

void DuMatoCPU::runKernel() {
    void* args[] = {&(d_dataGPU)};
    cudaLaunchKernel((void*)kernel, dim3(dataCPU->h_numberOfBlocks), dim3(dataCPU->h_blockSize), args, 0, main);    
}

void DuMatoCPU::waitKernel() {
    gpuErrorCheck(cudaStreamSynchronize(main));
    gpuErrorCheck(cudaPeekAtLastError());

    copyResult();
}

void DuMatoCPU::copyWarpDataFromGpu() {
    gpuErrorCheck(cudaMemcpy(dataCPU->h_id, dataGPU->d_id, dataCPU->h_numberOfWarps * dataCPU->h_warpSize * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaMemcpy(dataCPU->h_jobs, dataGPU->d_jobs, dataCPU->h_numberOfWarps * dataCPU->h_theoreticalJobsPerWarp * dataCPU->h_warpSize * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaMemcpy(dataCPU->h_currentJob, dataGPU->d_currentJob, dataCPU->h_numberOfWarps * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaMemcpy(dataCPU->h_currentPosOfJob, dataGPU->d_currentPosOfJob, dataCPU->h_numberOfWarps * dataCPU->h_theoreticalJobsPerWarp * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaMemcpy(dataCPU->h_validJobs, dataGPU->d_validJobs, dataCPU->h_numberOfWarps * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaMemcpy(dataCPU->h_numberOfExtensions, dataGPU->d_numberOfExtensions, dataCPU->h_numberOfWarps * dataCPU->h_warpSize * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaMemcpy(dataCPU->h_numberOfExtensionsFixed, dataGPU->d_numberOfExtensionsFixed, dataCPU->h_numberOfWarps * dataCPU->h_warpSize * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaMemcpy(dataCPU->h_currentPos, dataGPU->d_currentPos, dataCPU->h_numberOfWarps * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaMemcpy(dataCPU->h_extensions, dataGPU->d_extensions, dataCPU->h_numberOfWarps * dataCPU->h_extensionsLength * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrorCheck(cudaMemcpy(dataCPU->h_status, dataGPU->d_status, dataCPU->h_numberOfWarps*sizeof(int), cudaMemcpyDeviceToHost));
    if(dataCPU->h_relabeling) {
        gpuErrorCheck(cudaMemcpy(dataCPU->h_localSubgraphInduction, dataGPU->d_localSubgraphInduction, dataCPU->h_numberOfWarps * dataCPU->h_warpSize * sizeof(long unsigned int), cudaMemcpyDeviceToHost));
        // gpuErrorCheck(cudaMemcpy(dataCPU->h_inductions, dataGPU->d_inductions, dataCPU->h_numberOfWarps * dataCPU->h_theoreticalJobsPerWarp * dataCPU->h_warpSize * sizeof(int), cudaMemcpyDeviceToHost));
    }
}

void DuMatoCPU::copyWarpDataBackToGpu() {
    gpuErrorCheck(cudaMemcpy(dataGPU->d_jobs, dataCPU->h_jobs, dataCPU->h_numberOfWarps * dataCPU->h_theoreticalJobsPerWarp * dataCPU->h_warpSize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(dataGPU->d_currentJob, dataCPU->h_currentJob, dataCPU->h_numberOfWarps * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(dataGPU->d_currentPosOfJob, dataCPU->h_currentPosOfJob, dataCPU->h_numberOfWarps * dataCPU->h_theoreticalJobsPerWarp * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(dataGPU->d_validJobs, dataCPU->h_validJobs, dataCPU->h_numberOfWarps * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(dataGPU->d_numberOfExtensions, dataCPU->h_numberOfExtensions, dataCPU->h_numberOfWarps * dataCPU->h_warpSize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(dataGPU->d_numberOfExtensionsFixed, dataCPU->h_numberOfExtensionsFixed, dataCPU->h_numberOfWarps * dataCPU->h_warpSize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(dataGPU->d_extensions, dataCPU->h_extensions, dataCPU->h_numberOfWarps * dataCPU->h_extensionsLength * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrorCheck(cudaMemcpy(dataGPU->d_currentPos, dataCPU->h_currentPos, dataCPU->h_numberOfWarps * sizeof(int), cudaMemcpyHostToDevice));
    if(dataCPU->h_relabeling) {
        // gpuErrorCheck(cudaMemcpy(dataGPU->d_localSubgraphInduction, dataCPU->h_localSubgraphInduction, dataCPU->h_numberOfWarps * dataCPU->h_warpSize * sizeof(long unsigned int), cudaMemcpyHostToDevice));
        gpuErrorCheck(cudaMemcpy(dataGPU->d_inductions, dataCPU->h_inductions, dataCPU->h_numberOfWarps * dataCPU->h_theoreticalJobsPerWarp * dataCPU->h_warpSize * sizeof(int), cudaMemcpyHostToDevice));
    }
}

void DuMatoCPU::outputAggregateCounter() {
    printf("Result: %lu.\n", dataCPU->h_resultCounter);
}

void DuMatoCPU::outputAggregatePattern() {
    if(dataCPU->h_relabeling) {
        gpuErrorCheck(cudaMemcpy(dataCPU->h_hashPerWarp, dataGPU->d_hashPerWarp, dataCPU->h_numberOfWarps * quickMapping->numberOfCgs * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        for(int i = 0 ; i < dataCPU->h_numberOfWarps ; i++) {
            for(int j = 0 ; j < quickMapping->numberOfCgs ; j++) {
                dataCPU->h_hashGlobal[j] += dataCPU->h_hashPerWarp[i*quickMapping->numberOfCgs+j];
            }
        }

        long unsigned int validSubgraphs = 0;    
        unsigned counter = 0;
        for(int i = 0 ; i < quickMapping->numberOfCgs ; i++) {
            if(dataCPU->h_hashGlobal[i] > 0) {
                counter++;
                printf("%d, %lu\n", i, dataCPU->h_hashGlobal[i]);
                validSubgraphs += dataCPU->h_hashGlobal[i];
            }
        }
        printf("Aggregated: %lu, %u patterns.\n", validSubgraphs, counter);
    }
}

void DuMatoCPU::loadGpuOccupancyStatus() {
    gpuErrorCheck(cudaMemcpyAsync(dataCPU->h_status, dataGPU->d_status, dataCPU->h_numberOfWarps*sizeof(int), cudaMemcpyDeviceToHost, memory));
    gpuErrorCheck(cudaStreamSynchronize(memory));

    dataCPU->h_percentageWarpsIdle = 0;
    for(int i = 0 ; i < dataCPU->h_numberOfWarps ; i++)
        if(dataCPU->h_status[i] == 2) dataCPU->h_percentageWarpsIdle++;

    dataCPU->h_percentageWarpsIdle /= (float)dataCPU->h_numberOfWarps;
    dataCPU->h_percentageWarpsIdle *= 100;
}

void DuMatoCPU::sleepFor(int millisecs) {
    std::this_thread::sleep_for(std::chrono::milliseconds(millisecs));
}

bool DuMatoCPU::gpuIsIdle() {
    loadGpuOccupancyStatus();
    return dataCPU->h_percentageWarpsIdle >= 100 || dataCPU->h_percentageWarpsIdle > dataCPU->h_globalThreshold;
}

void DuMatoCPU::stopKernel() {
    *dataCPU->h_stop = true;
    gpuErrorCheck(cudaMemcpyAsync((bool*)dataGPU->d_stop, dataCPU->h_stop, sizeof(bool), cudaMemcpyHostToDevice, memory));
    gpuErrorCheck(cudaStreamSynchronize(memory));
    gpuErrorCheck(cudaStreamSynchronize(main));

    copyResult();

    *dataCPU->h_stop = false;
    gpuErrorCheck(cudaMemcpy((bool*)dataGPU->d_stop, dataCPU->h_stop, sizeof(bool), cudaMemcpyHostToDevice));   
}

void DuMatoCPU::copyResult() {
    gpuErrorCheck(cudaMemcpy(dataCPU->h_result, dataGPU->d_result, dataCPU->h_numberOfWarps*sizeof(unsigned long), cudaMemcpyDeviceToHost));
    for(int i = 0 ; i < dataCPU->h_numberOfWarps ; i++)
        dataCPU->h_resultCounter += dataCPU->h_result[i];
}

int DuMatoCPU::organizeThreadStatus(std::vector<int>* idles, std::priority_queue<Donator*, std::vector<Donator*>, Comparator>* actives, std::vector<int>* indifferents) {
    int totalWeight = 0;
    for(int wid = 0 ; wid < dataCPU->h_numberOfWarps ; wid++) {
        
        // TODO Update constructor to receive only the current data
        EnumerationHelper *helper = new EnumerationHelper(wid, dataCPU);

        // Idles for sure
        if(dataCPU->h_status[wid] == 2 && helper->jobQueueIsEmpty()) {
            // std::cout << "IDLE at all\n";
            idles->push_back(wid);
        }
        // Actives available for donation
        else if (dataCPU->h_status[wid] == 1) {
            Donator* donator = new Donator;
            int targetLevel = helper->getTargetLevel();
            if(targetLevel != -1) {
                donator->wid = wid;
                donator->targetLevel = targetLevel;
                donator->weight = helper->getWeight();
                totalWeight += donator->weight;
                actives->push(donator);
            }
            else {
                // Active, but contains few jobs
                indifferents->push_back(wid);
            }
        }
        // Actives because of the unpopped job queue
        else {
            indifferents->push_back(wid);
        }
        // helper->report();
        delete helper;
    }
    return totalWeight;
}

bool DuMatoCPU::donate(std::vector<int>* idles, std::priority_queue<Donator*, std::vector<Donator*>, Comparator>* actives, int totalWeight) {
    EnumerationHelper *donatorHelper, *idleHelper;
    Donator* donator;
    int currentIdle;

    int i = 0;
    float idealWeight = ceil((float)totalWeight / (actives->size()+idles->size()));
    int jobsPerWarp = idealWeight < dataCPU->h_jobsPerWarp ? idealWeight : dataCPU->h_jobsPerWarp;

    printf("jobsPerWarp: %d\n", jobsPerWarp);
    for(int job = 0 ; job < jobsPerWarp ; job++) {
        for(i = 0 ; i < idles->size() && !actives->empty() ; i++) {
            currentIdle = idles->at(i);
            donator = actives->top();
            actives->pop();
            
            donatorHelper = new EnumerationHelper(donator->wid, dataCPU);
            idleHelper = new EnumerationHelper(currentIdle, dataCPU);

            // printf("[BEFORE] Donator\n");
            // donatorHelper->report();
            // printf("[BEFORE] Idle\n");
            // idleHelper->report();

            if(job == 0) {
                idleHelper->setValidJobs(0);
                idleHelper->setCurrentJob(0);  
                idleHelper->setCurrentPos(-1);              
            }

            for(int k = 0 ; k <= donator->targetLevel ; k++) {
                idleHelper->setJob(job, k, donatorHelper->getId(k));
                if(dataCPU->h_relabeling)
                    idleHelper->setInductions(job, k, donatorHelper->getLocalSubgraphInduction(k));
            }
            idleHelper->setJob(job, donator->targetLevel+1, donatorHelper->popLastExtension(donator->targetLevel));
            idleHelper->setCurrentPosOfJob(job, donator->targetLevel+1);
            idleHelper->increaseJob();


            // printf("[AFTER] Donator\n");
            // donatorHelper->report();
            // printf("[AFTER] Idle\n");
            // idleHelper->report();

            // Remove donator or push back to the queue
            int targetLevel = donatorHelper->getTargetLevel();
            donator->weight--;
            if(targetLevel != -1 && donator->weight >= jobsPerWarp) {
                donator->targetLevel = targetLevel;
                actives->push(donator);
            }
            else {
                free(donator);
            }

            delete donatorHelper;
            delete idleHelper;
        }
    }
    

    // JOB COST SIMULATION
    // int j = 0;
    // while(j < 10) {
    //     for(i = 0 ; i < idles->size() && !actives->empty() ; i++) {
    //         currentIdle = idles->at(i);
    //         donator = actives->top();
    //         actives->pop();
    //         actives->push(donator);
    //     }
    //     j++;
    // }


    return i == idles->size();
}

void DuMatoCPU::invalidateResult() {
    gpuErrorCheck(cudaMemset(dataGPU->d_result, 0, dataCPU->h_numberOfWarps*sizeof(unsigned long)));
}

bool DuMatoCPU::rebalance() {
    copyWarpDataFromGpu();
    std::vector<int> idles, indifferents;
    std::priority_queue<Donator*, std::vector<Donator*>, Comparator> actives;
    int totalWeight = organizeThreadStatus(&idles, &actives, &indifferents);
    // printf("idles size: %lu, actives size: %lu, indifferents size: %lu\n", idles.size(), actives.size(), indifferents.size());
    if(actives.size() > 0) {
        bool full = donate(&idles, &actives, totalWeight);
        copyWarpDataBackToGpu();
        
        printf("[REBALANCING] full: %s, totalWeight: %d, #jobs/#idle_warps: %.2f\n", full ? "yes" : "no", totalWeight, (float)totalWeight/idles.size());
        return true;
    }
    else {
        invalidateResult();
        return false;
    }
}

void DuMatoCPU::validateAggregateCounter() {
    string validationFile = ""; 
    validationFile += datasetName;
    validationFile += ".";
    validationFile += to_string(dataCPU->h_k);
    validationFile += ".aggregateCounter";

    ifstream fp(validationFile);

    if (!fp.is_open()) {
        std::cerr << "Error opening the validation file of aggregate counter." << std::endl;
        return;
    }

    string line;
    long unsigned int peregrineCounter = 0;
    while (std::getline(fp, line)) {
        if(line.find("[") != string::npos) {
            stringstream check1(line);
            string intermediate;
            getline(check1, intermediate, ':');
            getline(check1, intermediate, ':');
            int value = atoi(intermediate.c_str());
            peregrineCounter += value;
        }
    }
    if(peregrineCounter == dataCPU->h_resultCounter)
        cout << "validateAggregateCounter: MATCH\n";
    else
        cout << "validateAggregateCounter: MISMATCH\n";
    
    fp.close();
}

void DuMatoCPU::validateAggregatePattern() {

    // Collecting Peregrine results file
    string validationFile = ""; 
    validationFile += datasetName;
    validationFile += ".";
    validationFile += to_string(dataCPU->h_k);
    validationFile += ".aggregatePattern";

    ifstream fp(validationFile);

    if (!fp.is_open()) {
        std::cerr << "Error opening the validation file of aggregate counter." << std::endl;
        return;
    }

    string line;
    vector<long int> aggregatePatternPeregrine;
    long int totalPeregrine = 0;
    while (std::getline(fp, line)) {
        if(line.find("[") != string::npos) {
            stringstream check1(line);
            string intermediate;
            getline(check1, intermediate, ':');
            getline(check1, intermediate, ':');
            long int value = stol(intermediate.c_str());
            if(value > 0) {
                totalPeregrine += value;
                aggregatePatternPeregrine.push_back(value);
            }
        }
    }
    sort(aggregatePatternPeregrine.begin(), aggregatePatternPeregrine.end());

    // Collecting DuMato results
    vector<long int> aggregatePatternDuMato;
    long int totalDuMato = 0;
    for(int i = 0 ; i < quickMapping->numberOfCgs ; i++) {
        if(dataCPU->h_hashGlobal[i] > 0) {
            aggregatePatternDuMato.push_back(dataCPU->h_hashGlobal[i]);
            totalDuMato += dataCPU->h_hashGlobal[i];
        }
    }
    sort(aggregatePatternDuMato.begin(), aggregatePatternDuMato.end());

    if(totalDuMato == totalPeregrine) {
        cout << "validateAggregatePattern (counting): MATCH\n";
        printf("\t Id \t DuMato \t Peregrine\n");
        for(int i = 0 ; i < aggregatePatternDuMato.size() ; i++) {
            printf("\t %d \t %lu \t %lu\n", i, aggregatePatternDuMato[i], aggregatePatternPeregrine[i]);
        }
    }
    else {
        cout << "validateAggregatePattern (counting): MISMATCH\n";
        cout << "DuMato: " << totalDuMato << "\n";
        cout << "Peregrine: " << totalPeregrine << "\n";
    }

    if(aggregatePatternDuMato == aggregatePatternPeregrine)
        cout << "validateAggregatePattern (per pattern): MATCH\n";
    else {
        cout << "validateAggregatePattern (per pattern): MISMATCH\n";
        cout << "# patterns (DuMato): " << aggregatePatternDuMato.size() << "\n";
        cout << "# patterns (Peregrine): " << aggregatePatternPeregrine.size() << "\n";
    }
    
    fp.close();
}

Graph* DuMatoCPU::getGraphReader() {
    return graphReader;
}

DuMatoCPU::~DuMatoCPU() {
    releaseCpuDataStructures();
    releaseGpuDataStructures();
}