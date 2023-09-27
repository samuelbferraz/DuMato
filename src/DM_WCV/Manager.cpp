#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <queue>
#include <vector>
#include <thread>
#include <unordered_map>
#include <ctime>
#include <algorithm>
#include <string>
#include <stdlib.h>
#include <fstream>
#include "Manager.h"
#include "Timer.h"
#include "CudaHelperFunctions.h"
#include "EnumerationHelper.h"
#define MAXN 10
#include "nauty.h"

Manager::Manager(const char* graphFile, int k, int numberOfActiveThreads, int blockSize, void (*kernel)(Device*), int numberOfSMs, int reportInterval, int jobsPerWarp, int induce) {
    this->h_k = k;

    if(induce)
        this->quickMapping = new QuickMapping(k);
    else
        this->quickMapping = new QuickMapping(-1);

    this->graphFile = graphFile;
    this->mainGraph = new Graph(graphFile);
    this->h_numberOfActiveThreads = numberOfActiveThreads;
    this->blockSize = blockSize;
    this->result = 0;
    this->amountNewVertices = 0;
    this->removedEdges = 0;
    this->addedEdges = 0;
    this->activeThreads = 0;
    this->percentageWarpsIdle = 0;
    this->kernel = kernel;
    this->timer = new Timer();
    this->gpuFinished = false;
    this->numberOfSMs = numberOfSMs;
    this->smOccupancy = new int[numberOfSMs];
    this->currentRound = 0;
    this->reportInterval = reportInterval;
    this->h_induce = induce;
    this->h_subgraphsProcessed = 0;
    

    quickToCgMap = new std::unordered_map<unsigned int,unsigned int>();
    cgs = new std::set<unsigned int>();

    h_maxVertexId = mainGraph->getMaxVertexId();
    h_maxDegree = mainGraph->getMaxDegree();

    h_warpSize = 32;
 
    numberOfBlocks = ceil(h_numberOfActiveThreads/(float)blockSize);
    warpsPerBlock = blockSize / h_warpSize;
    numberOfWarps = numberOfBlocks * warpsPerBlock;

    this->device = (Device*)malloc(sizeof(Device));
    this->h_jobsPerWarp = jobsPerWarp;
    this->h_initialJobsPerWarp = ceil((h_maxVertexId+1)/(float)numberOfWarps);
    this->h_theoreticalJobsPerWarp = std::max(h_initialJobsPerWarp, h_jobsPerWarp);

    if(h_initialJobsPerWarp > h_jobsPerWarp) {
        printf("******************WARNING******************\n");
        printf("Initial jobs per warp higher than the jobs used during rebalancing.\n");
        printf("Initial jobs per warp: %d, jobs per warp during rebalancing: %d.\n", h_initialJobsPerWarp, h_jobsPerWarp);
        printf("*******************************************\n");
    }

    gpuErrchk(cudaStreamCreate(&main));
    gpuErrchk(cudaStreamCreate(&memory));
    gpuErrchk(cudaStreamCreate(&bufferStream));

    prepareDataStructures();

    reportThread = new std::thread(reportFunction, this);
}

Manager::Manager(Graph* graph, int k, int numberOfActiveThreads, int blockSize, void (*kernel)(Device*), int numberOfSMs, int reportInterval, int jobsPerWarp, int induce) {
    this->h_k = k;

    if(induce)
        this->quickMapping = new QuickMapping(k);
    else
        this->quickMapping = new QuickMapping(-1);

    this->mainGraph = graph;
    this->h_numberOfActiveThreads = numberOfActiveThreads;
    this->blockSize = blockSize;
    this->result = 0;
    this->amountNewVertices = 0;
    this->removedEdges = 0;
    this->addedEdges = 0;
    this->activeThreads = 0;
    this->percentageWarpsIdle = 0;
    this->kernel = kernel;
    this->timer = new Timer();
    this->gpuFinished = false;
    this->numberOfSMs = numberOfSMs;
    this->smOccupancy = new int[numberOfSMs];
    this->currentRound = 0;
    this->reportInterval = reportInterval;
    this->h_induce = induce;
    this->h_subgraphsProcessed = 0;
    

    quickToCgMap = new std::unordered_map<unsigned int,unsigned int>();
    cgs = new std::set<unsigned int>();

    h_maxVertexId = mainGraph->getMaxVertexId();
    h_maxDegree = mainGraph->getMaxDegree();

    h_warpSize = 32;
 
    numberOfBlocks = ceil(h_numberOfActiveThreads/(float)blockSize);
    warpsPerBlock = blockSize / h_warpSize;
    numberOfWarps = numberOfBlocks * warpsPerBlock;

    this->device = (Device*)malloc(sizeof(Device));
    this->h_jobsPerWarp = jobsPerWarp;
    this->h_initialJobsPerWarp = ceil((h_maxVertexId+1)/(float)numberOfWarps);
    this->h_theoreticalJobsPerWarp = std::max(h_initialJobsPerWarp, h_jobsPerWarp);

    if(h_initialJobsPerWarp > h_jobsPerWarp) {
        printf("******************WARNING******************\n");
        printf("Initial jobs per warp higher than the jobs used during rebalancing.\n");
        printf("Initial jobs per warp: %d, jobs per warp during rebalancing: %d.\n", h_initialJobsPerWarp, h_jobsPerWarp);
        printf("*******************************************\n");
    }

    gpuErrchk(cudaStreamCreate(&main));
    gpuErrchk(cudaStreamCreate(&memory));
    gpuErrchk(cudaStreamCreate(&bufferStream));

    prepareDataStructures();

    reportThread = new std::thread(reportFunction, this);
}

Manager::~Manager() {
    gpuFinished = true;
    reportThread->join();

    cudaFree(device->d_degree);
    cudaFree(device->d_vertexOffset);
    cudaFree(device->d_adjacencyList);
    cudaFree(device->d_id);
    cudaFree(device->d_jobs);
    cudaFree(device->d_inductions);
    cudaFree(device->d_jobsPerWarp);
    cudaFree(device->d_currentJob);
    cudaFree(device->d_currentPosOfJob);
    cudaFree(device->d_validJobs);
    cudaFree(device->d_numberOfExtensions);
    cudaFree(device->d_result);
    cudaFree(device->d_status);
    cudaFree(device->d_smid);
    cudaFree((int*)device->d_globalVertexId);
    cudaFree(device->d_extensions);
    cudaFree(device->d_extensionSources);
    cudaFree(device->d_buffer);
    cudaFree(device->d_offsetBuffer);
    cudaFree(device->d_localSubgraphInduction);
    cudaFree(device->d_quickToCgLocal);
    cudaFree(device->d_hashPerWarp);
    cudaFree(device->d_induce);
    cudaFree(device->d_amountNewVertices);
    cudaFree(device->d_removedEdges);
    cudaFree(device->d_addedEdges);
    cudaFree((int*)device->d_bufferFull);
    cudaFree((int*)device->d_bufferDrain);
    cudaFree((bool*)device->d_stop);
    cudaFree(device->d_currentPos);

    free(h_degree);
    free(h_vertexOffset);
    free(h_adjacencyList);
    free(h_id);
    free(h_jobs);
    free(h_inductions);
    free(h_numberOfExtensions);
    free(h_currentJob);
    free(h_currentPosOfJob);
    free(h_validJobs);
    
    free(h_extensions);
    free(h_extensionsOffset);
    free(h_currentPos);
    free(h_hashPerWarp);
    free(h_hashGlobal);
    free(h_localSubgraphInduction);
    free(h_amountNewVertices);
    free(h_removedEdges);
    free(h_addedEdges);
    free(h_buffer);
    free(h_offsetBuffer);

    cudaFreeHost(h_status);
    cudaFreeHost(h_smid);
    cudaFreeHost(h_stop);
    cudaFreeHost(h_bufferFull);
    cudaFreeHost(h_bufferDrain);
    cudaFreeHost(h_result);

    cudaStreamDestroy(main);
    cudaStreamDestroy(memory);
    cudaStreamDestroy(bufferStream);

    // delete mainGraph;
    delete timer;

    delete quickToCgMap;
    delete cgs;
    delete quickMapping;
    delete[] smOccupancy;
    delete reportThread;
}

void Manager::initializeHostDataStructures() {
    h_extensionsOffset = (int*)malloc((h_k-1)*sizeof(int));

    h_extensionsLength = 0;
    h_extensionsOffset[0] = 0;
    for(int k = 1, length ; k <= h_k - 1 ; k++) {
        length = (int)ceilf(pow(2, ceilf(log2((float)(k * h_maxDegree)))));
        h_extensionsLength += length;
        if(k < h_k - 1)
            h_extensionsOffset[k] = h_extensionsLength;
    }

    h_globalVertexId = 0;
    h_vertexOffset = (int*)malloc((mainGraph->getMaxVertexId()+2)*sizeof(int));
    h_adjacencyList = (int*)malloc((mainGraph->getNumberOfEdges()*2 + (mainGraph->getMaxVertexId()+1)) * sizeof(int));
    h_degree = (int*)malloc((mainGraph->getMaxVertexId()+1)*sizeof(int));
    h_hashPerWarp = (unsigned long long*)malloc(numberOfWarps*quickMapping->numberOfCgs * sizeof(unsigned long long));
    h_hashGlobal = (long unsigned int*)malloc(quickMapping->numberOfCgs * sizeof(long unsigned int));
    h_localSubgraphInduction = (long unsigned int*)malloc(numberOfWarps * h_warpSize * sizeof(long unsigned int));
    h_buffer = (int*)malloc(numberOfWarps * GPU_BUFFER_SIZE_PER_WARP * sizeof(int));
    h_offsetBuffer = (unsigned int*)malloc(numberOfWarps * sizeof(unsigned int));

    h_id = (int*)malloc(numberOfWarps * h_warpSize * sizeof(int));
    h_numberOfExtensions = (int*)malloc(numberOfWarps * h_warpSize * sizeof(int));
    h_currentJob = (int*)malloc(numberOfWarps * sizeof(int));
    h_currentPosOfJob = (int*)malloc(numberOfWarps * h_theoreticalJobsPerWarp * sizeof(int));
    h_validJobs = (int*)malloc(numberOfWarps * sizeof(int));
    h_jobs = (int*)malloc(numberOfWarps * h_theoreticalJobsPerWarp * h_warpSize * sizeof(int));
    h_inductions = (int*)malloc(numberOfWarps * h_theoreticalJobsPerWarp * h_warpSize * sizeof(int));
    h_extensions = (int*)malloc(numberOfWarps * h_extensionsLength * sizeof(int));
    h_amountNewVertices = (long unsigned int*)malloc(numberOfWarps * sizeof(long unsigned int));
    h_removedEdges = (long unsigned int*)malloc(numberOfWarps * sizeof(long unsigned int));
    h_addedEdges = (long unsigned int*)malloc(numberOfWarps * sizeof(long unsigned int));
    
    gpuErrchk(cudaMallocHost((void**)&h_status, h_numberOfActiveThreads * sizeof(int)));
    gpuErrchk(cudaMallocHost((void**)&h_smid, h_numberOfActiveThreads * sizeof(int)));
    gpuErrchk(cudaMallocHost((void**)&h_stop, sizeof(bool)));
    gpuErrchk(cudaMallocHost((void**)&h_bufferFull, sizeof(int)));
    gpuErrchk(cudaMallocHost((void**)&h_bufferDrain, sizeof(int)));
    gpuErrchk(cudaMallocHost((void**)&h_result, numberOfWarps*sizeof(unsigned long)));
    *h_stop = false;
    *h_bufferFull = 0;
    *h_bufferDrain = 0;
    
    h_currentPos = (int*)malloc(numberOfWarps * sizeof(int));
    for(int i = 0 ; i < numberOfWarps ; i++) {
        h_currentPos[i] = -1;

        for(int j = 0 ; j < quickMapping->numberOfCgs ; j++) {
            h_hashPerWarp[i*quickMapping->numberOfCgs + j] = 0;
        }
    }
    for(int i = 0 ; i < quickMapping->numberOfCgs ; i++)
        h_hashGlobal[i] = 0;

    int offset = 0;
    for(int vertexId = 0 ; vertexId <= mainGraph->getMaxVertexId() ; vertexId++) {
        h_vertexOffset[vertexId] = offset;
        // printf("%d, vertexOffset: %d\n", vertexId, h_vertexOffset[vertexId]);
        for(std::set<int>::iterator itEdges = mainGraph->getNeighbours(vertexId)->begin() ; itEdges != mainGraph->getNeighbours(vertexId)->end() ; itEdges++)
            h_adjacencyList[offset++] = *itEdges;
        h_adjacencyList[offset++] = -1;

        h_degree[vertexId] = mainGraph->getNeighbours(vertexId)->size();
        // printf("%d, vertexOffset: %d, degree: %d\n", vertexId, h_vertexOffset[vertexId], h_degree[vertexId]);
    }
    h_vertexOffset[mainGraph->getMaxVertexId()+1] = h_vertexOffset[mainGraph->getMaxVertexId()]+h_degree[mainGraph->getMaxVertexId()]+1;

    memset(h_localSubgraphInduction, 0, numberOfWarps * h_warpSize * sizeof(long unsigned int));

    for(int i = 0 ; i < numberOfWarps ; i++) {
        h_currentJob[i] = 0;
        h_validJobs[i] = 0;
    }
    for(int round = 0 ; round < h_initialJobsPerWarp ; round++) {
        for(int i = 0 ; i < numberOfWarps ; i++) {
            int jobId = round*numberOfWarps+i;
            if(jobId <= h_maxVertexId) {
                h_validJobs[i]++;
                h_jobs[i*h_theoreticalJobsPerWarp*h_warpSize + round*h_warpSize + 0] = jobId;
                h_currentPosOfJob[i*h_theoreticalJobsPerWarp+round] = 0;
            } 
        }
    }

    for(int i = 0 ; i < h_numberOfActiveThreads ; i++)
        h_status[i] = 2;
}

void Manager::initializeDeviceDataStructures() {
    gpuErrchk(cudaMalloc((void**)&device->d_vertexOffset, (mainGraph->getMaxVertexId()+2)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_adjacencyList, (mainGraph->getNumberOfEdges()*2 + (mainGraph->getMaxVertexId()+1)) * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_degree, (mainGraph->getMaxVertexId()+1)*sizeof(int)));

    gpuErrchk(cudaMalloc((void**)&device->d_id, numberOfWarps * h_warpSize * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_jobs, numberOfWarps * h_theoreticalJobsPerWarp * h_warpSize * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_inductions, numberOfWarps * h_theoreticalJobsPerWarp * h_warpSize * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_jobsPerWarp, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_currentJob, numberOfWarps * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_currentPosOfJob, numberOfWarps * h_theoreticalJobsPerWarp * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_validJobs, numberOfWarps * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_numberOfExtensions, numberOfWarps * h_warpSize * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_result, numberOfWarps * sizeof(unsigned long)));
    gpuErrchk(cudaMalloc((void**)&device->d_status, h_numberOfActiveThreads * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_smid, h_numberOfActiveThreads * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_globalVertexId, sizeof(int)));

    gpuErrchk(cudaMalloc((void**)&device->d_extensions, numberOfWarps * h_extensionsLength * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_extensionSources, numberOfWarps * h_extensionsLength * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_buffer, numberOfWarps * GPU_BUFFER_SIZE_PER_WARP * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_offsetBuffer, numberOfWarps * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void**)&device->d_localSubgraphInduction, numberOfWarps * h_warpSize * sizeof(long unsigned int)));
    gpuErrchk(cudaMalloc((void**)&device->d_hashPerWarp, numberOfWarps * quickMapping->numberOfCgs * sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc((void**)&device->d_stop, sizeof(bool)));
    gpuErrchk(cudaMalloc((void**)&device->d_currentPos, numberOfWarps * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_induce, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_amountNewVertices, numberOfWarps * sizeof(long unsigned int)));
    gpuErrchk(cudaMalloc((void**)&device->d_removedEdges, numberOfWarps * sizeof(long unsigned int)));
    gpuErrchk(cudaMalloc((void**)&device->d_addedEdges, numberOfWarps * sizeof(long unsigned int)));

    gpuErrchk(cudaMalloc((void**)&device->d_k, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_extensionsLength, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_warpSize, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_extensionsOffset, (h_k-1) * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_quickToCgLocal, quickMapping->numberOfQuicks * sizeof(long unsigned int)));
    gpuErrchk(cudaMalloc((void**)&device->d_numberOfActiveThreads, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_maxVertexId, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_maxDegree, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_numberOfCgs, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_bufferFull, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_bufferDrain, sizeof(int)));

    gpuErrchk(cudaMalloc((void**)&d_device, sizeof(Device)));
    
    gpuErrchk(cudaMemcpy(device->d_vertexOffset, h_vertexOffset, (mainGraph->getMaxVertexId()+2)*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_adjacencyList, h_adjacencyList, (mainGraph->getNumberOfEdges()*2 + (mainGraph->getMaxVertexId()+1)) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_degree, h_degree, (mainGraph->getMaxVertexId()+1)*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((int*)device->d_globalVertexId, &h_globalVertexId, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((bool*)device->d_stop, h_stop, sizeof(bool), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_currentPos, h_currentPos, numberOfWarps * sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(device->d_k, &h_k, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_extensionsLength, &h_extensionsLength, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_warpSize, &h_warpSize, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_extensionsOffset, h_extensionsOffset, (h_k-1) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_quickToCgLocal, quickMapping->quickToCgLocal, quickMapping->numberOfQuicks * sizeof(long unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_numberOfActiveThreads, &h_numberOfActiveThreads, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_maxVertexId, &h_maxVertexId, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_maxDegree, &h_maxDegree, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_numberOfCgs, &(quickMapping->numberOfCgs), sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_jobsPerWarp, &h_theoreticalJobsPerWarp, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_induce, &h_induce, sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(device->d_jobs, h_jobs, numberOfWarps * h_theoreticalJobsPerWarp * h_warpSize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_currentJob, h_currentJob, numberOfWarps * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_validJobs, h_validJobs, numberOfWarps * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_currentPosOfJob, h_currentPosOfJob, numberOfWarps * h_theoreticalJobsPerWarp * sizeof(int), cudaMemcpyHostToDevice));    
    gpuErrchk(cudaMemcpy(device->d_status, h_status, h_numberOfActiveThreads * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((int*)device->d_bufferFull, h_bufferFull, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((int*)device->d_bufferDrain, h_bufferDrain, sizeof(int), cudaMemcpyHostToDevice));


    gpuErrchk(cudaMemset(device->d_smid, 0, h_numberOfActiveThreads * sizeof(int)));
    gpuErrchk(cudaMemset(device->d_offsetBuffer, 0, numberOfWarps * sizeof(unsigned int)));
    gpuErrchk(cudaMemset(device->d_localSubgraphInduction, 0, numberOfWarps * h_warpSize * sizeof(long unsigned int)));
    gpuErrchk(cudaMemset(device->d_hashPerWarp, 0, numberOfWarps * quickMapping->numberOfCgs * sizeof(unsigned long long)));
    gpuErrchk(cudaMemset(device->d_amountNewVertices, 0, numberOfWarps * sizeof(long unsigned int)));
    gpuErrchk(cudaMemset(device->d_removedEdges, 0, numberOfWarps * sizeof(long unsigned int)));
    gpuErrchk(cudaMemset(device->d_addedEdges, 0, numberOfWarps * sizeof(long unsigned int)));    
    
    gpuErrchk(cudaMemcpy(d_device, device, sizeof(Device), cudaMemcpyHostToDevice));
}

void Manager::prepareDataStructures() {
    initializeHostDataStructures();
    initializeDeviceDataStructures();
}

void Manager::detailedIdlenessReport() {
    int warpsPerBlock = blockSize / h_warpSize;
    int amountActive = 0;
    std::cout << warpsPerBlock << ";" << blockSize << ";" << warpsPerBlock << ";" << numberOfBlocks << "\n";

    for(int i = 0, blockIdle, blockActive, blockBusy, warpsBusy ; i < numberOfBlocks ; i++) {
        blockIdle = blockActive = blockBusy = warpsBusy = 0;
        for(int j = 0, warpIdle, warpActive, warpBusy ; j < warpsPerBlock ; j++) {
            warpIdle = warpActive = warpBusy = 0;
            for(int k = 0, offset ; k < h_warpSize ; k++) {
                offset = i*blockSize+j*h_warpSize+k;
                if(h_status[offset] == 0)
                    warpActive++;
                else if(h_status[offset] == 1)
                    warpBusy++;
                else if(h_status[offset] == 2)
                    warpIdle++;
                else {
                    std::cout << "BUUUUUUUUUUUUUGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG\n";
                    exit(1);
                }
            }

            if(warpBusy > 0)
                warpsBusy++;
            //std::cout << (warpActive/(float)h_warpSize)*100 << " available | " << (warpBusy/(float)h_warpSize)*100 << " busy | " << (warpIdle/(float)h_warpSize)*100 << " idle.\n";

            blockActive+=warpActive;
            blockBusy+=warpBusy;
            blockIdle+=warpIdle;
        }
        if(blockBusy > 0) {
            std::cout << "Block " << i << " summary: ";
            std::cout << "Block stats -> " << blockActive << " available (" << (blockActive/(float)blockSize)*100 << " %) | " << blockBusy << " busy (" << (blockBusy/(float)blockSize)*100 << " %) | " << blockIdle << " idle (" << (blockIdle/(float)blockSize)*100 << " %) ; Warp stats -> " << warpsBusy << " warps active | " << (((warpsBusy*32)-blockBusy)/((float)warpsBusy*32))*100 << " % warp idleness.\n";
            amountActive++;
        }
     }
     std::cout << "-> Amount active: " << amountActive << "\n";
}

void Manager::shortIdlenessReport(const char* message) {
    std::time_t result = std::time(nullptr);
    printf("%s warps idle: %d, numberOfWarps: %d, %f%% warps active, %s", message, amountWarpsIdle, numberOfWarps, 100-percentageWarpsIdle, std::asctime(std::localtime(&result)));
    smOccupancyReport();
}

void Manager::smOccupancyReport() {
    memset(smOccupancy, 0, numberOfSMs * sizeof(int));
    //
    for(int tid = 0 ; tid < h_numberOfActiveThreads ; tid+=32) {
        if(h_status[tid] == 1 && h_smid[tid] != -1) {
            // std::cout << tid << "," << tid / 32 << "," << h_smid[tid] << "\n";
            smOccupancy[h_smid[tid]]++;
        }
    }
    //
    for(int smId = 0 ; smId < numberOfSMs ; smId++) {
        std::cout << smOccupancy[smId] << ",";
    }
    std::cout << "\n";

}

void Manager::queuesReport() {

}

void Manager::startTimer() {
    timer->play("Manager");
}

void Manager::stopTimer() {
    timer->pause();
}

double Manager::getRuntimeInMilliseconds() {
    return timer->getElapsedTimeInMiliseconds();
}

double Manager::getRuntimeInSeconds() {
    return timer->getElapsedTimeInSeconds();
}

void Manager::runKernel() {
    void* args[] = {&(d_device)};
    cudaLaunchKernel((void*)kernel, dim3(numberOfBlocks), dim3(blockSize), args, 0, main);
}

void Manager::waitKernel() {
    gpuErrchk(cudaStreamSynchronize(main));
}

void Manager::stopKernel(){
    *h_stop = true;
    gpuErrchk(cudaMemcpyAsync((bool*)device->d_stop, h_stop, sizeof(bool), cudaMemcpyHostToDevice, memory));
    gpuErrchk(cudaStreamSynchronize(memory));
    gpuErrchk(cudaStreamSynchronize(main));
}


//TODO Leave this function more "high-level" to allow customizations from user.
bool Manager::rebalance() {
    Timer* timer = new Timer();
    timer->play("Rebalance");

    copyWarpDataFromGpu();
    copyResult();
    std::vector<int> idles, indifferents;
    std::priority_queue<Donator*, std::vector<Donator*>, Comparator> actives;
    int totalWeight = organizeThreadStatus(&idles, &actives, &indifferents);
    // printf("idles size: %lu, actives size: %lu, indifferents size: %lu\n", idles.size(), actives.size(), indifferents.size());
    if(actives.size() > 0) {
        bool full = donate(&idles, &actives, totalWeight);
        copyWarpDataBackToGpu();
        timer->pause();

        printf("[REBALANCING] time: %.3f, full: %s, totalWeight: %d, #jobs/#idle_warps: %.2f\n", timer->getElapsedTimeInMiliseconds(), full ? "yes" : "no", totalWeight, (float)totalWeight/idles.size());
        delete timer;

        return true;
    }
    else {
        delete timer;

        invalidateResult();
        return false;
    }
}

void Manager::copyResult() {
    gpuErrchk(cudaMemcpy(h_result, device->d_result, numberOfWarps*sizeof(unsigned long), cudaMemcpyDeviceToHost));
    
    amountNewVertices = removedEdges = addedEdges = 0;
    gpuErrchk(cudaMemcpy(h_amountNewVertices, device->d_amountNewVertices, numberOfWarps*sizeof(long unsigned int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_removedEdges, device->d_removedEdges, numberOfWarps*sizeof(long unsigned int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_addedEdges, device->d_addedEdges, numberOfWarps*sizeof(long unsigned int), cudaMemcpyDeviceToHost));

    for(int i = 0 ; i < numberOfWarps ; i++) {
        result += h_result[i];
    }
}

void Manager::aggregate() {
    if(!h_induce) {
        printf("ERROR! You can aggregate only with the induce option enabled...\n");
        exit(1);
    }

    gpuErrchk(cudaMemcpy(h_hashPerWarp, device->d_hashPerWarp, numberOfWarps * quickMapping->numberOfCgs * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    for(int i = 0 ; i < numberOfWarps ; i++) {
        for(int j = 0 ; j < quickMapping->numberOfCgs ; j++) {
            h_hashGlobal[j] += h_hashPerWarp[i*quickMapping->numberOfCgs+j];
        }
    }
}

void Manager::printResult() {
    aggregate();

    long unsigned int validSubgraphs = 0;
    for(int i = 0 ; i < quickMapping->numberOfCgs ; i++) {
        if(h_hashGlobal[i] > 0) {
            // printf("%d, %lu\n", i, h_hashGlobal[i]);
            validSubgraphs += h_hashGlobal[i];
        }
    }
    printf("(%lu,%lu)\n", result, validSubgraphs);
}

void Manager::printCount() {
    
    printf("(%lu)\n", result);
    printf("Subgraphs processed: %s.\n", h_subgraphsProcessed == result ? "MATCH" : "MISMATCH");
}

void Manager::loadGpuThreadStatus() {
    gpuErrchk(cudaMemcpyAsync(h_status, device->d_status, h_numberOfActiveThreads*sizeof(int), cudaMemcpyDeviceToHost, memory));
    gpuErrchk(cudaMemcpyAsync(h_smid, device->d_smid, h_numberOfActiveThreads*sizeof(int), cudaMemcpyDeviceToHost, memory));
    gpuErrchk(cudaStreamSynchronize(memory));

    amountThreadsIdle = 0;
    amountWarpsIdle = 0;
    for(int i = 0, intraWarpIdle ; i < numberOfWarps ; i++)
    {
        intraWarpIdle = 0;
        for(int j = 0 ; j < h_warpSize ; j++) {
            if(h_status[i*h_warpSize+j] == 2) {
                amountThreadsIdle++;
                intraWarpIdle++;
            }
        }

        if(intraWarpIdle == h_warpSize)
            amountWarpsIdle++;
    }

    percentageWarpsIdle = (amountWarpsIdle/(double)numberOfWarps)*100;
    activeThreads = h_numberOfActiveThreads - amountThreadsIdle;
}

bool Manager::thereAreActiveGpuThreads() {
    loadGpuThreadStatus();
    return amountThreadsIdle < h_numberOfActiveThreads;
}

void Manager::sleepFor(int millisecs) {
    std::this_thread::sleep_for(std::chrono::milliseconds(millisecs));
    currentRound+=millisecs;
}

bool Manager::gpuIsIdle(int threshold) {
    return percentageWarpsIdle >= 100 || percentageWarpsIdle > threshold;
}

int Manager::amountActiveGpuThreads() {
    return activeThreads;
}

double Manager::percentageWarpIdleness() {
    return percentageWarpsIdle;
}

void Manager::copyWarpDataFromGpu() {
    //  Read:
    //      - h_id
    //      - h_numberOfExtensions
    //      - h_extensions
    //      - h_status
    //      - h_currentPos
    //      - h_extensionsOffset (don't have to be copied, 'cause it's calculated previously)
    //      - h_result

    // Consistent copy of h_status array
    // std::cout << "Copying result from device...\n";
    gpuErrchk(cudaMemcpy(h_id, device->d_id, numberOfWarps * h_warpSize * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_jobs, device->d_jobs, numberOfWarps * h_theoreticalJobsPerWarp * h_warpSize * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_inductions, device->d_inductions, numberOfWarps * h_theoreticalJobsPerWarp * h_warpSize * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_currentJob, device->d_currentJob, numberOfWarps * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_currentPosOfJob, device->d_currentPosOfJob, numberOfWarps * h_theoreticalJobsPerWarp * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_validJobs, device->d_validJobs, numberOfWarps * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_numberOfExtensions, device->d_numberOfExtensions, numberOfWarps * h_warpSize * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_extensions, device->d_extensions, numberOfWarps * h_extensionsLength * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_status, device->d_status, h_numberOfActiveThreads*sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_smid, device->d_smid, h_numberOfActiveThreads*sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_currentPos, device->d_currentPos, numberOfWarps * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_localSubgraphInduction, device->d_localSubgraphInduction, numberOfWarps * h_warpSize * sizeof(long unsigned int), cudaMemcpyDeviceToHost));

}

void Manager::invalidateResult() {
    gpuErrchk(cudaMemset(device->d_result, 0, numberOfWarps*sizeof(unsigned long)));
}

void Manager::copyWarpDataBackToGpu() {
    //  Written
    //      - h_id
    //      - h_numberOfExtensions
    //      - h_currentPos
    //      - h_stop

    // gpuErrchk(cudaMemcpy(device->d_id, h_id, numberOfWarps * h_warpSize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_jobs, h_jobs, numberOfWarps * h_theoreticalJobsPerWarp * h_warpSize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_inductions, h_inductions, numberOfWarps * h_theoreticalJobsPerWarp * h_warpSize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_currentJob, h_currentJob, numberOfWarps * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_currentPosOfJob, h_currentPosOfJob, numberOfWarps * h_theoreticalJobsPerWarp * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_validJobs, h_validJobs, numberOfWarps * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_numberOfExtensions, h_numberOfExtensions, numberOfWarps * h_warpSize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_extensions, h_extensions, numberOfWarps * h_extensionsLength * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_currentPos, h_currentPos, numberOfWarps * sizeof(int), cudaMemcpyHostToDevice));
    *h_stop = false;
    gpuErrchk(cudaMemcpy((bool*)device->d_stop, h_stop, sizeof(bool), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_localSubgraphInduction, h_localSubgraphInduction, numberOfWarps * h_warpSize * sizeof(long unsigned int), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
}

int Manager::organizeThreadStatus(std::vector<int>* idles, std::priority_queue<Donator*, std::vector<Donator*>, Comparator>* actives, std::vector<int>* indifferents) {
    // printf("#organizeThreadStatus\n");
    //  Data structures read
    //      - h_status
    //      - h_currentPos
    //      - h_embeddigs->numberOfExtensions
    int totalWeight = 0;
    for(int i = 0, wid ; i < h_numberOfActiveThreads ; i+=h_warpSize) {
        // std::cout << i << "\n";
        wid = (int)(i/h_warpSize);

        EnumerationHelper *helper = new EnumerationHelper(wid, h_k, h_warpSize, h_theoreticalJobsPerWarp, h_extensionsLength, h_id, h_extensions, h_extensionsOffset, h_numberOfExtensions, h_currentPos, h_currentPosOfJob, h_localSubgraphInduction, h_inductions, h_validJobs, h_currentJob, h_jobs, h_result);

        // Idles for sure
        if(h_status[i] == 2 && helper->jobQueueIsEmpty()) {
            // std::cout << "IDLE at all\n";
            idles->push_back(wid);
        }
        // Actives available for donation
        else if (h_status[i] == 1) {
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

    // printf("INDIFFERENTS' SIZE: %lu\n", indifferents->size());

    return totalWeight;

    // DEBUG Idles
    // std::cout << "**************\n";
    // std::cout << "Printing idles\n";
    // std::cout << "**************\n";
    // for(int i = 0 ; i < idles->size() ; i++)
    //     std::cout << "[IDLE]:" << idles->at(i) << "\n";

    // DEBUG Actives
    // std::cout << "****************\n";
    // std::cout << "Printing actives\n";
    // std::cout << "****************\n";
    // printQueue(*actives);

    // DEBUG Indifferents
    // std::cout << "*********************\n";
    // std::cout << "Printing indifferents\n";
    // std::cout << "*********************\n";
    // for(int i = 0 ; i < indifferents->size() ; i++)
    //     std::cout << "[INDIFFERENT]:" << indifferents->at(i) << "\n";
}

bool Manager::donate(std::vector<int>* idles, std::priority_queue<Donator*, std::vector<Donator*>, Comparator>* actives, int totalWeight) {
    // printf("#donate\n");
    //  Read:
    //      - h_id
    //      - h_numberOfExtensions
    //      - h_extensions
    //      - h_extensionsOffset
    //  Written
    //      - h_id
    //      - h_numberOfExtensions
    //      - h_currentPos

    EnumerationHelper *donatorHelper, *idleHelper;
    Donator* donator;
    int currentIdle;
    bool active;

    int i = 0;
    float idealWeight = ceil((float)totalWeight / (actives->size()+idles->size()));
    int jobsPerWarp = idealWeight < h_jobsPerWarp ? idealWeight : h_jobsPerWarp;

    printf("jobsPerWarp: %d\n", jobsPerWarp);
    for(int job = 0 ; job < jobsPerWarp ; job++) {
        for(i = 0 ; i < idles->size() && !actives->empty() ; i++) {
            currentIdle = idles->at(i);
            donator = actives->top();
            actives->pop();
            
            donatorHelper = new EnumerationHelper(donator->wid, h_k, h_warpSize, h_theoreticalJobsPerWarp, h_extensionsLength, h_id, h_extensions, h_extensionsOffset, h_numberOfExtensions, h_currentPos, h_currentPosOfJob, h_localSubgraphInduction, h_inductions, h_validJobs, h_currentJob, h_jobs, h_result);

            // printf("[BEFORE] Donator\n");
            // donatorHelper->report();

            idleHelper = new EnumerationHelper(currentIdle, h_k, h_warpSize, h_theoreticalJobsPerWarp, h_extensionsLength, h_id, h_extensions, h_extensionsOffset, h_numberOfExtensions, h_currentPos, h_currentPosOfJob, h_localSubgraphInduction, h_inductions, h_validJobs, h_currentJob, h_jobs, h_result);
            
            // printf("[BEFORE] Idle\n");
            // idleHelper->report();

            if(job == 0) {
                idleHelper->setValidJobs(0);
                idleHelper->setCurrentJob(0);  
                idleHelper->setCurrentPos(-1);              
            }

            for(int k = 0 ; k <= donator->targetLevel ; k++) {
                idleHelper->setJob(job, k, donatorHelper->getId(k));
                idleHelper->setInductions(job, k, donatorHelper->getLocalSubgraphInduction(k));
            }
            idleHelper->setJob(job, donator->targetLevel+1, donatorHelper->popLastExtension(donator->targetLevel));
            idleHelper->setCurrentPosOfJob(job, donator->targetLevel+1);
            idleHelper->increaseJob();

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
            
            // printf("[AFTER] Donator\n");
            // donatorHelper->report();
            // printf("[AFTER] Idle\n");
            // idleHelper->report();

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

void Manager::debug(const char* message) {
    std::cout << message << "\n";
    std::cout << "Debug reached. Type any int value to move on...\n";
    int a;
    std::cin >> a;
}

std::string Manager::generatePattern(graph* g, int m, int n) {
    std::string pattern = "";
    for (int k = 0; k < m*(size_t)n; ++k) {
        pattern += std::to_string(g[k]);
        pattern += ",";
    }
    return pattern;
}

void* Manager::reportFunction(Manager* manager) {
    // std::cout << "[ReportFunction] Begin\n";

    int *status, *smid, *smOccupancy, *result;
    cudaStream_t stream;
    int currentRound = 0;

    smOccupancy = new int[manager->numberOfSMs];
    gpuErrchk(cudaStreamCreate(&stream));
    gpuErrchk(cudaMallocHost((void**)&status, manager->h_numberOfActiveThreads * sizeof(int)));
    gpuErrchk(cudaMallocHost((void**)&smid, manager->h_numberOfActiveThreads * sizeof(int)));
    gpuErrchk(cudaMallocHost((void**)&result, manager->numberOfWarps * sizeof(unsigned long)));
    int ticks = 0;
    while(!manager->gpuFinished) {

        gpuErrchk(cudaMemcpyAsync(status, manager->device->d_status, manager->h_numberOfActiveThreads*sizeof(int), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaMemcpyAsync(smid, manager->device->d_smid, manager->h_numberOfActiveThreads*sizeof(int), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaMemcpyAsync(result, manager->device->d_result, manager->numberOfWarps * sizeof(unsigned long), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));

        // Idleness
        int amountThreadsIdle = 0;
        int amountWarpsIdle = 0;
        float percentageWarpsIdle = 0;
        int activeThreads = 0;
        unsigned long totalResult = 0;
        for(int i = 0, intraWarpIdle ; i < manager->numberOfWarps ; i++)
        {
            intraWarpIdle = 0;
            for(int j = 0 ; j < manager->h_warpSize ; j++) {
                if(status[i*manager->h_warpSize+j] == 2) {
                    amountThreadsIdle++;
                    intraWarpIdle++;
                }
            }
            if(intraWarpIdle == manager->h_warpSize)
                amountWarpsIdle++;

            totalResult += result[i];
        }
        percentageWarpsIdle = (amountWarpsIdle/(double)manager->numberOfWarps)*100;
        activeThreads = manager->h_numberOfActiveThreads - amountThreadsIdle;
        std::time_t result = std::time(nullptr);

        // SM Occupancy
        memset(smOccupancy, 0, manager->numberOfSMs * sizeof(int));
        for(int tid = 0 ; tid < manager->h_numberOfActiveThreads ; tid+=32) {
            if(status[tid] == 1 && smid[tid] != -1)
                smOccupancy[smid[tid]]++;
        }

        // printf("[REPORT]numberOfWarps:%d,active warps:%d,idle warps:%d,%f%% warps active,", manager->numberOfWarps,manager->numberOfWarps-amountWarpsIdle, amountWarpsIdle, 100-percentageWarpsIdle);
        int totalWarps = 0;
        for(int i = 0 ; i < manager->numberOfSMs ; i++) {
            // printf("%d,", smOccupancy[i]);
            totalWarps += smOccupancy[i];
        }
        // printf("%d,%lu,%s",totalWarps,totalResult,std::asctime(std::localtime(&result)));

        // for(int smId = 0 ; smId < manager->numberOfSMs ; smId++)
        //     std::cout << currentRound << "," << smId << "," << smOccupancy[smId] << "\n";


        std::this_thread::sleep_for(std::chrono::milliseconds(manager->reportInterval));
        currentRound += manager->reportInterval;
        // std::cout << "[reportFunction] " << currentRound << "\n";
        ticks++;
    }

    delete[] smOccupancy;
    cudaFreeHost(status);
    cudaFreeHost(smid);
    cudaFreeHost(result);
    cudaStreamDestroy(stream);

    // std::cout << "[ReportFunction] End " << "\n";
    return NULL;
}

void Manager::readQuickToCgMap() {
    std::string filename = "datasets/quicks/quick_to_cg_";
    filename += std::to_string(h_k);
    filename += ".csv";

    unsigned int g, cg;

    printf("%s\n", filename.c_str());

    FILE *fpr = fopen(filename.c_str(), "r");

    if(fpr == NULL) {
        printf("[readQuickToCgMap] Quick to cg mapping doesn't exist. Creating mapping...\n");
        FILE *fpw = fopen(filename.c_str(), "w");
        if(fpw == NULL) {
            printf("[readQuickToCgMap] Aborting... creating quick to cg file denied\n");
            exit(1);
        }
        generateQuickToCgMap();

        for(auto it = quickToCgMap->begin() ; it != quickToCgMap->end() ; it++)
            fprintf(fpw, "%u,%u\n", it->first, it->second);

        fclose(fpw);
    }
    else {
        while(fscanf(fpr,"%u,%u", &g, &cg) != EOF) {
            // printf("%u,%u\n", g, cg);
            (*quickToCgMap)[g] = cg;
        }
        fclose(fpr);
    }

    printf("[readQuickToCgMap] FINISHED!\n");
}

void Manager::printBinaryLong(graph value) {
    long unsigned int mask;

    printf("value: %lu\n", value);

    for(int i = 63 ; i >= 0 ; i--) {
        mask = (long unsigned int)pow(2,i);
        printf("%lu", ((long unsigned int)(value & mask)) >> i);
    }
}

void Manager::generateQuickToCgMap() {
    unsigned int quickG, quickCg;
    int bits = ((h_k+1)*(h_k-2))/2;
    // printf("[START] Number of bits for k = %d: %d\n", k, bits);

    int m, n;
    n = h_k;
    m = SETWORDSNEEDED(n);

    graph g[MAXN*MAXM], cg[MAXN*MAXM];
    int lab[MAXN],ptn[MAXN],orbits[MAXN];
    static DEFAULTOPTIONS_GRAPH(options);
    statsblk stats;

    options.getcanon = TRUE;

    nauty_check(WORDSIZE,m,n,NAUTYVERSIONID);

    for(unsigned int i = 0 ; i < pow(2,bits) ; i++) {
        EMPTYGRAPH(g,m,n);
        EMPTYGRAPH(cg,m,n);

        ADDONEEDGE(g,0,1,m);

        unsigned int value = i;
        for (int source = 2, offset = 0; source < n; source++, offset+=source) {
            for(int target = 0 ; target < source ; target++, value=value>>1) {
                if((value & 1) == 1)
                    ADDONEEDGE(g,source,target,m);
            }
        }

        densenauty(g,lab,ptn,orbits,&options,&stats,m,n,cg);

        // for (int k = 0; k < m*(size_t)n; ++k) {
        //     printBinaryLong(g[k]);
        //     printf("\n");
        // }
        quickG = generateQuickG(g, h_k, m, n);
        // printf("quick: %lu\n", quickG);

        // for (int k = 0; k < m*(size_t)n; ++k) {
        //     printBinaryLong(cg[k]);
        //     printf("\n");
        // }
        quickCg = generateQuickCg(cg, h_k, m, n);
        // printf("quick: %lu\n", quickCg);
        // printf("\n\n");

        // printf("\n\n\n");

        (*quickToCgMap)[quickG] = quickCg;
        cgs->insert(quickCg);
    }
}

void printBinaryLong(graph value) {
    long unsigned int mask;

    printf("value: %lu\n", value);

    for(int i = WORDSIZE-1 ; i >= 0 ; i--) {
        mask = (long unsigned int)pow(2,i);
        printf("%lu", ((long unsigned int)(value & mask)) >> i);
    }
}

unsigned int Manager::generateQuickCg(graph* g, int k, int m, int n) {
    long unsigned int quick = 0, mask;
    for(int i = 1, weight = 0 ; i < m*(size_t)n ; i++) {
        for(int j = 0, l = WORDSIZE-1 ; j < i ; j++, l--) {
            mask = pow(2,l);
            quick += ((g[i] & mask) >> l)*pow(2,weight);
            weight++;
        }
    }
    return quick;
}

unsigned int Manager::generateQuickG(graph* g, int k, int m, int n) {
    long unsigned int quick = 0, mask;
    for(int i = 2, weight = 0 ; i < m*(size_t)n ; i++) {
        for(int j = 0, l = WORDSIZE-1 ; j < i ; j++, l--) {
            mask = pow(2,l);
            quick += ((g[i] & mask) >> l)*pow(2,weight);
            weight++;
        }
    }
    return quick;
}

void Manager::check(int flag) {
    if(!flag)
        printf("Status: unchecked.\n");
    else {
        unsigned long gt[15];
        int k;

        std::string groundTruth = graphFile;
        groundTruth += ".ground_truth";
        
        FILE* fp = fopen(groundTruth.c_str(), "r");

        int i = 3;
        while(fscanf(fp,"%d,%lu", &k, &gt[i]) != EOF) i++;

        fclose(fp);

        if(result == gt[h_k])
            printf("Status: OK.\n");
        else
            printf("Status: FAILED. Expected: %lu. Found: %lu. Difference: %ld.\n", gt[h_k], result, result-gt[h_k]);
            
    }
}

void Manager::processEdgeWeights() {
    printf("[BEGIN] Copying buffer and processing edge weights...\n");
    gpuErrchk(cudaMemcpy(h_buffer, device->d_buffer, numberOfWarps * GPU_BUFFER_SIZE_PER_WARP * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_offsetBuffer, device->d_offsetBuffer, numberOfWarps * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemset(device->d_offsetBuffer, 0, numberOfWarps * sizeof(unsigned int)));

    
    unsigned int subgraph[10];

    for(int i = 0 ; i < numberOfWarps ; i++) {
        int amount = 0;
        int j = 0;

        while(j < h_offsetBuffer[i]) {
            amount = h_buffer[i*GPU_BUFFER_SIZE_PER_WARP + j + h_k - 1];
            h_subgraphsProcessed += amount;

            int k = 0;
            for(k = 0 ; k < h_k - 1 ; k++)
                subgraph[k] = h_buffer[i*GPU_BUFFER_SIZE_PER_WARP + j + k];

            for(int a = 0 ; a < h_k - 2 ; a++) {
                for(int b = a + 1 ; b < h_k - 1 ; b++) {
                    mainGraph->addEdgeWeight(subgraph[a], subgraph[b], amount);
                }
            }
            
            for(int l = 0 ; l < amount ; l++) {
                subgraph[k] = h_buffer[i*GPU_BUFFER_SIZE_PER_WARP + j + h_k + l];
                
                for(int m = 0 ; m < h_k - 1 ; m++)
                    mainGraph->addEdgeWeight(subgraph[m], subgraph[k], 1);
                
                // subgraph[0 .. h_k-1] is ready here!
                // unsigned int* currentSubgraph = new unsigned int[h_k];
                // memcpy(currentSubgraph, subgraph, h_k * sizeof(unsigned int));
                // subgraphs->push_back(currentSubgraph);
            }
            j += (h_k+amount);
        }
    }
    printf("[END] Copying buffer and processing edge weights...\n");
}

int Manager::bufferDrain() {
    gpuErrchk(cudaMemcpyAsync(h_bufferDrain, (int*)device->d_bufferDrain, sizeof(int), cudaMemcpyDeviceToHost, memory));
    gpuErrchk(cudaStreamSynchronize(memory));
    return *h_bufferDrain;
}

int Manager::bufferFull() {
    gpuErrchk(cudaMemcpyAsync(h_bufferFull, (int*)device->d_bufferFull, sizeof(int), cudaMemcpyDeviceToHost, memory));
    gpuErrchk(cudaStreamSynchronize(memory));
    return *h_bufferFull; 
}

void Manager::processCompressionScore() {
    printf("[BEGIN] Processing compression score...\n");

    // Write subgraph buffer to the file
    // if(bufferFull()) {
    //     printf("ERROR! Buffer is full and compression will not work properly...\n");
    //     return;
    // }
    gpuErrchk(cudaMemcpy(h_buffer, device->d_buffer, numberOfWarps * GPU_BUFFER_SIZE_PER_WARP * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_offsetBuffer, device->d_offsetBuffer, numberOfWarps * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemset(device->d_offsetBuffer, 0, numberOfWarps * sizeof(unsigned int)));

    
    unsigned int subgraph[10];

    for(int i = 0 ; i < numberOfWarps ; i++) {
        int amount = 0;
        int j = 0;

        while(j < h_offsetBuffer[i]) {
            amount = h_buffer[i*GPU_BUFFER_SIZE_PER_WARP + j + h_k - 1];
            h_subgraphsProcessed += amount;

            int k = 0;
            for(k = 0 ; k < h_k - 1 ; k++)
                subgraph[k] = h_buffer[i*GPU_BUFFER_SIZE_PER_WARP + j + k];
            
            for(int l = 0 ; l < amount ; l++) {
                subgraph[k] = h_buffer[i*GPU_BUFFER_SIZE_PER_WARP + j + h_k + l];

                // subgraph[0 .. h_k-1] is ready here!            
                int weight = mainGraph->calculateWeight(subgraph, h_k);
                if(weight > 0) {
                    weights.push_back(mainGraph->calculateWeight(subgraph, h_k));
                }
                
            }
            j += (h_k+amount);
        }
    }

    printf("[END] Copying buffer and processing edge weights...\n");
}

void Manager::compressionResults() {
    sort(weights.begin(), weights.end(), std::greater<int>());
    int sum = 0, totalSubgraphsUsed = 0;
    for(auto it:weights) {
        sum += it;
        totalSubgraphsUsed++;
    }

    // int outOfMemoryWarps = 0;
    // for(int i = 0 ; i < numberOfWarps ; i++) {
    //     if(h_removedEdges[i] == 1)
    //         outOfMemoryWarps++;
    // }
    // printf("##############################################################\n");
    // printf("%d warps ficaram sem espaço para bufferização.\n", outOfMemoryWarps);
    // printf("Relatório de ocupação dos buffers...\n");
    // int maxOccupancy = -1, minOccupancy = 101;
    // for(int i = 0 ; i < numberOfWarps ; i++) {
    //     if(h_removedEdges[i] == 1) {
    //         printf("%d: FULL\n", i);
    //         maxOccupancy = 100;
    //     }
    //     else {
    //         float bufferOccupancy = ((float)h_removedEdges[i]/(float)GPU_BUFFER_SIZE_PER_WARP)*100;
    //         if(bufferOccupancy > maxOccupancy)
    //             maxOccupancy = bufferOccupancy;
    //         if(bufferOccupancy < minOccupancy)
    //             minOccupancy = bufferOccupancy;
    //         printf("%d: %.2f\n", i, bufferOccupancy);
    //     }
    // }
    // printf("Min/max ocupação do buffer: %d / %d\n", minOccupancy, maxOccupancy);

    printf("##############################################################\n");
    printf("Espaço salvo / subgrafos usados para compressão: %d / %d\n", sum, totalSubgraphsUsed);
    printf("##############################################################\n");

}