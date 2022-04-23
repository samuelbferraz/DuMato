#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <queue>
#include <vector>
#include <thread>
#include <unordered_map>
#include <ctime>
#include "Manager.h"
#include "Timer.h"
#include "CudaHelperFunctions.h"
#define MAXN 10
#include "nauty.h"

Manager::Manager(const char* graphFile, int k, int numberOfActiveThreads, int blockSize, void (*kernel)(Device*), bool getcanon, int numberOfWorkerThreads, int numberOfSMs, int reportInterval, bool canonical_relabeling) {
    this->h_k = k;
    this->quickMapping = new QuickMapping(k, canonical_relabeling);
    this->mainGraph = new Graph(graphFile);
    this->h_numberOfActiveThreads = numberOfActiveThreads;
    this->blockSize = blockSize;
    this->result = 0;
    this->activeThreads = 0;
    this->percentageWarpsIdle = 0;
    this->kernel = kernel;
    this->timer = new Timer();
    this->numberOfWorkerThreads = numberOfWorkerThreads;
    this->gpuFinished = false;
    this->getcanon = getcanon;
    this->numberOfSMs = numberOfSMs;
    this->smOccupancy = new int[numberOfSMs];
    this->currentRound = 0;
    this->reportInterval = reportInterval;

    quickToCgMap = new std::unordered_map<unsigned int,unsigned int>();
    cgCounting = new std::unordered_map<unsigned int, unsigned long int>*[numberOfWorkerThreads];
    cgs = new std::set<unsigned int>();

    h_maxVertexId = mainGraph->getMaxVertexId();
    h_maxDegree = mainGraph->getMaxDegree();

    // std::cout << "Done!" << "\n";

    h_warpSize = 32;
    // std::cout << h_numberOfActiveThreads << " active threads\n";
    numberOfBlocks = ceil(h_numberOfActiveThreads/(float)blockSize);
    warpsPerBlock = blockSize / h_warpSize;
    numberOfWarps = numberOfBlocks * warpsPerBlock;

    this->device = (Device*)malloc(sizeof(Device));

    gpuErrchk(cudaStreamCreate(&main));
    gpuErrchk(cudaStreamCreate(&memory));
    gpuErrchk(cudaStreamCreate(&bufferStream));

    prepareDataStructures();

    chunksWorker = new unsigned int*[numberOfWorkerThreads];
    chunksWorkerSize = new int[numberOfWorkerThreads];
    chunksEmptySemaphore = new sem_t[CHUNKS_CPU];
    chunksFullSemaphore = new sem_t[CHUNKS_CPU];
    for(int i = 0 ; i < numberOfWorkerThreads ; i++) {
        chunksWorker[i] = new unsigned int[CHUNK_SIZE];
        cgCounting[i] = new std::unordered_map<unsigned int,unsigned long int>();
    }
    for(int i = 0 ; i < CHUNKS_CPU ; i++) {
        sem_init(&(chunksEmptySemaphore[i]), 0, 1);
        sem_init(&(chunksFullSemaphore[i]), 0, 0);
    }

    if(getcanon) {
        readGpuBufferThread = new std::thread(readGpuBufferFunction, this);
        workersThread = new std::thread*[numberOfWorkerThreads];
        for(int i = 0 ; i < numberOfWorkerThreads ; i++)
            workersThread[i] = new std::thread(induceCanonicalizeFunction, this, i);
    }

    reportThread = new std::thread(reportFunction, this);
}

Manager::~Manager() {
    gpuFinished = true;
    reportThread->join();

    cudaFree(device->d_degree);
    cudaFree(device->d_vertexOffset);
    cudaFree(device->d_adjacencyList);
    cudaFree(device->d_id);
    cudaFree(device->d_numberOfExtensions);
    cudaFree(device->d_embeddings);
    cudaFree(device->d_result);
    cudaFree(device->d_status);
    cudaFree(device->d_smid);
    cudaFree((int*)device->d_globalVertexId);
    cudaFree(device->d_ext);
    cudaFree(device->d_extensions);
    // cudaFree(device->d_extensionsQuick);
    cudaFree(device->d_extensionSources);
    cudaFree(device->d_extensionSourcesOffset);
    cudaFree(device->d_updateCompactionCounters);
    cudaFree(device->d_buffer);
    cudaFree(device->d_bufferCounter);
    cudaFree(device->d_chunksStatus);
    cudaFree(device->d_localSubgraphInduction);
    cudaFree(device->d_quickToCgLocal);
    cudaFree(device->d_hashPerWarp);

    cudaFree((bool*)device->d_stop);
    cudaFree(device->d_currentPos);

    free(h_degree);
    free(h_vertexOffset);
    free(h_adjacencyList);
    free(h_embeddings->id);
    free(h_embeddings->numberOfExtensions);
    free(h_embeddings);
    free(h_extensions->extensions);
    // free(h_extensions->extensionsQuick);
    free(h_extensions);
    free(h_extensionsOffset);
    free(h_buffer);
    free(h_chunksStatus);
    free(h_currentPos);
    free(h_hashPerWarp);
    free(h_hashGlobal);
    free(h_localSubgraphInduction);

    cudaFreeHost(h_status);
    cudaFreeHost(h_smid);
    cudaFreeHost(h_stop);
    cudaFreeHost(h_result);

    cudaStreamDestroy(main);
    cudaStreamDestroy(memory);
    cudaStreamDestroy(bufferStream);

    delete mainGraph;
    delete timer;
    if(getcanon)
        delete readGpuBufferThread;


    for(int i = 0 ; i < numberOfWorkerThreads ; i++) {
        delete[] chunksWorker[i];
        if(getcanon)
            delete workersThread[i];
        delete cgCounting[i];
    }

    for(int i = 0 ; i < CHUNKS_CPU ; i++) {
        sem_destroy(&chunksEmptySemaphore[i]);
        sem_destroy(&chunksFullSemaphore[i]);
    }

    delete[] chunksWorker;
    delete[] chunksWorkerSize;
    delete[] chunksEmptySemaphore;
    delete[] chunksFullSemaphore;
    if(getcanon)
        delete[] workersThread;
    delete[] cgCounting;
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
    h_localSubgraphInduction = (long unsigned int*)malloc(numberOfWarps * h_k * sizeof(long unsigned int));

    h_embeddings = (Embeddings*)malloc(sizeof(Embeddings));
    h_embeddings->id = (int*)malloc(numberOfWarps * h_warpSize * sizeof(int));
    h_embeddings->numberOfExtensions = (int*)malloc(numberOfWarps * h_warpSize * sizeof(int));
    h_extensions = (Extensions*)malloc(sizeof(Extensions));
    h_extensions->extensions = (int*)malloc(numberOfWarps * h_extensionsLength * sizeof(int));
    // h_extensions->extensionsQuick = (long unsigned int*)malloc(numberOfWarps * h_extensionsLength * sizeof(long unsigned int));

    gpuErrchk(cudaMallocHost((void**)&h_status, h_numberOfActiveThreads * sizeof(int)));
    gpuErrchk(cudaMallocHost((void**)&h_smid, h_numberOfActiveThreads * sizeof(int)));
    gpuErrchk(cudaMallocHost((void**)&h_stop, sizeof(bool)));
    gpuErrchk(cudaMallocHost((void**)&h_result, numberOfWarps*sizeof(unsigned long)));
    *h_stop = false;
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

    // Sum of PA [2 ... k-1], needed to store each quick pattern
    h_bufferSize = ((h_k+1)*(h_k-2))/2;
    h_bufferSize = pow(2, h_bufferSize);
    h_buffer = (unsigned int*)malloc(CPU_BUFFER_SIZE * h_k * sizeof(unsigned int));
    h_chunksStatus = (int*)malloc(CHUNKS_CPU * sizeof(int));
    h_keepMonitoring = false;

    for(int i = 0 ; i < CHUNKS_CPU ; i++)
        h_chunksStatus[i] = 0;

    memset(h_localSubgraphInduction, 0, numberOfWarps * h_k * sizeof(long unsigned int));
}

void Manager::initializeDeviceDataStructures() {
    gpuErrchk(cudaMalloc((void**)&device->d_vertexOffset, (mainGraph->getMaxVertexId()+2)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_adjacencyList, (mainGraph->getNumberOfEdges()*2 + (mainGraph->getMaxVertexId()+1)) * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_degree, (mainGraph->getMaxVertexId()+1)*sizeof(int)));

    gpuErrchk(cudaMalloc((void**)&device->d_embeddings, sizeof(Embeddings)));
    gpuErrchk(cudaMalloc((void**)&device->d_id, numberOfWarps * h_warpSize * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_numberOfExtensions, numberOfWarps * h_warpSize * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_result, numberOfWarps * sizeof(unsigned long)));
    gpuErrchk(cudaMalloc((void**)&device->d_status, h_numberOfActiveThreads * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_smid, h_numberOfActiveThreads * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_globalVertexId, sizeof(int)));

    gpuErrchk(cudaMalloc((void**)&device->d_ext, sizeof(Extensions)));
    gpuErrchk(cudaMalloc((void**)&device->d_extensions, numberOfWarps * h_extensionsLength * sizeof(int)));
    // gpuErrchk(cudaMalloc((void**)&device->d_extensionsQuick, numberOfWarps * h_extensionsLength * sizeof(long unsigned int)));
    gpuErrchk(cudaMalloc((void**)&device->d_extensionSources, numberOfWarps * h_extensionsLength * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_extensionSourcesOffset, numberOfWarps * h_k * h_k * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_updateCompactionCounters, numberOfWarps * h_k * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_buffer, GPU_BUFFER_SIZE * h_k * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void**)&device->d_bufferCounter, sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void**)&device->d_chunksStatus, CHUNKS_GPU * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_localSubgraphInduction, numberOfWarps * h_k * sizeof(long unsigned int)));
    gpuErrchk(cudaMalloc((void**)&device->d_hashPerWarp, numberOfWarps * quickMapping->numberOfCgs * sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc((void**)&device->d_stop, sizeof(bool)));
    gpuErrchk(cudaMalloc((void**)&device->d_currentPos, numberOfWarps * sizeof(int)));

    gpuErrchk(cudaMalloc((void**)&device->d_k, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_extensionsLength, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_warpSize, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_extensionsOffset, (h_k-1) * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_quickToCgLocal, quickMapping->numberOfQuicks * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc((void**)&device->d_numberOfActiveThreads, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_maxVertexId, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_maxDegree, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_numberOfCgs, sizeof(int)));

    gpuErrchk(cudaMalloc((void**)&d_device, sizeof(Device)));




    gpuErrchk(cudaMemcpy(&(device->d_embeddings->id), &device->d_id, sizeof(device->d_embeddings->id), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&(device->d_embeddings->numberOfExtensions), &device->d_numberOfExtensions, sizeof(device->d_embeddings->numberOfExtensions), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(&(device->d_ext->extensions), &device->d_extensions, sizeof(device->d_ext->extensions), cudaMemcpyHostToDevice));
    // gpuErrchk(cudaMemcpy(&(device->d_ext->extensionsQuick), &device->d_extensionsQuick, sizeof(device->d_ext->extensionsQuick), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&(device->d_ext->extensionSources), &device->d_extensionSources, sizeof(device->d_ext->extensionSources), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&(device->d_ext->extensionSourcesOffset), &device->d_extensionSourcesOffset, sizeof(device->d_ext->extensionSourcesOffset), cudaMemcpyHostToDevice));

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
    gpuErrchk(cudaMemcpy(device->d_quickToCgLocal, quickMapping->quickToCgLocal, quickMapping->numberOfQuicks * sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_numberOfActiveThreads, &h_numberOfActiveThreads, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_maxVertexId, &h_maxVertexId, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_maxDegree, &h_maxDegree, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_numberOfCgs, &(quickMapping->numberOfCgs), sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemset(device->d_status, 0, h_numberOfActiveThreads * sizeof(int)));
    gpuErrchk(cudaMemset(device->d_smid, 0, h_numberOfActiveThreads * sizeof(int)));
    gpuErrchk(cudaMemset(device->d_bufferCounter, 0, sizeof(unsigned int)));
    gpuErrchk(cudaMemset(device->d_chunksStatus, 0, CHUNKS_GPU * sizeof(int)));
    gpuErrchk(cudaMemset(device->d_localSubgraphInduction, 0, numberOfWarps * h_k * sizeof(long unsigned int)));
    gpuErrchk(cudaMemset(device->d_hashPerWarp, 0, numberOfWarps * quickMapping->numberOfCgs * sizeof(unsigned long long)));

    gpuErrchk(cudaMemcpy(d_device, device, sizeof(Device), cudaMemcpyHostToDevice));
    /* Workaround to allow global __device__ variables */
    // gpuErrchk(cudaMemcpyToSymbol(device->ds_vertexOffset, &device->d_vertexOffset, sizeof(device->d_vertexOffset)));
    // gpuErrchk(cudaMemcpyToSymbol(device->ds_adjacencyList, &device->d_adjacencyList, sizeof(device->d_adjacencyList)));
    // gpuErrchk(cudaMemcpyToSymbol(device->ds_degree, &device->d_degree, sizeof(device->d_degree)));
    // gpuErrchk(cudaMemcpyToSymbol(device->ds_embeddings, &device->d_embeddings, sizeof(device->d_embeddings)));
    // gpuErrchk(cudaMemcpyToSymbol(device->ds_extensions, &device->d_ext, sizeof(device->d_ext)));


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

void Manager::shortIdlenessReport() {
    std::time_t result = std::time(nullptr);
    printf("[REBALANCING] warps idle: %d, numberOfWarps: %d, %f warps active, %s", amountWarpsIdle, numberOfWarps, 100-percentageWarpsIdle, std::asctime(std::localtime(&result)));
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
        std::cout << currentRound << "," << smId << "," << smOccupancy[smId] << "\n";
    }

}

void Manager::startTimer() {
    timer->play();
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
    copyWarpDataFromGpu();
    copyResult();
    std::vector<int> idles, indifferents;
    std::queue<Donator*> actives;
    organizeThreadStatus(&idles, &actives, &indifferents);
    if(actives.size() > 0) {
        donate(&idles, &actives);
        copyWarpDataBackToGpu();
        return true;
    }
    else {
        invalidateResult();
        return false;
    }
}

void Manager::copyResult() {

    gpuErrchk(cudaMemcpy(h_result, device->d_result, numberOfWarps*sizeof(unsigned long), cudaMemcpyDeviceToHost));
    for(int i = 0 ; i < numberOfWarps ; i++) {
        result += h_result[i];
    }

    // gpuErrchk(cudaMemcpy(h_buffer, device->d_buffer, result * h_k * sizeof(unsigned int), cudaMemcpyDeviceToHost));
}

void Manager::aggregate() {
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
    for(int i = 0 ; i < quickMapping->numberOfCgs ; i++)
        validSubgraphs += h_hashGlobal[i];

    // for(int i = 0 ; i < result ; i++) {
    //     for(int j = 0 ; j < h_k-1 ; j++) {
    //         printf("%d,", h_buffer[i*h_k+j]);
    //     }
    //     printf("%d\n", h_buffer[((i+1)*h_k)-1]);
    // }
    // canonicalizeBufferSerial();

    printf("Total number of motifs: %lu\n", validSubgraphs);
    // for(int i = 0 ; i < h_bufferCounter ; i++) {
    //     for(int k = 0 ; k < h_k ; k++) {
    //         printf("%d ", h_buffer[i*h_k + k]);
    //     }
    //     printf("\n");
    // }
}

void Manager::printCount() {
    printf("%lu\n", result);
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
    //      - h_embeddings -> id
    //      - h_embeddings -> numberOfExtensions
    //      - h_extensions -> extensions
    //      - h_status
    //      - h_currentPos
    //      - h_extensionsOffset (don't have to be copied, 'cause it's calculated previously)
    //      - h_result

    // Consistent copy of h_status array
    // std::cout << "Copying result from device...\n";
    gpuErrchk(cudaMemcpy(h_embeddings->id, device->d_id, numberOfWarps * h_warpSize * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_embeddings->numberOfExtensions, device->d_numberOfExtensions, numberOfWarps * h_warpSize * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_extensions->extensions, device->d_extensions, numberOfWarps * h_extensionsLength * sizeof(int), cudaMemcpyDeviceToHost));
    // gpuErrchk(cudaMemcpy(h_extensions->extensionsQuick, device->d_extensionsQuick, numberOfWarps * h_extensionsLength * sizeof(long unsigned int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_status, device->d_status, h_numberOfActiveThreads*sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_smid, device->d_smid, h_numberOfActiveThreads*sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_currentPos, device->d_currentPos, numberOfWarps * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_localSubgraphInduction, device->d_localSubgraphInduction, numberOfWarps * h_k * sizeof(long unsigned int), cudaMemcpyDeviceToHost));

}

void Manager::invalidateResult() {
    gpuErrchk(cudaMemset(device->d_result, 0, numberOfWarps*sizeof(unsigned long)));
}

void Manager::copyWarpDataBackToGpu() {
    //  Written
    //      - h_embeddings->id
    //      - h_embeddings->numberOfExtensions
    //      - h_currentPos
    //      - h_stop

    gpuErrchk(cudaMemcpy(device->d_id, h_embeddings->id, numberOfWarps * h_warpSize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_numberOfExtensions, h_embeddings->numberOfExtensions, numberOfWarps * h_warpSize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_currentPos, h_currentPos, numberOfWarps * sizeof(int), cudaMemcpyHostToDevice));
    *h_stop = false;
    gpuErrchk(cudaMemcpy((bool*)device->d_stop, h_stop, sizeof(bool), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_localSubgraphInduction, h_localSubgraphInduction, numberOfWarps * h_k * sizeof(long unsigned int), cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
}

void Manager::printQueue(std::queue<Donator*> queue) {
    while(!queue.empty()) {
        Donator* donator = queue.front();
        std::cout << "[ACTIVE]:" << donator->wid << ";" << donator->targetLevel << "\n";
        queue.pop();
    }
}

void Manager::organizeThreadStatus(std::vector<int>* idles, std::queue<Donator*>* actives, std::vector<int>* indifferents) {
    //  Data structures read
    //      - h_status
    //      - h_currentPos
    //      - h_embeddigs->numberOfExtensions

    for(int i = 0, wid ; i < h_numberOfActiveThreads ; i+=h_warpSize) {
        // std::cout << i << "\n";
        wid = (int)(i/h_warpSize);
        if(h_status[i] == 2) {
            // std::cout << "IDLE at all\n";
            idles->push_back(wid);
        }
        else {
            bool active = false;
            int offsetWarp = wid * h_warpSize;
            for(int k = 0 ; k <= h_currentPos[wid] && k < h_k - 2 && !active ; k++) {
                if(h_embeddings->numberOfExtensions[offsetWarp+k] > 0) {
                    active = true;
                    Donator* donator = new Donator;
                    donator->wid = wid;
                    donator->targetLevel = k;
                    actives->push(donator);
                }
            }

            if(!active) {
                // std::cout << "IDLE for no options\n";
                indifferents->push_back(wid);
            }
            // else {
                // std::cout << "ACTIVE\n";
            // }
        }
    }

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

void Manager::donate(std::vector<int>* idles, std::queue<Donator*>* actives) {
    //  Read:
    //      - h_embeddings -> id
    //      - h_embeddings -> numberOfExtensions
    //      - h_extensions -> extensions
    //      - h_extensionsOffset
    //  Written
    //      - h_embeddings->id
    //      - h_embeddings->numberOfExtensions
    //      - h_currentPos

    Donator* donator;
    int offsetWarpIdle, offsetWarpDonator, localOffsetExtensionsDonator, numberOfExtensionsDonator, currentIdle;
    bool active;

    for(int i = 0 ; i < idles->size() && !actives->empty() ; i++) {

        currentIdle = idles->at(i);
        donator = actives->front();
        actives->pop();

        offsetWarpIdle = currentIdle*h_warpSize;
        offsetWarpDonator = donator->wid*h_warpSize;
        localOffsetExtensionsDonator = donator->wid * h_extensionsLength + h_extensionsOffset[donator->targetLevel];

        // std::cout << "[IDLE   ]:\t" << currentIdle << "\n";
        // std::cout << "[DONATOR]:\t" << donator->wid << "," << h_currentPos[donator->wid] << "," << donator->targetLevel << "\n";
        // std::cout << "[DONATOR BEFORE]\n";
        // for(int k = 0 ; k <= h_currentPos[donator->wid] ; k++) {
        //     std::cout << "\t " << h_embeddings->id[offsetWarpDonator+k] << "," << h_embeddings->numberOfExtensions[offsetWarpDonator+k] << "\n";
        //     for(int j = 0 ; j < h_embeddings->numberOfExtensions[offsetWarpDonator+k] ; j++)
        //         std::cout << "\t\t" << h_extensions->extensions[donator->wid * h_extensionsLength + h_extensionsOffset[k] + j] << "\n";
        // }
        // debug("DONATOR BEFORE FIRST DONATION...\n");

        for(int k = 0 ; k <= donator->targetLevel ; k++) {
            h_embeddings->id[offsetWarpIdle+k] = h_embeddings->id[offsetWarpDonator+k];
            h_localSubgraphInduction[donator->wid*h_k+k] = h_localSubgraphInduction[donator->wid*h_k+k];
            h_embeddings->numberOfExtensions[offsetWarpIdle+k] = 0;
        }
        numberOfExtensionsDonator = h_embeddings->numberOfExtensions[offsetWarpDonator+donator->targetLevel];
        h_embeddings->id[offsetWarpIdle+donator->targetLevel+1] = h_extensions->extensions[localOffsetExtensionsDonator+numberOfExtensionsDonator-1];
        h_embeddings->numberOfExtensions[offsetWarpDonator+donator->targetLevel]--;
        h_embeddings->numberOfExtensions[offsetWarpIdle+donator->targetLevel+1] = -1;
        h_currentPos[currentIdle] = donator->targetLevel+1;

        // std::cout << "[IDLE AFTER]" << "," << h_currentPos[currentIdle] << "\n";
        // for(int k = 0 ; k <= h_currentPos[currentIdle] ; k++) {
        //     std::cout << "\t " << h_embeddings->id[offsetWarpIdle+k] << "," << h_embeddings->numberOfExtensions[offsetWarpIdle+k] << "\n";
        //     for(int j = 0 ; j < h_embeddings->numberOfExtensions[offsetWarpIdle+k] ; j++)
        //         std::cout << "\t\t" << h_extensions->extensions[currentIdle * h_extensionsLength + h_extensionsOffset[k] + j] << "\n";
        // }

        // debug("IDLE AFTER DONATION...\n");

        // std::cout << "[DONATOR AFTER]\n";
        // for(int k = 0 ; k <= h_currentPos[donator->wid] ; k++) {
        //     std::cout << "\t " << h_embeddings->id[offsetWarpDonator+k] << "," << h_embeddings->numberOfExtensions[offsetWarpDonator+k] << "\n";
        //     for(int j = 0 ; j < h_embeddings->numberOfExtensions[offsetWarpDonator+k] ; j++)
        //         std::cout << "\t\t" << h_extensions->extensions[donator->wid * h_extensionsLength + h_extensionsOffset[k] + j] << "\n";
        // }
        // debug("DONATOR AFTER DONATION...\n");

        // Remove donator or push back to the queue
        active = false;
        for(int k = donator->targetLevel ; k <= h_currentPos[donator->wid] && k < h_k - 2 && !active ; k++)
            if(h_embeddings->numberOfExtensions[offsetWarpDonator+k] > 0) {
                active = true;
                donator->targetLevel = k;
                actives->push(donator);
            }
        if(!active)
            free(donator);
    }
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

void* Manager::readGpuBufferFunction(Manager* manager) {

    int currentChunkStatus, currentGpuChunk = 0, currentCpuChunk = 0;
    while(!(manager->gpuFinished)) {
        gpuErrchk(cudaMemcpyAsync(&currentChunkStatus, manager->device->d_chunksStatus+currentGpuChunk, sizeof(int), cudaMemcpyDeviceToHost, manager->bufferStream));
        gpuErrchk(cudaStreamSynchronize(manager->bufferStream));

        if(currentChunkStatus == CHUNK_SIZE) {
            sem_wait(&(manager->chunksEmptySemaphore[currentCpuChunk]));

            gpuErrchk(cudaMemcpyAsync(manager->h_buffer + currentCpuChunk * CHUNK_SIZE , manager->device->d_buffer + currentGpuChunk * CHUNK_SIZE, CHUNK_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost, manager->bufferStream));
            gpuErrchk(cudaStreamSynchronize(manager->bufferStream));

            currentChunkStatus = 0;
            gpuErrchk(cudaMemcpyAsync(manager->device->d_chunksStatus+currentGpuChunk, &currentChunkStatus, sizeof(int), cudaMemcpyHostToDevice, manager->bufferStream));
            gpuErrchk(cudaStreamSynchronize(manager->bufferStream));

            manager->h_chunksStatus[currentCpuChunk] = CHUNK_SIZE;

            sem_post(&(manager->chunksFullSemaphore[currentCpuChunk]));

            // std::cout << "[readGpuBufferFunction][ORDINARY] GPU Chunk " << currentGpuChunk << " (" << manager->h_chunksStatus[currentCpuChunk] << " subgraphs) consumed by CPU Chunk " << currentCpuChunk << "\n";

            currentGpuChunk = (currentGpuChunk + 1) % CHUNKS_GPU;
            currentCpuChunk = (currentCpuChunk + 1) % CHUNKS_CPU;
        }
    }


    while(true) {
        gpuErrchk(cudaMemcpyAsync(&currentChunkStatus, manager->device->d_chunksStatus+currentGpuChunk, sizeof(int), cudaMemcpyDeviceToHost, manager->bufferStream));
        gpuErrchk(cudaStreamSynchronize(manager->bufferStream));

        if(currentChunkStatus > 0 && currentChunkStatus <= CHUNK_SIZE) {
            sem_wait(&(manager->chunksEmptySemaphore[currentCpuChunk]));

            gpuErrchk(cudaMemcpyAsync(manager->h_buffer + currentCpuChunk * CHUNK_SIZE , manager->device->d_buffer + currentGpuChunk * CHUNK_SIZE, CHUNK_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost, manager->bufferStream));
            gpuErrchk(cudaStreamSynchronize(manager->bufferStream));

            manager->h_chunksStatus[currentCpuChunk] = currentChunkStatus;

            currentChunkStatus = 0;
            gpuErrchk(cudaMemcpyAsync(manager->device->d_chunksStatus+currentGpuChunk, &currentChunkStatus, sizeof(int), cudaMemcpyHostToDevice, manager->bufferStream));
            gpuErrchk(cudaStreamSynchronize(manager->bufferStream));

            sem_post(&(manager->chunksFullSemaphore[currentCpuChunk]));

            // std::cout << "[readGpuBufferFunction][RESIDUAL] GPU Chunk " << currentGpuChunk << " (" << manager->h_chunksStatus[currentCpuChunk] << ") consumed by CPU Chunk " << currentCpuChunk << "\n";

            currentCpuChunk = (currentCpuChunk + 1) % CHUNKS_CPU;
            currentGpuChunk = (currentGpuChunk + 1) % CHUNKS_GPU;
        }
        else
            break;
    }

    for(int i = 0 ; i < CHUNKS_CPU ; i++) {
        if(manager->h_chunksStatus[i] == 0) {
            sem_wait(&(manager->chunksEmptySemaphore[i]));
            manager->h_chunksStatus[i] = -1;
            sem_post(&(manager->chunksFullSemaphore[i]));
        }
    }

    std::cout << "[readGpuBufferFunction] FINISHED!\n";
}

void* Manager::induceCanonicalizeFunction(Manager* manager, int tid) {
    // Consume a job, if there's one
    int myCurrentChunk = tid;
    double waitTime, induceCanonicalizeTime;
    unsigned int quick, cg;
    std::unordered_map<unsigned int,unsigned int> *quickToCgMap = manager->quickToCgMap;
    std::unordered_map<unsigned int,unsigned long int> *cgCounting = manager->cgCounting[tid];

    while(manager->h_chunksStatus[myCurrentChunk] != -1) {
        Timer *waitTime = new Timer(), *induceCanonicalizeTime = new Timer();
        waitTime->play();

        sem_wait(&(manager->chunksFullSemaphore[myCurrentChunk]));

        if(manager->h_chunksStatus[myCurrentChunk] != -1) {
            memcpy(manager->chunksWorker[tid], manager->h_buffer + myCurrentChunk *CHUNK_SIZE, manager->h_chunksStatus[myCurrentChunk] * sizeof(unsigned int));
            manager->chunksWorkerSize[tid] = manager->h_chunksStatus[myCurrentChunk];
            manager->h_chunksStatus[myCurrentChunk] = 0;
            sem_post(&(manager->chunksEmptySemaphore[myCurrentChunk]));

            waitTime->pause();

            for(int i = 0 ; i < manager->chunksWorkerSize[tid] ; i++) {
                quick = manager->chunksWorker[tid][i];
                cg = (*quickToCgMap)[quick];

                if(cgCounting->find(cg) == cgCounting->end())
                    (*cgCounting)[cg] = 1;
                else
                    (*cgCounting)[cg]++;
            }

            if(manager->chunksWorkerSize[tid] < CHUNK_SIZE) {
                break;
            }
            myCurrentChunk = (myCurrentChunk+manager->numberOfWorkerThreads) % CHUNKS_CPU;
        }

        delete waitTime;
        delete induceCanonicalizeTime;
    }

    long unsigned int total = 0;
    for(auto it = cgCounting->begin() ; it != cgCounting->end() ; ++it) {
        // printf("[WORKER] %u -> %lu\n", it->first, it->second);
        total += it->second;
    }
    // printf("[WORKER][%d] total: %ld\n", tid, total);
    // std::cout << "[WORKER][" << tid << "] exiting...\n";
}

void Manager::canonicalize(unsigned int *buffer, Graph* mainGraph, std::unordered_map<std::string,int>* patternCounting, int h_k, int numberOfSubgraphs) {
    printf("[canonicalize][numberOfSubgraphs] %d\n", numberOfSubgraphs);

    std::unordered_map<int,int> relabeling;
    std::string pattern;

    int m, n;
    n = h_k;
    m = SETWORDSNEEDED(n);

    graph g[MAXN*MAXM], cg[MAXN*MAXM];
    int lab[MAXN],ptn[MAXN],orbits[MAXN];
    static DEFAULTOPTIONS_GRAPH(options);
    statsblk stats;

    options.getcanon = TRUE;

    nauty_check(WORDSIZE,m,n,NAUTYVERSIONID);

    for(int i = 0, offset ; i < numberOfSubgraphs ; i++) {
        EMPTYGRAPH(g,m,n);
        EMPTYGRAPH(cg,m,n);
        offset = i * h_k;

        ADDONEEDGE(g,0,1,m);
        for (int source = 2, offset = 0; source < n; source++, offset+=source) {
            for(int target = 0 ; target < source ; target++) {
                if(buffer[offset+target] == 1)
                    ADDONEEDGE(g,source,target,m);
            }
        }

        densenauty(g,lab,ptn,orbits,&options,&stats,m,n,cg);


        pattern = generatePattern(cg, m, n);

        if(patternCounting->find(pattern) == patternCounting->end())
            (*patternCounting)[pattern] = 1;
        else
            (*patternCounting)[pattern]++;

        relabeling.clear();
    }
}

void Manager::canonicalizeBufferSerial() {
    Timer* timerC = new Timer();
    Timer* timerI = new Timer();

    printf("[canonicalizeBufferSerial] Begin\n");

    std::string pattern;
    std::unordered_map<std::string,int>* patternCounting = new std::unordered_map<std::string,int>();

    int m, n;
    n = h_k;
    m = SETWORDSNEEDED(n);

    graph g[MAXN*MAXM], cg[MAXN*MAXM];
    int lab[MAXN],ptn[MAXN],orbits[MAXN];
    static DEFAULTOPTIONS_GRAPH(options);
    statsblk stats;

    options.getcanon = TRUE;

    nauty_check(WORDSIZE,m,n,NAUTYVERSIONID);

    for(int i = 0, offset ; i < result ; i++) {
        EMPTYGRAPH(g,m,n);
        EMPTYGRAPH(cg,m,n);
        offset = i * h_k;

        timerI->play();
        ADDONEEDGE(g,0,1,m);
        for(int j = 2 ; j < h_k ; j++) {
            for(int k = 0 ; k < j ; k++) {
                if(mainGraph->areNeighbours(h_buffer[offset+k], h_buffer[offset+j])) {
                    ADDONEEDGE(g,k,j,m);
                }
            }
        }
        timerI->pause();

        timerC->play();
        densenauty(g,lab,ptn,orbits,&options,&stats,m,n,cg);
        timerC->pause();


        pattern = generatePattern(cg, m, n);

        if(patternCounting->find(pattern) == patternCounting->end())
            (*patternCounting)[pattern] = 1;
        else
            (*patternCounting)[pattern]++;

    }

    printf("[canonicalizeBufferSerial] End ; %f,%f = %f\n", timerC->getElapsedTimeInMiliseconds(), timerI->getElapsedTimeInMiliseconds(), timerC->getElapsedTimeInMiliseconds()+timerI->getElapsedTimeInMiliseconds());

    delete patternCounting;
    delete timerC;
    delete timerI;
}

void* Manager::reportFunction(Manager* manager) {
    std::cout << "[ReportFunction] Begin\n";

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

        printf("[REPORT]numberOfWarps:%d,active warps:%d,idle warps:%d,%f%% warps active,", manager->numberOfWarps,manager->numberOfWarps-amountWarpsIdle, amountWarpsIdle, 100-percentageWarpsIdle);
        int totalWarps = 0;
        for(int i = 0 ; i < manager->numberOfSMs ; i++) {
            printf("%d,", smOccupancy[i]);
            totalWarps += smOccupancy[i];
        }
        printf("%d,%lu,%s",totalWarps,totalResult,std::asctime(std::localtime(&result)));

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

    std::cout << "[ReportFunction] End " << "\n";
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
