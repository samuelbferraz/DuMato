#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <queue>
#include <vector>
#include "Manager.h"
#include "Timer.h"

using namespace std;

Manager::Manager(const char* graphFile, int k, int numberOfActiveThreads, int blockSize, void (*kernel)(Device*)) {
    this->graph = new Graph(graphFile);

    cout << graph->getMaxDegree() << "\n";

    this->h_numberOfActiveThreads = numberOfActiveThreads;
    this->h_k = k;
    this->blockSize = blockSize;
    this->result = 0;
    this->activeThreads = 0;
    this->percentageWarpsIdle = 0;
    this->kernel = kernel;
    this->timer = new Timer();
    this->round = 0;

    h_maxVertexId = graph->getMaxVertexId();
    h_maxDegree = graph->getMaxDegree();

    // cout << "Done!" << "\n";

    h_warpSize = 32;
    h_virtualWarpSize = 1;
    // cout << h_numberOfActiveThreads << " active threads\n";
    numberOfBlocks = ceil(h_numberOfActiveThreads/(float)blockSize);
    warpsPerBlock = blockSize / h_virtualWarpSize;
    numberOfWarps = numberOfBlocks * warpsPerBlock;

    this->device = (Device*)malloc(sizeof(Device));

    gpuErrchk(cudaStreamCreate(&main));
    gpuErrchk(cudaStreamCreate(&memory));

    prepareDataStructures();
}

Manager::~Manager() {
    cudaFree(device->d_degree);
    cudaFree(device->d_vertexOffset);
    cudaFree(device->d_adjacencyList);
    cudaFree(device->d_id);
    cudaFree(device->d_numberOfExtensions);
    cudaFree(device->d_traversedExtensions);
    cudaFree(device->d_embeddings);
    cudaFree(device->d_result);
    cudaFree((int*)device->d_globalVertexId);

    free(h_degree);
    free(h_vertexOffset);
    free(h_adjacencyList);
    free(h_embeddings->id);
    free(h_embeddings->numberOfExtensions);
    free(h_embeddings->traversedExtensions);
    free(h_embeddings);
    free(h_result);
    free(h_extensionsOffset);

    cudaStreamDestroy(main);
    cudaStreamDestroy(memory);

    delete graph;
    delete timer;
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
    h_vertexOffset = (int*)malloc((graph->getMaxVertexId()+2)*sizeof(int));
    h_adjacencyList = (int*)malloc((graph->getNumberOfEdges()*2 + (graph->getMaxVertexId()+1)) * sizeof(int));
    h_degree = (int*)malloc((graph->getMaxVertexId()+1)*sizeof(int));
    h_result = (unsigned long*)malloc(numberOfWarps*sizeof(unsigned long));

    h_embeddings = (Embeddings*)malloc(sizeof(Embeddings));
    h_embeddings->id = (int*)malloc(numberOfWarps * h_warpSize * sizeof(int));
    h_embeddings->numberOfExtensions = (int*)malloc(numberOfWarps * h_k * sizeof(int));
    h_embeddings->traversedExtensions = (int*)malloc(numberOfWarps * h_k * sizeof(int));


    int offset = 0;
    for(int vertexId = 0 ; vertexId <= graph->getMaxVertexId() ; vertexId++) {
        h_vertexOffset[vertexId] = offset;
        // printf("%d, vertexOffset: %d\n", vertexId, h_vertexOffset[vertexId]);
        for(set<int>::iterator itEdges = graph->getNeighbours(vertexId)->begin() ; itEdges != graph->getNeighbours(vertexId)->end() ; itEdges++)
            h_adjacencyList[offset++] = *itEdges;
        h_adjacencyList[offset++] = -1;

        h_degree[vertexId] = graph->getNeighbours(vertexId)->size();
        // printf("%d, vertexOffset: %d, degree: %d\n", vertexId, h_vertexOffset[vertexId], h_degree[vertexId]);
    }
    h_vertexOffset[graph->getMaxVertexId()+1] = h_vertexOffset[graph->getMaxVertexId()]+h_degree[graph->getMaxVertexId()]+1;

    h_keepMonitoring = false;
}

void Manager::initializeDeviceDataStructures() {
    gpuErrchk(cudaMalloc((void**)&device->d_vertexOffset, (graph->getMaxVertexId()+2)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_adjacencyList, (graph->getNumberOfEdges()*2 + (graph->getMaxVertexId()+1)) * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_degree, (graph->getMaxVertexId()+1)*sizeof(int)));

    gpuErrchk(cudaMalloc((void**)&device->d_embeddings, sizeof(Embeddings)));
    gpuErrchk(cudaMalloc((void**)&device->d_id, numberOfWarps * h_k * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_numberOfExtensions, numberOfWarps * h_k * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_traversedExtensions, numberOfWarps * h_k * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_result, numberOfWarps * sizeof(unsigned long)));
    gpuErrchk(cudaMalloc((void**)&device->d_globalVertexId, sizeof(int)));

    gpuErrchk(cudaMalloc((void**)&device->d_k, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_extensionsLength, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_warpSize, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_virtualWarpSize, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_extensionsOffset, (h_k-1) * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_numberOfActiveThreads, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_maxVertexId, sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&device->d_maxDegree, sizeof(int)));

    gpuErrchk(cudaMalloc((void**)&d_device, sizeof(Device)));

    gpuErrchk(cudaMemcpy(&(device->d_embeddings->id), &device->d_id, sizeof(device->d_embeddings->id), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&(device->d_embeddings->numberOfExtensions), &device->d_numberOfExtensions, sizeof(device->d_embeddings->numberOfExtensions), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&(device->d_embeddings->traversedExtensions), &device->d_traversedExtensions, sizeof(device->d_embeddings->traversedExtensions), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(device->d_vertexOffset, h_vertexOffset, (graph->getMaxVertexId()+2)*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_adjacencyList, h_adjacencyList, (graph->getNumberOfEdges()*2 + (graph->getMaxVertexId()+1)) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_degree, h_degree, (graph->getMaxVertexId()+1)*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy((int*)device->d_globalVertexId, &h_globalVertexId, sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(device->d_k, &h_k, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_extensionsLength, &h_extensionsLength, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_warpSize, &h_warpSize, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_virtualWarpSize, &h_virtualWarpSize, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_extensionsOffset, h_extensionsOffset, (h_k-1) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_numberOfActiveThreads, &h_numberOfActiveThreads, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_maxVertexId, &h_maxVertexId, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device->d_maxDegree, &h_maxDegree, sizeof(int), cudaMemcpyHostToDevice));

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
    int warpsPerBlock = blockSize / h_virtualWarpSize;
    int amountActive = 0;
    cout << warpsPerBlock << ";" << blockSize << ";" << warpsPerBlock << ";" << numberOfBlocks << "\n";

    for(int i = 0, blockIdle, blockActive, blockBusy, warpsBusy ; i < numberOfBlocks ; i++) {
        blockIdle = blockActive = blockBusy = warpsBusy = 0;
        for(int j = 0, warpIdle, warpActive, warpBusy ; j < warpsPerBlock ; j++) {
            warpIdle = warpActive = warpBusy = 0;
            for(int k = 0, offset ; k < h_virtualWarpSize ; k++) {
                offset = i*blockSize+j*h_virtualWarpSize+k;
                if(h_status[offset] == 0)
                    warpActive++;
                else if(h_status[offset] == 1)
                    warpBusy++;
                else if(h_status[offset] == 2)
                    warpIdle++;
                else {
                    cout << "BUUUUUUUUUUUUUGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG\n";
                    exit(1);
                }
            }

            if(warpBusy > 0)
                warpsBusy++;
            //cout << (warpActive/(float)h_virtualWarpSize)*100 << " available | " << (warpBusy/(float)h_virtualWarpSize)*100 << " busy | " << (warpIdle/(float)h_virtualWarpSize)*100 << " idle.\n";

            blockActive+=warpActive;
            blockBusy+=warpBusy;
            blockIdle+=warpIdle;
        }
        if(blockBusy > 0) {
            cout << "Block " << i << " summary: ";
            cout << "Block stats -> " << blockActive << " available (" << (blockActive/(float)blockSize)*100 << " %) | " << blockBusy << " busy (" << (blockBusy/(float)blockSize)*100 << " %) | " << blockIdle << " idle (" << (blockIdle/(float)blockSize)*100 << " %) ; Warp stats -> " << warpsBusy << " warps active | " << (((warpsBusy*32)-blockBusy)/((float)warpsBusy*32))*100 << " % warp idleness.\n";
            amountActive++;
        }
     }
     cout << "-> Amount active: " << amountActive << "\n";
}

void Manager::shortIdlenessReport() {
    printf("(%d | %d) ->  %f\n", amountWarpsIdle, numberOfWarps, 100-percentageWarpsIdle);
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

void Manager::copyResult() {
    gpuErrchk(cudaMemcpy(h_result, device->d_result, numberOfWarps*sizeof(unsigned long), cudaMemcpyDeviceToHost));

    for(int i = 0 ; i < numberOfWarps ; i++) {
        result += h_result[i];
    }
}

void Manager::printResult() {
    printf("%lu\n", result);
}

void Manager::loadGpuThreadStatus() {
    gpuErrchk(cudaMemcpyAsync(h_status, device->d_status, h_numberOfActiveThreads*sizeof(int), cudaMemcpyDeviceToHost, memory));
    gpuErrchk(cudaStreamSynchronize(memory));

    amountThreadsIdle = 0;
    amountWarpsIdle = 0;
    for(int i = 0, intraWarpIdle ; i < numberOfWarps ; i++)
    {
        intraWarpIdle = 0;
        for(int j = 0 ; j < h_virtualWarpSize ; j++) {
            if(h_status[i*h_virtualWarpSize+j] == 2) {
                amountThreadsIdle++;
                intraWarpIdle++;
            }
        }

        if(intraWarpIdle == h_virtualWarpSize)
            amountWarpsIdle++;
    }

    percentageWarpsIdle = (amountWarpsIdle/(double)numberOfWarps)*100;
    activeThreads = h_numberOfActiveThreads - amountThreadsIdle;
}

bool Manager::thereAreActiveGpuThreads() {
    return amountThreadsIdle < h_numberOfActiveThreads;
}

void Manager::sleepFor(int millisecs) {
    std::this_thread::sleep_for(std::chrono::milliseconds(millisecs));
}

bool Manager::gpuIsIdle(int threshold) {
    return percentageWarpsIdle == 100 || percentageWarpsIdle > threshold;
}

int Manager::amountActiveGpuThreads() {
    return activeThreads;
}

double Manager::percentageWarpIdleness() {
    return percentageWarpsIdle;
}

void Manager::debug(const char* message) {
    cout << message << "\n";
    cout << "Debug reached. Type any int value to move on...\n";
    int a;
    cin >> a;
}
