#include <cuda_runtime.h>
#include <vector>
#include <queue>
#include <set>
#include <thread>
#include "Graph.h"
#include "CudaHelperFunctions.h"
#include "Structs.h"
#include "Device.h"
#include "Timer.h"

#ifndef MANAGER_H
#define MANAGER_H

class Manager {
    public:
        /**************/
        /* Attributes */
        /**************/

        // Graph
        Graph* graph;
        int *h_vertexOffset;
        int *h_adjacencyList;
        int *h_degree;

        // Thread monitoring/sync
        int *h_status;
        int *h_smid;
        bool *h_stop;
        bool h_keepMonitoring;

        // Enumeration data structures
        Embeddings* h_embeddings;
        Extensions* h_extensions;
        int *h_extensionsOffset;
        int h_globalVertexId;
        unsigned long* h_result;
        unsigned long result;

        // Constants
        int h_k;
        int h_numberOfActiveThreads;
        int h_maxVertexId;
        int h_maxDegree;
        int h_extensionsLength;
        int h_warpSize, h_virtualWarpSize;

        // Streams
        cudaStream_t main, memory;

        // Others
        int blockSize;
        int numberOfBlocks;
        int warpsPerBlock;
        int numberOfWarps;
        int numberOfSMs;
        int globalThreshold;

        Device *device, *d_device;

        // Kernel function
        void (*kernel)(Device*);

        // Load balancing
        int activeThreads;
        int amountThreadsIdle;
        int amountWarpsIdle;
        double percentageWarpsIdle;
        int round;

        // Timers
        Timer* timer;

        /***********/
        /* Methods */
        /***********/

        Manager(const char* graphFile, int k, int numberOfActiveThreads, int blockSize, void (*kernel)(Device*));
        ~Manager();

        void initializeHostDataStructures();
        void initializeDeviceDataStructures();
        void prepareDataStructures();

        void startTimer();
        void stopTimer();
        double getRuntimeInMilliseconds();
        double getRuntimeInSeconds();

        void runKernel();
        void waitKernel();
        void copyResult();
        void printResult();
        void loadGpuThreadStatus();
        bool thereAreActiveGpuThreads();
        int amountActiveGpuThreads();
        double percentageWarpIdleness();
        void sleepFor(int millisecs);
        bool gpuIsIdle(int threshold);

        void detailedIdlenessReport();
        void shortIdlenessReport();


        void invalidateResult();
        void printQueue(queue<Donator*> queue);
        void debug(const char* message);

};

#endif
