#ifndef MANAGER_H
#define MANAGER_H

#include <cuda_runtime.h>
#include <vector>
#include <queue>
#include <unordered_map>
#include <thread>
#include <semaphore.h>

#include "Graph.h"
#include "Structs.h"
#include "Device.h"
#include "Timer.h"
#include "QuickMapping.h"
#define MAXN 10
#include "nauty.h"

class Manager {
    public:

        /**************/
        /* Attributes */
        /**************/

        // Graph
        Graph* mainGraph;
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
        int *h_currentPos;
        unsigned int*h_buffer;
        unsigned long h_bufferSize;
        int *h_chunksStatus;
        int h_globalVertexId;
        unsigned long* h_result;
        unsigned long result;
        unsigned long long *h_hashPerWarp;
        long unsigned int *h_hashGlobal;
        long unsigned int *h_localSubgraphInduction;

        // Constants
        int h_k;
        int h_numberOfActiveThreads;
        int h_maxVertexId;
        int h_maxDegree;
        int h_extensionsLength;
        int h_warpSize, h_virtualWarpSize;

        // Streams
        cudaStream_t main, memory, bufferStream;

        // Others
        int blockSize;
        int numberOfBlocks;
        int warpsPerBlock;
        int numberOfWarps;
        int numberOfSMs;
        int *smOccupancy;
        int globalThreshold;
        int currentRound;

        // Report
        int reportInterval;
        std::thread *reportThread;

        // Pattern counting
        std::unordered_map<unsigned int,long unsigned int> **cgCounting;
        std::unordered_map<unsigned int,unsigned int> *quickToCgMap;
        std::set<unsigned int> *cgs;
        int numberOfWorkerThreads;
        bool getcanon;

        QuickMapping* quickMapping;

        bool gpuFinished;

        // Thread 1: read subgraphs from gpu
        std::thread *readGpuBufferThread, **workersThread;

        // Thread(s) 2: induce subgraphs and canonicalize
        unsigned int **chunksWorker;
        int *chunksWorkerSize;
        sem_t *chunksEmptySemaphore;
        sem_t *chunksFullSemaphore;


        Device *device, *d_device;

        // Kernel function
        void (*kernel)(Device*);

        // Load balancing
        int activeThreads;
        int amountThreadsIdle;
        int amountWarpsIdle;
        double percentageWarpsIdle;

        // Timers
        Timer* timer;

        /***********/
        /* Methods */
        /***********/

        Manager(const char* graphFile, int k, int numberOfActiveThreads, int blockSize, void (*kernel)(Device*), bool getcanon, int numberOfWorkerThreads, int numberOfSMs, int reportInterval);
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
        void aggregate();
        void printResult();
        void loadGpuThreadStatus();
        bool thereAreActiveGpuThreads();
        int amountActiveGpuThreads();
        double percentageWarpIdleness();
        void sleepFor(int millisecs);
        bool gpuIsIdle(int threshold);

        void detailedIdlenessReport();
        void shortIdlenessReport();
        void smOccupancyReport();


        void stopKernel();
        bool rebalance();
        void copyWarpDataFromGpu();
        void copyWarpDataBackToGpu();
        void invalidateResult();
        void printQueue(std::queue<Donator*> queue);
        void organizeThreadStatus(std::vector<int>* idles, std::queue<Donator*>* actives, std::vector<int>* indifferents);
        void donate(std::vector<int>* idles, std::queue<Donator*>* actives);
        void debug(const char* message);
        void canonicalizeP();
        void readQuickToCgMap();
        void generateQuickToCgMap();
        unsigned int generateQuickCg(graph* g, int k, int m, int n);
        unsigned int generateQuickG(graph* g, int k, int m, int n);
        void printBinaryLong(graph value);

        void canonicalizeBufferSerial();
        static std::string generatePattern(graph* g, int m, int n);
        static void canonicalize(unsigned int *buffer, Graph* mainGraph, std::unordered_map<std::string,int>* patternCounting, int h_k, int numberOfSubgraphs);
        static void* readGpuBufferFunction(Manager *manager);
        static void* induceCanonicalizeFunction(Manager* manager, int tid);
        static void* reportFunction(Manager *manager);

};

#endif
