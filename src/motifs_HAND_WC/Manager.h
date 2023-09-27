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
        class Comparator {
            public: 
            
            bool operator()(Donator* s1, Donator* s2) {
                return s1->weight < s2->weight;    
            }
        };        

        // auto sortWeight = [] (Donator* s1, Donator* s2) -> bool
        // {
        //     return s1->weight > s2->weight;
        // };
        // auto sortCurrentWeight = [] (Donator* s1, Donator* s2) -> bool
        // {
        //     return s1->currentWeight > s2->currentWeight;
        // };

        // Graph
        Graph* mainGraph;
        const char *graphFile;
        int *h_vertexOffset;
        int *h_adjacencyList;
        int *h_degree;

        // Thread monitoring/sync
        int *h_status;
        int *h_smid;
        bool *h_stop;
        int *h_bufferFull;
        int *h_bufferDrain;
        std::vector<int> weights;

        // Enumeration data structures
        int *h_id;
        int *h_numberOfExtensions;
        int *h_jobs;
        int *h_inductions;
        int *h_currentPosOfJob;
        int *h_validJobs;
        int *h_currentJob;
        int *h_extensions;
        int *h_extensionsOffset;
        int *h_currentPos;
        int h_induce;
        int h_globalVertexId;
        unsigned long* h_result;
        unsigned long h_subgraphsProcessed;
        long unsigned int* h_amountNewVertices;
        long unsigned int* h_removedEdges;
        long unsigned int* h_addedEdges;
        unsigned long result;
        long unsigned int amountNewVertices;
        long unsigned int removedEdges;
        long unsigned int addedEdges;
        unsigned long long *h_hashPerWarp;
        long unsigned int *h_hashGlobal;
        long unsigned int *h_localSubgraphInduction;

        int *h_buffer;
        unsigned int *h_offsetBuffer;

        // Constants
        int h_k;
        int h_numberOfActiveThreads;
        int h_maxVertexId;
        int h_maxDegree;
        int h_extensionsLength;
        int h_warpSize;
        int h_initialJobsPerWarp;
        int h_jobsPerWarp;
        int h_theoreticalJobsPerWarp;

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

        Manager(const char* graphFile, int k, int numberOfActiveThreads, int blockSize, void (*kernel)(Device*), int numberOfSMs, int reportInterval, int jobsPerWarp, int induce);
        Manager(Graph* graph, int k, int numberOfActiveThreads, int blockSize, void (*kernel)(Device*), int numberOfSMs, int reportInterval, int jobsPerWarp, int induce);
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
        void printCount();
        void loadGpuThreadStatus();
        bool thereAreActiveGpuThreads();
        int amountActiveGpuThreads();
        double percentageWarpIdleness();
        void sleepFor(int millisecs);
        bool gpuIsIdle(int threshold);

        void detailedIdlenessReport();
        void shortIdlenessReport(const char* message);
        void smOccupancyReport();
        void queuesReport();


        void stopKernel();
        bool rebalance();
        void copyWarpDataFromGpu();
        void copyWarpDataBackToGpu();
        void invalidateResult();
        int organizeThreadStatus(std::vector<int>* idles, std::priority_queue<Donator*, std::vector<Donator*>, Comparator>* actives, std::vector<int>* indifferents);
        bool donate(std::vector<int>* idles, std::priority_queue<Donator*, std::vector<Donator*>, Comparator>* actives, int totalWeight);
        void debug(const char* message);
        void readQuickToCgMap();
        void generateQuickToCgMap();
        unsigned int generateQuickCg(graph* g, int k, int m, int n);
        unsigned int generateQuickG(graph* g, int k, int m, int n);
        void printBinaryLong(graph value);
        void check(int flag);
        int bufferDrain();
        int bufferFull();

        void processEdgeWeights();
        void processCompressionScore();
        void compressionResults();

        static std::string generatePattern(graph* g, int m, int n);
        static void* reportFunction(Manager *manager);

};

#endif
