#include <cuda_runtime.h>
#include <vector>
#include <queue>
#include "Structs.cu"
#include "Graph.h"
#include "EnumerationHelper.h"
#include "QuickMapping.h"

#ifndef __DUMATO_CPU_H__
#define __DUMATO_CPU_H__

class DuMatoCPU {
    public:
        typedef struct {
            int wid;
            int targetLevel;
            int weight;
        } Donator;

        class Comparator {
            public: 
            
            bool operator()(Donator* s1, Donator* s2) {
                return s1->weight < s2->weight;    
            }
        };

        /************************************/
        /*           CPU Pointers           */
        const char *datasetName;
        Graph *graphReader;
        DataCPU *dataCPU;
        QuickMapping *quickMapping;
        bool relabeling;
        bool patternAware;
        /***********************************/

        /************************************/
        /*           GPU Pointers           */
        DataGPU *dataGPU, *d_dataGPU;
        void (*kernel)(DataGPU*);

        cudaStream_t main, memory;
        /***********************************/

        DuMatoCPU(const char *datasetName, int k, int numberOfActiveThreads, int blockSize, int numberOfSMs, int jobsPerWarp, void (*kernel)(DataGPU*), int globalThreshold, bool relabeling, bool patternAware);
        DuMatoCPU(Graph *graphReader, int k, int numberOfActiveThreads, int blockSize, int numberOfSMs, int jobsPerWarp, void (*kernel)(DataGPU*), int globalThreshold, bool relabeling, bool patternAware);
        void runKernel();
        void waitKernel();
        void copyWarpDataBackToGpu();
        void copyWarpDataFromGpu();
        void loadGpuOccupancyStatus();
        int organizeThreadStatus(std::vector<int>* idles, std::priority_queue<Donator*, std::vector<Donator*>, Comparator>* actives, std::vector<int>* indifferents);
        bool donate(std::vector<int>* idles, std::priority_queue<Donator*, std::vector<Donator*>, Comparator>* actives, int totalWeight);
        bool gpuIsIdle();
        void stopKernel();
        void sleepFor(int millisecs);
        void outputAggregateCounter();
        void outputAggregatePattern();
        void copyResult();
        void invalidateResult();
        void validateAggregateCounter();
        void validateAggregatePattern();
        bool rebalance();
        Graph *getGraphReader();
        ~DuMatoCPU();
    
    private:
        void initializeCpuDataStructures();
        void releaseCpuDataStructures();
        void initializeGpuDataStructures();
        void releaseGpuDataStructures();
};

#endif