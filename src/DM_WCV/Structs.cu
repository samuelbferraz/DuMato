#include <iostream>
#include <cuda_runtime.h>

#ifndef __STRUCTS_AND_HELPERS__
#define __STRUCTS_AND_HELPERS__

#define gpuErrorCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"cudaAssert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

typedef struct {
    /****************************************/
    /****************Graph*******************/
    int *d_vertexOffset;
    int *d_adjacencyList;
    int *d_degree;
    /****************************************/

    /****************************************/
    /*************Enumeration****************/
    int *d_k;                   // TODO Migrate to __const__
    int *d_id;
    int *d_jobsPerWarp;         // TODO Migrate to __const__
    int *d_jobs;
    int *d_currentJob;
    int *d_currentPosOfJob;
    int *d_validJobs;
    int *d_numberOfExtensions;
    int *d_extensions;
    int *d_currentPos;
    unsigned int *d_warpSize;            // TODO Migrate to __const__
    int *d_extensionsOffset;
    int *d_extensionsLength;    // TODO Migrate to __const__
    unsigned long* d_result;
    volatile bool *d_stop;
    int *d_status;

    unsigned long long *d_hashPerWarp;
    long unsigned int *d_quickToCgLocal;
    int *d_numberOfCgs;
    int *d_extensionSources;
    long unsigned int *d_localSubgraphInduction;
    int *d_inductions;
    int *d_relabeling;
    /****************************************/
} DataGPU;

typedef struct {
    /****************************************/
    /***************Graph********************/
    int *h_vertexOffset;
    int *h_adjacencyList;
    int *h_degree;
    /****************************************/

    /****************************************/
    /*************Enumeration****************/
    unsigned long* h_result;
    int *h_id;
    int *h_numberOfExtensions;
    int *h_extensions;
    int *h_jobs;
    int *h_currentJob;
    int *h_currentPosOfJob;
    int *h_validJobs;
    int *h_currentPos;
    int *h_extensionsOffset;
    bool *h_stop;
    int *h_status;
    int h_relabeling;
    /****************************************/

    /****************************************/
    /*************Constants******************/
    int h_k;
    int h_numberOfActiveThreads;
    int h_blockSize;
    int h_numberOfSMs;
    int h_jobsPerWarp;
    
    unsigned int h_warpSize;
    int h_virtualWarpSize;
    int h_numberOfBlocks;
    int h_warpsPerBlock;
    int h_numberOfWarps;
    int h_numberOfVirtualWarps;
    float h_percentageWarpsIdle;
    
    int h_maxVertexId;
    int h_maxDegree;
    int h_maxDegreeRounded;
    int h_numberOfEdges;
    int h_extensionsLength;
    int h_initialJobsPerWarp;
    int h_theoreticalJobsPerWarp;
    int h_globalThreshold;
    long unsigned h_resultCounter;

    unsigned long long *h_hashPerWarp;
    long unsigned int *h_hashGlobal;
    long unsigned int *h_localSubgraphInduction;
    int *h_inductions;
    /****************************************/
} DataCPU;

typedef struct {
    int tid;
    int wid;
    int lane;
    int k;
    int offsetWarp;
    int offsetExtensions;
    int offsetHash;
    unsigned int mask;
} GPULocalVariables;

#endif