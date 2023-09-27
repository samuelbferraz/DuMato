#include "Structs.h"

#ifndef DEVICE_H
#define DEVICE_H

typedef struct {
    // Graph
    int *d_vertexOffset;
    int *d_adjacencyList;
    int *d_degree;

    // Thread monitoring/sync
    int *d_status;
    int *d_smid;
    volatile bool *d_stop;
    volatile int *d_bufferFull;
    volatile int *d_bufferDrain;

    // Enumeration data structures
    int *d_id;
    int *d_jobs;
    int *d_inductions;
    int *d_jobsPerWarp;
    int *d_currentJob;
    int *d_currentPosOfJob;
    int *d_validJobs;
    int *d_numberOfExtensions;
    int *d_extensions;
    int *d_extensionSources;
    int *d_currentPos;
    int *d_induce;
    int *d_buffer;
    unsigned int *d_offsetBuffer;
    int *d_chunksStatus;
    long unsigned int *d_localSubgraphInduction;
    unsigned long long *d_hashPerWarp;

    long unsigned int *d_amountNewVertices;
    long unsigned int *d_removedEdges;
    long unsigned int *d_addedEdges;

    volatile int *d_globalVertexId;

    Embeddings* d_embeddings;
    unsigned long* d_result;

    // Constants
    int *d_k;
    int *d_numberOfActiveThreads;
    int *d_maxVertexId;
    int *d_maxDegree;
    int *d_extensionsLength;
    int *d_warpSize;
    int *d_extensionsOffset;
    long unsigned int *d_quickToCgLocal;
    int *d_numberOfCgs;
} Device;

#endif
