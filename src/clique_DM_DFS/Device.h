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

    // Enumeration data structures
    int *d_id;
    int *d_numberOfExtensions;
    int *d_traversedExtensions;
    int *d_currentOffsetNeighboursWarps;
    // int *d_extensionSources;
    int *d_warpLocalCompactionCounters;
    int *d_readAllNeighbours;
    int *d_currentPos;
    int *d_debugBuffer;
    int *d_debugBufferCounter;
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
    int *d_virtualWarpSize;
    int *d_extensionsOffset;
} Device;

#endif
