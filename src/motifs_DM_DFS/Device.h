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
    int *d_extensions;
    // long unsigned int *d_extensionsQuick;
    int *d_extensionSources;
    int *d_extensionSourcesOffset;
    int *d_updateCompactionCounters;
    int *d_currentPos;
    unsigned int *d_buffer;
    unsigned int *d_bufferCounter;
    int *d_chunksStatus;
    long unsigned int *d_localSubgraphInduction;
    unsigned long long *d_hashPerWarp;

    volatile int *d_globalVertexId;

    Embeddings* d_embeddings;
    Extensions* d_ext;
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
    unsigned int *d_quickToCgLocal;
    int *d_numberOfCgs;
} Device;

#endif
