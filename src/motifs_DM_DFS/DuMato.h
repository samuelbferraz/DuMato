#include "CudaHelperFunctions.h"

namespace DuMato {
    __device__ Device* d;

    __device__ int *vertexOffset;
    __device__ int *adjacencyList;
    __device__ int *degree;

    // Thread monitoring/sync
    __device__ int *status;
    __device__ int *smid;
    __device__ volatile bool *stop;

    // Enumeration data structures
    __device__ int *id;
    __device__ int *numberOfExtensions;
    __device__ int *extensions;
    // __device__ long unsigned int *extensionsQuick;
    __device__ int *extensionSources;
    __device__ int *extensionSourcesOffset;
    __device__ int *updateCompactionCounters;
    __device__ int *currentPos;
    __device__ unsigned int *buffer;
    __device__ unsigned int *bufferCounter;
    __device__ int *chunksStatus;
    __device__ long unsigned int *localSubgraphInduction;
    __device__ int *extensionsOffset;
    __device__ unsigned long* result;
    __device__ unsigned int* quickToCgLocal;
    __device__ unsigned long long* hashPerWarp;


    /*------------------------------------------------------------------------*/

    // __shared__ int k[1024];
    __device__ int k[409600];

    __device__ int dm_getTid() {
        return (blockIdx.x * blockDim.x) + threadIdx.x;
    }

    __device__ int dm_getWid() {
        return dm_getTid() / *(d->d_virtualWarpSize);
    }

    __device__ int dm_getLocalWid() {
        return threadIdx.x / *(d->d_virtualWarpSize);
    }

    __device__ int dm_getLane() {
        return threadIdx.x & 0x1f;
    }

    __device__ int dm_globalK() {
        return *(d->d_k);
    }

    __device__ int dm_k() {
        return k[dm_getTid()];
        // return k[threadIdx.x];
    }

    __device__ void dm_k(int value) {
        k[dm_getTid()] = value;
        // k[threadIdx.x] = value;
    }

    __device__ int neighbour(int vertexId, int relativePosition) {
        return adjacencyList[vertexOffset[vertexId]+relativePosition];
    }

    __device__ int roundToWarpSize(int value) {
        return ((int)ceilf((float)value / (float)(*(d->d_virtualWarpSize)))) * (*(d->d_virtualWarpSize));
    }

    __device__ int dm_hashOffset() {
        return dm_getWid() * *(d->d_numberOfCgs);
    }

    __device__ int dm_offsetWarp() {
        return dm_getWid() * *(d->d_warpSize);
    }

    __device__ int dm_offsetExtensions() {
        return dm_getWid() * (*d->d_extensionsLength);
    }

    __device__ int dm_offsetExtensionSourcesK() {
        return dm_getWid() * dm_globalK() * dm_globalK() + dm_k() * dm_globalK();
    }

    __device__ int dm_extensionSourcesOffsetK(int k) {
        return extensionSourcesOffset[dm_offsetExtensionSourcesK()+k];
    }

    __device__ int dm_offsetCurrentInduction() {
        return dm_getWid() * dm_globalK();
    }

    __device__ void dm_popJob() {
        if(dm_k() != -1)
            return;

        int vertex;

        vertex = atomicAdd((int*)d->d_globalVertexId, 1);
        // job = 0;

        if(vertex > *(d->d_maxVertexId)) {
            vertex = -1;
            status[dm_getTid()] = 2;
            smid[dm_getTid()] = -1;
        }
        else {
            dm_k(0);
            id[dm_offsetWarp()] = vertex;
            for(int i = 0 ; i < *(d->d_warpSize) ; i++)
                numberOfExtensions[dm_offsetWarp()+i] = -1;
            status[dm_getTid()] = 1;
            smid[dm_getTid()] = __mysmid();
        }
    }

    __device__ void dm_start(Device* device) {
        d = device;

        vertexOffset = device->d_vertexOffset;
        adjacencyList = device->d_adjacencyList;
        degree = device->d_degree;

        id = device->d_id;
        result = device->d_result;
        status = device->d_status;
        smid = device->d_smid;
        currentPos = device->d_currentPos;
        numberOfExtensions = device->d_numberOfExtensions;
        extensions = device->d_extensions;
        // extensionsQuick = device->d_extensionsQuick;
        extensionsOffset = device->d_extensionsOffset;
        extensionSources = device->d_extensionSources;
        extensionSourcesOffset = device->d_extensionSourcesOffset;
        updateCompactionCounters = device->d_updateCompactionCounters;
        buffer = device->d_buffer;
        bufferCounter = device->d_bufferCounter;
        chunksStatus = device->d_chunksStatus;
        localSubgraphInduction = device->d_localSubgraphInduction;
        quickToCgLocal = device->d_quickToCgLocal;
        hashPerWarp = device->d_hashPerWarp;

        status[dm_getTid()] = 1;
        smid[dm_getTid()] = __mysmid();
        result[dm_getWid()] = 0;
        dm_k(currentPos[dm_getWid()]);
        dm_popJob();
    }

    __device__ void dm_end() {
        currentPos[dm_getWid()] = dm_k();
    }

    __device__ int dm_id(int position) {
        return id[dm_offsetWarp()+position];
    }

    __device__ bool dm_active() {
        return dm_k() >= 0;
    }

    __device__ bool dm_gpuIsBalanced() {
        return !(*(d->d_stop));
    }

    __device__ int dm_numberOfExtensions() {
        return numberOfExtensions[dm_offsetWarp()+dm_k()];
    }

    __device__ int dm_getExtension(int position) {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
        return extensions[localOffsetExtensions + position];
    }

    // __device__ void dm_setExtensionQuick(int position, long unsigned int value) {
    //     int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
    //     extensionsQuick[localOffsetExtensions + position] = value;
    // }
    //
    // __device__ long unsigned int dm_getExtensionQuick(int position) {
    //     int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
    //     return extensionsQuick[localOffsetExtensions + position];
    // }

    __device__ void dm_printExtensions() {
        if(dm_getLane() == 0) {
            printf("*\n");
            for(int i = 0 ; i <= dm_k() ; i++) {
                printf("%d ", dm_id(i));
            }
            printf("\n");

            printf("**\n");
            for(int i = 0 ; i < dm_numberOfExtensions() ; i++) {
                printf("%d ", dm_getExtension(i));
            }
            printf("\n\n");
        }
        __syncwarp();

    }

    __device__ bool dm_findExtension(int start, int end, int possibleExtension) {
        int validExtensions = numberOfExtensions[dm_offsetWarp()+dm_k()], found = 0;
        if(end > validExtensions)
            return false;

        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];

        for(int warpPosition = start+dm_getLane() ; warpPosition < roundToWarpSize(end) && !found ;  warpPosition += 32) {
            found = __any_sync(0xffffffff, warpPosition < end && extensions[localOffsetExtensions+warpPosition] == possibleExtension ? 1 : 0);
        }

        return found == 1 ? true : false;
    }

    __device__ bool dm_findExtension2(int start, int end, int possibleExtension) {
        int validExtensions = numberOfExtensions[dm_offsetWarp()+dm_k()];
        bool found = false;

        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];

        for(int threadPosition = start ; threadPosition < end && !found ;  threadPosition++)
            found = extensions[localOffsetExtensions+threadPosition] == possibleExtension;

        return found;
    }

    __device__ void dm_generateUniqueExtensions(int begin, int end) {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
        int currentOffsetExtensions = 0;

        numberOfExtensions[dm_offsetWarp()+dm_k()] = 0;

        for(int i = begin, currentVertexDegree ; i <= end ; i++) {
            currentVertexDegree = degree[dm_id(i)];

            extensionSourcesOffset[dm_offsetExtensionSourcesK()+i] = currentOffsetExtensions;

            for(int threadPosition = 0, currentNeighbour ; threadPosition < currentVertexDegree ;  threadPosition++) {
                currentNeighbour = neighbour(dm_id(i), threadPosition);
                extensions[localOffsetExtensions+currentOffsetExtensions+threadPosition] = !dm_findExtension2(0, currentOffsetExtensions, currentNeighbour) ? currentNeighbour : -1;
                extensionSources[localOffsetExtensions+currentOffsetExtensions+threadPosition] = i;
            }

            // TODO Allow this increment to be more memory-aligned (coalescence)
            currentOffsetExtensions += currentVertexDegree;
        }
        numberOfExtensions[dm_offsetWarp()+dm_k()] = currentOffsetExtensions;
    }

    __device__ void dm_generateAllExtensions(int begin, int end) {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
        int currentOffsetExtensions = 0;

        numberOfExtensions[dm_offsetWarp()+dm_k()] = 0;

        for(int i = begin, currentVertexDegree ; i <= end ; i++) {
            currentVertexDegree = degree[dm_id(i)];
            extensionSourcesOffset[dm_offsetExtensionSourcesK()+i] = currentOffsetExtensions;

            for(int threadPosition = 0 ; threadPosition < currentVertexDegree ;  threadPosition++) {
                extensions[localOffsetExtensions+currentOffsetExtensions+threadPosition] =  neighbour(dm_id(i), threadPosition);
                extensionSources[localOffsetExtensions+currentOffsetExtensions+threadPosition] = i;
            }

            // TODO Allow this increment to be more memory-aligned (coalescence)
            currentOffsetExtensions += currentVertexDegree;
        }
        numberOfExtensions[dm_offsetWarp()+dm_k()] = currentOffsetExtensions;
    }

    __device__ void dm_compactExtensions() {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
        int length = dm_numberOfExtensions();
        unsigned int localCompactionCounter = 0;

        for(int i = 0 ; i < dm_globalK() ; i++)
            updateCompactionCounters[dm_offsetCurrentInduction()+i] = 0;

        for(int threadPosition = 0, element, source ; threadPosition < length ; threadPosition++) {

            element = extensions[localOffsetExtensions+threadPosition];
            source = extensionSources[localOffsetExtensions+threadPosition];

            if(element != -1) {
                extensions[localOffsetExtensions+localCompactionCounter] = element;
                extensionSources[localOffsetExtensions+localCompactionCounter] = source;
                localCompactionCounter++;
            }
            else {
                updateCompactionCounters[dm_offsetCurrentInduction()+source]++;
            }
        }

        numberOfExtensions[dm_offsetWarp()+dm_k()] = localCompactionCounter;

        for(int i = 1 ; i <= dm_k() ; i++) {
            extensionSourcesOffset[dm_offsetExtensionSourcesK()+i] -= updateCompactionCounters[dm_offsetCurrentInduction()+i-1];
            updateCompactionCounters[dm_offsetCurrentInduction()+i] += updateCompactionCounters[dm_offsetCurrentInduction()+i-1];
        }
    }

    __device__ void dm_filterExtensionsLowerThan(int id) {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
        int validExtensions = numberOfExtensions[dm_offsetWarp()+dm_k()];
        for(int threadPosition = 0, extension ; threadPosition < validExtensions ;  threadPosition++) {
            extension = extensions[localOffsetExtensions+threadPosition];
            extensions[localOffsetExtensions+threadPosition] = extension > id ? extension : -1;
        }
    }

    __device__ void dm_filterExtensionsLowerOrEqualThan(int start, int end, int id) {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];

        // if(dm_k() == 1 && dm_id(0) == 20 && dm_id(1) == 661)
        //     printf("localOffsetExtensions: %d\n", localOffsetExtensions);

        for(int threadPosition = start, extension ; threadPosition < end ;  threadPosition++) {
            extension = extensions[localOffsetExtensions+threadPosition];
            // if(dm_k() == 1 && dm_id(0) == 20 && dm_id(1) == 661)
            //     printf("extension: %d\n", extension);
            extensions[localOffsetExtensions+threadPosition] = extension <= id ? -1 : extension;
        }
    }

    __device__ void dm_filterExtensionsEqual(int start, int end, int id) {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
        for(int threadPosition = start, extension ; threadPosition < end ;  threadPosition++) {
            extension = extensions[localOffsetExtensions+threadPosition];
            extensions[localOffsetExtensions+threadPosition] = extension == id ? -1 : extension;
        }
    }

    // Careful: don't use it in a virtual warp size != 32
    __device__ bool dm_findNeighbourhood(int vertexId, int possibleNeighbour) {
        int found = 0, currentDegree = degree[vertexId];
        for(int warpPosition = dm_getLane() ; warpPosition < roundToWarpSize(currentDegree) && !found ; warpPosition += 32) {
            found = __any_sync(0xffffffff, warpPosition < currentDegree && neighbour(vertexId, warpPosition) == possibleNeighbour ? 1 : 0);
        }
        return found == 1 ? true : false;
    }

    // Careful: works only for virtual warp size == 1
    __device__ bool dm_findNeighbourhood2(int vertexId, int possibleNeighbour) {
        int found = 0, currentDegree = degree[vertexId];
        for(int threadPosition = 0 ; threadPosition < currentDegree && !found ; threadPosition++) {
            if(neighbour(vertexId, threadPosition) == possibleNeighbour)
                 found = 1;
        }
        return found == 1 ? true : false;
    }

    __device__ void dm_invalidateExtension(int position) {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
        extensions[localOffsetExtensions+position] = -1;
    }

    __device__ int dm_firstInChunk(int position) {
        return position % CHUNK_SIZE == 0;
    }

    __device__ int dm_lastInChunk(int position) {
        return (position + 1) % CHUNK_SIZE == 0;
    }

    __device__ int dm_myChunk(int position) {
        return position / CHUNK_SIZE;
    }

    __device__ void dm_accumulateValidSubgraphs() {
        result[dm_getWid()] += dm_numberOfExtensions();
    }

    __device__ void dm_agregateValidSubgraphs() {
        int lane = dm_getLane();
        int nextEmbeddingId = dm_id(dm_k());
        int localOffsetInduction = (((dm_k()-2)*(2+dm_k()-1))/2);

        unsigned long quickPattern = 0;
        for(int i = 0, currentPow = powf(2,localOffsetInduction), found ; i < dm_k() ; i++, currentPow*=2) {
            found = dm_findNeighbourhood2(dm_id(i),nextEmbeddingId) ? 1 : 0;
            quickPattern += (found*currentPow);
        }
        localSubgraphInduction[dm_offsetCurrentInduction()+dm_k()] = localSubgraphInduction[dm_offsetCurrentInduction()+dm_k()-1] + quickPattern;

        localOffsetInduction = (((dm_k()-1)*(2+dm_k()))/2);

        // if(dm_k() == 1 && dm_id(0) == 20 && dm_id(1) == 661) {
        //     printf("*** %d ***\n", dm_numberOfExtensions());
        //     for(int i = 0 ; i < dm_numberOfExtensions() ; i++) {
        //         printf("** %d **\n", dm_getExtension(i));
        //     }
        // }

        for(int threadPosition = 0, nextEmbeddingId ; threadPosition < dm_numberOfExtensions() ; threadPosition++) {

            nextEmbeddingId = dm_getExtension(threadPosition);

            // Final induction (quick pattern)
            quickPattern = 0;
            for(int j = 0, currentPow = powf(2,localOffsetInduction), found ; j <= dm_k() ; j++, currentPow*=2) {
                found = dm_findNeighbourhood2(dm_id(j),nextEmbeddingId) ? 1 : 0;
                quickPattern += (found*currentPow);
            }

            // for(int i = 0 ; i <= dm_k() ; i++) {
            //     printf("[%d]", dm_id(i));
            // }
            // printf("[%d]\n", nextEmbeddingId);

            quickPattern += localSubgraphInduction[dm_offsetCurrentInduction()+dm_k()];
            hashPerWarp[dm_hashOffset()+quickToCgLocal[quickPattern]] += 1;
        }
    }

    __device__ void dm_forward() {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
        int nextEmbeddingId = extensions[localOffsetExtensions+dm_numberOfExtensions()-1];
        // long unsigned int nextEmbeddingQuick = extensionsQuick[localOffsetExtensions+dm_numberOfExtensions()-1];
        numberOfExtensions[dm_offsetWarp()+dm_k()]--;
        dm_k(dm_k()+1);
        id[dm_offsetWarp()+dm_k()] = nextEmbeddingId;
        numberOfExtensions[dm_offsetWarp()+dm_k()] = -1;

        // Induce current subgraph
        if(dm_k() >= 2) {
            // (((dm_k()-2)*(2+dm_k()-1))/2) -> Offset created by previous inductions (sum of PA starting in 2)
            int localOffsetInduction = (((dm_k()-2)*(2+dm_k()-1))/2);

            unsigned long quickPattern = 0;
            for(int i = 0, currentPow = powf(2,localOffsetInduction), found ; i < dm_k() ; i++, currentPow*=2) {
                found = dm_findNeighbourhood2(dm_id(i),nextEmbeddingId) ? 1 : 0;
                quickPattern += (found*currentPow);
            }
            localSubgraphInduction[dm_offsetCurrentInduction()+dm_k()] = localSubgraphInduction[dm_offsetCurrentInduction()+dm_k()-1] + quickPattern;
        }
    }

    __device__ void dm_backward() {
        dm_k(dm_k()-1);
        if(!dm_active())
            dm_popJob();
    }
}
