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


    // Constants
    int *d_maxVertexId;

    /*------------------------------------------------------------------------*/

    // __shared__ int k[1024];
    __device__ int k[12800];

    __device__ int dm_getTid() {
        return (blockIdx.x * blockDim.x) + threadIdx.x;
    }

    __device__ int dm_getWid() {
        return dm_getTid() / 32;
    }

    __device__ int dm_getLocalWid() {
        return threadIdx.x / 32;
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
        return ((int)ceilf((float)value / (float)32)) * 32;
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

        if(dm_getLane() == 0) {
            vertex = atomicAdd((int*)d->d_globalVertexId, 1);
        }
        vertex = __shfl_sync(0xffffffff, vertex, 0);
        // job = 0;

        if(vertex > *(d->d_maxVertexId)) {
            vertex = -1;
            status[dm_getTid()] = 2;
            smid[dm_getTid()] = -1;
        }
        else {
            dm_k(0);
            id[dm_offsetWarp()] = vertex;
            numberOfExtensions[dm_offsetWarp()+dm_getLane()] = -1;
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

        for(int warpPosition = start ; warpPosition < end && !found ;  warpPosition ++)
            found = extensions[localOffsetExtensions+warpPosition] == possibleExtension;

        return found;
    }

    __device__ void dm_extend(int begin, int end) {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
        int currentOffsetExtensions = 0;

        numberOfExtensions[dm_offsetWarp()+dm_k()] = 0;

        for(int i = begin, currentVertexDegree ; i <= end ; i++) {
            currentVertexDegree = degree[dm_id(i)];

            extensionSourcesOffset[dm_offsetExtensionSourcesK()+i] = currentOffsetExtensions;

            for(int warpPosition = dm_getLane(), currentNeighbour ; warpPosition < roundToWarpSize(currentVertexDegree) ;  warpPosition += 32) {
                currentNeighbour = neighbour(dm_id(i), warpPosition);
                extensions[localOffsetExtensions+currentOffsetExtensions+warpPosition] = warpPosition < currentVertexDegree && !dm_findExtension2(0, currentOffsetExtensions, currentNeighbour) ? currentNeighbour : -1;
                __syncwarp();
                extensionSources[localOffsetExtensions+currentOffsetExtensions+warpPosition] = warpPosition < currentVertexDegree ? i : -1;
            }

            // TODO Allow this increment to be more memory-aligned (coalescence)
            currentOffsetExtensions += currentVertexDegree;
        }
        numberOfExtensions[dm_offsetWarp()+dm_k()] = currentOffsetExtensions;
    }

    __device__ void dm_extend_all(int begin, int end) {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
        int currentOffsetExtensions = 0;

        numberOfExtensions[dm_offsetWarp()+dm_k()] = 0;

        for(int i = begin, currentVertexDegree ; i <= end ; i++) {
            currentVertexDegree = degree[dm_id(i)];
            extensionSourcesOffset[dm_offsetExtensionSourcesK()+i] = currentOffsetExtensions;

            for(int warpPosition = dm_getLane() ; warpPosition < roundToWarpSize(currentVertexDegree) ;  warpPosition += 32) {
                extensions[localOffsetExtensions+currentOffsetExtensions+warpPosition] = warpPosition < currentVertexDegree ? neighbour(dm_id(i), warpPosition) : -1;
                extensionSources[localOffsetExtensions+currentOffsetExtensions+warpPosition] = warpPosition < currentVertexDegree ? i : -1;
            }

            currentOffsetExtensions += currentVertexDegree;
        }
        numberOfExtensions[dm_offsetWarp()+dm_k()] = currentOffsetExtensions;
    }

    __device__ void dm_compact() {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
        int length = dm_numberOfExtensions();
        unsigned int amountSoFar = 0;

        for(int warpPosition = dm_getLane(), element, source ; warpPosition < roundToWarpSize(length) ; warpPosition += 32) {
            if(warpPosition < dm_globalK())
                updateCompactionCounters[dm_getWid()*dm_globalK()+warpPosition] = 0;
            __syncwarp();

            unsigned int actives, myActives, activesOnMyRight;

            element = warpPosition >= length ? -1 : extensions[localOffsetExtensions+warpPosition];
            source = warpPosition >= length ? -1 : extensionSources[localOffsetExtensions+warpPosition];

            actives = __ballot_sync(0xffffffff, element == -1 ? 0 : 1);
            myActives = actives;
            myActives = (myActives << ((unsigned int)32-(unsigned int)dm_getLane()));
            activesOnMyRight = __popc(myActives);

            if(element != -1) {
                extensions[localOffsetExtensions+amountSoFar+activesOnMyRight] = element;
                extensionSources[localOffsetExtensions+amountSoFar+activesOnMyRight] = source;
            }
            else if(warpPosition < length) {
                atomicAdd_block(updateCompactionCounters + dm_getWid()*dm_globalK() + source, 1);
            }

            __syncwarp();

            amountSoFar += __popc(actives);
        }

        if(dm_getLane() == 0) {
            for(int i = 1 ; i <= dm_k() ; i++) {
                extensionSourcesOffset[dm_offsetExtensionSourcesK()+i]-=updateCompactionCounters[dm_getWid()*dm_globalK()+i-1];
                updateCompactionCounters[dm_getWid()*dm_globalK()+i]+=updateCompactionCounters[dm_getWid()*dm_globalK()+i-1];
            }
        }
        __syncwarp();

        numberOfExtensions[dm_offsetWarp()+dm_k()] = amountSoFar;
    }

    __device__ void dm_filterExtensionsLowerThan(int id) {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
        int validExtensions = numberOfExtensions[dm_offsetWarp()+dm_k()];
        for(int warpPosition = dm_getLane(), extension ; warpPosition < roundToWarpSize(validExtensions) ;  warpPosition += 32) {
            extension = extensions[localOffsetExtensions+warpPosition];
            extensions[localOffsetExtensions+warpPosition] = warpPosition < validExtensions && extension > id ? extension : -1;
        }
    }

    __device__ void dm_filterExtensionsLowerOrEqualThan(int start, int end, int id) {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
        for(int warpPosition = start+dm_getLane(), extension ; warpPosition < roundToWarpSize(end) ;  warpPosition += 32) {
            extension = extensions[localOffsetExtensions+warpPosition];
            extensions[localOffsetExtensions+warpPosition] = warpPosition < end && extension <= id ? -1 : extension;
        }
    }

    __device__ void dm_filterExtensionsEqual(int start, int end, int id) {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
        for(int warpPosition = start+dm_getLane(), extension ; warpPosition < roundToWarpSize(end) ;  warpPosition += 32) {
            extension = extensions[localOffsetExtensions+warpPosition];
            extensions[localOffsetExtensions+warpPosition] = warpPosition < end && extension == id ? -1 : extension;
        }
    }

    __device__ bool dm_findNeighbourhood(int vertexId, int possibleNeighbour) {
        int found = 0, currentDegree = degree[vertexId];
        for(int warpPosition = dm_getLane() ; warpPosition < roundToWarpSize(currentDegree) && !found ; warpPosition += 32) {
            found = __any_sync(0xffffffff, warpPosition < currentDegree && neighbour(vertexId, warpPosition) == possibleNeighbour ? 1 : 0);
        }
        return found == 1 ? true : false;
    }

    __device__ bool dm_findNeighbourhood2(int vertexId, int possibleNeighbour) {
        int found = 0, currentDegree = degree[vertexId];
        for(int warpPosition = 0 ; warpPosition < currentDegree && !found ; warpPosition++) {
            if(neighbour(vertexId, warpPosition) == possibleNeighbour)
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

    __device__ void dm_aggregate_count() {
        result[dm_getWid()] += dm_numberOfExtensions();
    }

    __device__ void dm_agregate_pattern() {
        int lane = dm_getLane();
        int nextEmbeddingId = dm_id(dm_k());
        int localOffsetInduction = (((dm_k()-2)*(2+dm_k()-1))/2);

        unsigned long quickPattern = 0;
        for(int i = 0, currentPow = powf(2,localOffsetInduction), found ; i < dm_k() ; i++, currentPow*=2) {
            found = dm_findNeighbourhood(dm_id(i),nextEmbeddingId) ? 1 : 0;
            quickPattern += (found*currentPow);
        }
        localSubgraphInduction[dm_offsetCurrentInduction()+dm_k()] = localSubgraphInduction[dm_offsetCurrentInduction()+dm_k()-1] + quickPattern;

        localOffsetInduction = (((dm_k()-1)*(2+dm_k()))/2);
        for(int warpPosition = dm_getLane(), nextEmbeddingId ; warpPosition < roundToWarpSize(dm_numberOfExtensions()) ; warpPosition+=32) {
            if(warpPosition < dm_numberOfExtensions()) {
                nextEmbeddingId = dm_getExtension(warpPosition);

                // Final induction (quick pattern)
                quickPattern = 0;
                for(int j = 0, currentPow = powf(2,localOffsetInduction), found ; j <= dm_k() ; j++, currentPow*=2) {
                    found = dm_findNeighbourhood2(dm_id(j),nextEmbeddingId) ? 1 : 0;
                    quickPattern += (found*currentPow);
                }
            }
            __syncwarp();
            if(warpPosition < dm_numberOfExtensions()) {
                quickPattern += localSubgraphInduction[dm_offsetCurrentInduction()+dm_k()];
                atomicAdd_block(&(hashPerWarp[dm_hashOffset()+quickToCgLocal[quickPattern]]), 1);
            }
            __syncwarp();


            // if(dm_getLane() == 0) {
            //     for(int i = 0 ; i <= dm_k() ; i++) {
            //         printf("%d,", dm_id(i));
            //     }
            //     printf("%d", nextEmbeddingId);
            //     // printf("%u ", localSubgraphInduction[dm_offsetCurrentInduction()+dm_k()+1]);
            //     printf("\n");
            // }
            // __syncwarp();

            // unsigned int position = lane > 0 ? 0 : atomicInc(bufferCounter, GPU_BUFFER_SIZE-1);
            // position = __shfl_sync(0xffffffff, position, 0);
            //
            // for(int i = dm_getLane() ; i <= dm_k()+1 ; i += 32) {
            //     buffer[position * dm_globalK() + i] = i < dm_k() + 1 ? dm_id(i) : nextEmbeddingId;
            // }

            // if(chunksStatus[dm_myChunk(position)] != CHUNK_SIZE) {
            //     buffer[position] = quickPattern;
            //
            //     if(dm_getLane() == 0)
            //         atomicAdd(chunksStatus + dm_myChunk(position), 1);
            //
            //     __syncwarp();
            // }
            // else {
            //     if(dm_getLane() == 0)
            //         printf("buffer is full! Subgraphs are not being buffered... %lu;%lu;%d\n", position, dm_myChunk(position), chunksStatus[dm_myChunk(position)]);
            //     __syncwarp();
            // }
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
                found = dm_findNeighbourhood(dm_id(i),nextEmbeddingId) ? 1 : 0;
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
