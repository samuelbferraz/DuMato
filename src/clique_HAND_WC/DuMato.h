#include "CudaHelperFunctions.h"

namespace DuMato {
    __device__ Device* d;

    __device__ int *vertexOffset;
    __device__ int *adjacencyList;
    __device__ int *degree;

    // Thread monitoring/sync
    __device__ int *status;
    __device__ int *smid;
    
    // Enumeration data structures
    __device__ int *id;
    __device__ int *numberOfExtensions;
    __device__ int *extensions;
    __device__ int *neighbours;
    __device__ int *neighbourSources;
    
    __device__ int *extensionSources;
    __device__ int *currentPos;
    __device__ int induce;
    __device__ int *buffer;
    __device__ unsigned int *offsetBuffer;
    __device__ int *chunksStatus;
    __device__ long unsigned int *localSubgraphInduction;
    __device__ int *extensionsOffset;
    __device__ unsigned long* result;
    __device__ long unsigned int* quickToCgLocal;
    __device__ unsigned long long* hashPerWarp;

    __device__ long unsigned int* amountNewVertices;
    __device__ long unsigned int* removedEdges;
    __device__ long unsigned int* addedEdges;

    __device__ int jobsPerWarp;
    __device__ int *jobs;
    __device__ int *inductions;
    __device__ int *currentJob;
    __device__ int *currentPosOfJob;
    __device__ int *validJobs;
    __device__ int maxCliqueEdges;

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

    __device__ int dm_offsetWarp() {
        return dm_getWid() * *(d->d_warpSize);
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

    __device__ int dm_offsetExtensions() {
        return dm_getWid() * (*d->d_extensionsLength);
    }

    __device__ int dm_offsetExtensionSourcesK() {
        return dm_getWid() * dm_globalK() * dm_globalK() + dm_k() * dm_globalK();
    }

    __device__ int dm_offsetCurrentInduction() {
        return dm_getWid() * 32;
    }

    __device__ unsigned int dm_offsetBuffer() {
        return dm_getWid() * GPU_BUFFER_SIZE_PER_WARP;
    }

    __device__ unsigned int dm_localOffsetBuffer() {
        return dm_offsetBuffer() + offsetBuffer[dm_getWid()];
    }


    __device__ int dm_getCurrentJob() {
        return currentJob[dm_getWid()];
    }

    __device__ void dm_setCurrentJob(int job) {
        currentJob[dm_getWid()] = job;
    }

    __device__ int dm_getValidJobs() {
        return validJobs[dm_getWid()];
    }

    __device__ void dm_setValidJobs(int job) {
        validJobs[dm_getWid()] = job;
    }

    __device__ void dm_setInduction(int k, int value) {
        inductions[dm_getWid()*jobsPerWarp*32 + dm_getCurrentJob()*32 + k] = value;
    }

    __device__ int dm_getInduction(int k) {
        return inductions[dm_getWid()*jobsPerWarp*32 + dm_getCurrentJob()*32 + k];
    }

    __device__ int dm_getJob(int k) {
        return jobs[dm_getWid()*jobsPerWarp*32 + dm_getCurrentJob()*32 + k];
    }

    __device__ int dm_getCurrentPosOfJob() {
        return currentPosOfJob[dm_getWid()*jobsPerWarp+dm_getCurrentJob()];
    }

    __device__ void dm_increaseJob() {
        currentJob[dm_getWid()]++;
    }

    __device__ void dm_setNumberOfExtensions(int value) {
        numberOfExtensions[dm_offsetWarp()+dm_k()] = value;
    }

    __device__ void dm_initializeNumberOfExtensions(int value) {
        numberOfExtensions[dm_offsetWarp()+dm_getLane()] = value;
    }

    __device__ int dm_getExtension(int position) {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
        return extensions[localOffsetExtensions + position];
    }

    __device__ void dm_setExtension(int position, int value) {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
        extensions[localOffsetExtensions + position] = value;
    }

    __device__ void dm_popJob() {
        if(dm_k() != -1)
            return;

        if(dm_getCurrentJob() >= dm_getValidJobs()) {
            status[dm_getTid()] = 2;
            smid[dm_getTid()] = -1;
        }
        else {
            int k = dm_getCurrentPosOfJob();
            dm_k(k);
            id[dm_offsetWarp()+dm_getLane()] = dm_getJob(dm_getLane());
            localSubgraphInduction[dm_offsetCurrentInduction()+dm_getLane()] = dm_getInduction(dm_getLane());
            dm_initializeNumberOfExtensions(0);
            dm_setNumberOfExtensions(-1);

            status[dm_getTid()] = 1;
            smid[dm_getTid()] = __mysmid();

            dm_increaseJob();
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
        buffer = device->d_buffer;
        offsetBuffer = device->d_offsetBuffer;
        chunksStatus = device->d_chunksStatus;
        localSubgraphInduction = device->d_localSubgraphInduction;
        quickToCgLocal = device->d_quickToCgLocal;
        hashPerWarp = device->d_hashPerWarp;
        jobsPerWarp = *(device->d_jobsPerWarp);
        jobs = device->d_jobs;
        inductions = device->d_inductions;
        currentPosOfJob = device->d_currentPosOfJob;
        validJobs = device->d_validJobs;
        currentJob = device->d_currentJob;
        induce = *(device->d_induce);
        amountNewVertices = device->d_amountNewVertices;
        removedEdges = device->d_removedEdges;
        addedEdges = device->d_addedEdges;

        *(d->d_bufferFull) = 0;
        *(d->d_bufferDrain) = 0;
        maxCliqueEdges = (dm_globalK() * (dm_globalK()-1))/2;
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

    __device__ bool dm_bufferIsHealthy() {
        return !(*(d->d_bufferFull)) && !(*(d->d_bufferDrain));
    }

    __device__ int dm_numberOfExtensions() {
        return numberOfExtensions[dm_offsetWarp()+dm_k()];
    }

    __device__ int dm_getSource(int position) {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
        return extensionSources[localOffsetExtensions + position];
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

    __device__ void dm_generateExtensionsSingleSource(int source) {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
        int currentOffsetExtensions = 0;
        int v0 = dm_id(source);
        int currentVertexDegree = degree[dm_id(source)];

        unsigned long pos, actives, totalActives, activesOnMyRight, idlesOnMyRight;
        
        for(int warpPosition = dm_getLane(), currentNeighbour ; warpPosition < roundToWarpSize(currentVertexDegree) ;  warpPosition += 32) {
            currentNeighbour = neighbour(dm_id(source), warpPosition);
            currentNeighbour = warpPosition < currentVertexDegree && currentNeighbour > v0 ? currentNeighbour : -1;
            __syncwarp();

            actives = __ballot_sync(0xffffffff, currentNeighbour == -1 ? 0 : 1);
            totalActives = __popc(actives);
            actives = (actives << ((unsigned int)32-(unsigned int)dm_getLane()));
            activesOnMyRight = __popc(actives);
            idlesOnMyRight = dm_getLane() - activesOnMyRight;

            pos = currentNeighbour != -1 ? activesOnMyRight : totalActives + idlesOnMyRight;
            extensions[localOffsetExtensions+currentOffsetExtensions+pos] = currentNeighbour;
            currentOffsetExtensions += totalActives;
        }
        numberOfExtensions[dm_offsetWarp()+dm_k()] = currentOffsetExtensions;
    }

    __device__ void dm_generateUniqueExtensions(int begin, int end) {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
        int currentOffsetExtensions = 0;
        int v0 = dm_id(begin);
        unsigned long pos, actives, totalActives, activesOnMyRight, idlesOnMyRight;

        numberOfExtensions[dm_offsetWarp()+dm_k()] = 0;

        for(int i = begin, currentVertexDegree ; i <= end ; i++) {
            currentVertexDegree = degree[dm_id(i)];

            for(int warpPosition = dm_getLane(), currentNeighbour ; warpPosition < roundToWarpSize(currentVertexDegree) ;  warpPosition += 32) {
                currentNeighbour = neighbour(dm_id(i), warpPosition);
                currentNeighbour = warpPosition < currentVertexDegree && currentNeighbour > v0 && !dm_findExtension2(0, currentOffsetExtensions, currentNeighbour) ? currentNeighbour : -1;
                __syncwarp();
                
                actives = __ballot_sync(0xffffffff, currentNeighbour == -1 ? 0 : 1);
                totalActives = __popc(actives);
                actives = (actives << ((unsigned int)32-(unsigned int)dm_getLane()));
                activesOnMyRight = __popc(actives);
                idlesOnMyRight = dm_getLane() - activesOnMyRight;

                pos = currentNeighbour != -1 ? activesOnMyRight : totalActives + idlesOnMyRight;
                extensions[localOffsetExtensions+currentOffsetExtensions+pos] = currentNeighbour;
                extensionSources[localOffsetExtensions+currentOffsetExtensions+pos] = i;
                currentOffsetExtensions += totalActives;
            }
        }
        numberOfExtensions[dm_offsetWarp()+dm_k()] = currentOffsetExtensions;
    }

    __device__ void dm_canonicalFilter() {
        unsigned long pos, actives, totalActives, activesOnMyRight, idlesOnMyRight;

        // Canonical filtering
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()], currentOffsetExtensions;

        for(int i = 1, target ; i <= dm_k() ; i++) {
            target = dm_id(i);
            currentOffsetExtensions = 0;
            for(int warpPosition = dm_getLane(), ext, src ; warpPosition < roundToWarpSize(dm_numberOfExtensions()) ; warpPosition += 32) {
                ext = warpPosition < dm_numberOfExtensions() ? dm_getExtension(warpPosition) : -1;
                src = warpPosition < dm_numberOfExtensions() ? dm_getSource(warpPosition) : -1;
                
                ext = (i > src && ext <= target) || (i < src && ext == target) ? -1 : ext; 

                actives = __ballot_sync(0xffffffff, ext == -1 ? 0 : 1);
                totalActives = __popc(actives);
                actives = (actives << ((unsigned int)32-(unsigned int)dm_getLane()));
                activesOnMyRight = __popc(actives);
                idlesOnMyRight = dm_getLane() - activesOnMyRight;

                pos = ext != -1 ? activesOnMyRight : totalActives + idlesOnMyRight;
                extensions[localOffsetExtensions+currentOffsetExtensions+pos] = ext;
                extensionSources[localOffsetExtensions+currentOffsetExtensions+pos] = src;
                currentOffsetExtensions += totalActives;
            }
            numberOfExtensions[dm_offsetWarp()+dm_k()] = currentOffsetExtensions;
        }
    }


    __device__ void dm_generateAllExtensions(int begin, int end) {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
        int currentOffsetExtensions = 0;

        numberOfExtensions[dm_offsetWarp()+dm_k()] = 0;

        for(int i = begin, currentVertexDegree ; i <= end ; i++) {
            currentVertexDegree = degree[dm_id(i)];
            
            for(int warpPosition = dm_getLane() ; warpPosition < roundToWarpSize(currentVertexDegree) ;  warpPosition += 32) {
                extensions[localOffsetExtensions+currentOffsetExtensions+warpPosition] = warpPosition < currentVertexDegree ? neighbour(dm_id(i), warpPosition) : -1;
                extensionSources[localOffsetExtensions+currentOffsetExtensions+warpPosition] = warpPosition < currentVertexDegree ? i : -1;
            }

            // TODO Allow this increment to be more memory-aligned (coalescence)
            currentOffsetExtensions += currentVertexDegree;
        }
        numberOfExtensions[dm_offsetWarp()+dm_k()] = currentOffsetExtensions;
    }

    __device__ void dm_filterExtensionsLowerThan(int id) {
        int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()];
        int validExtensions = numberOfExtensions[dm_offsetWarp()+dm_k()];
        for(int warpPosition = dm_getLane(), extension ; warpPosition < roundToWarpSize(validExtensions) ;  warpPosition += 32) {
            extension = extensions[localOffsetExtensions+warpPosition];
            extensions[localOffsetExtensions+warpPosition] = warpPosition < validExtensions && extension < id ? extension : -1;
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

    __device__ void dm_accumulateValidSubgraphs() {
        result[dm_getWid()] += dm_numberOfExtensions();
    }

    __device__ void dm_agregateValidSubgraphs() {
        if(!induce) {
            printf("Error! You can not aggregate without the induce option...\n");
            return;
        }

        int lane = dm_getLane();
        int nextEmbeddingId = dm_id(dm_k());
        int localOffsetInduction = (((dm_k()-2)*(2+dm_k()-1))/2);

        unsigned long quickPattern = 0;
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
        }
    }

    __device__ void dm_calculateCompression() {
        int numberOfExtensions = dm_numberOfExtensions();

        if(offsetBuffer[dm_getWid()] + numberOfExtensions < GPU_BUFFER_SIZE_PER_WARP) {
            for(int i = dm_getLane() ; i <= dm_k() ; i += 32)
                buffer[dm_localOffsetBuffer() + i] = dm_id(i);
            __syncwarp();

            buffer[dm_localOffsetBuffer()+dm_k()+1] = numberOfExtensions;
            offsetBuffer[dm_getWid()] += (dm_k()+2);

            for(int i = dm_getLane() ; i < numberOfExtensions ; i += 32)
                buffer[dm_localOffsetBuffer() + i] = dm_getExtension(i);
            __syncwarp();
            offsetBuffer[dm_getWid()] += numberOfExtensions;

            removedEdges[dm_getWid()] = offsetBuffer[dm_getWid()];
        }
        else {
            // Error! Flag to indicate whether a warp ran out of memory for buffering.
            *(d->d_bufferFull) = 1;
        }

        if(offsetBuffer[dm_getWid()] + numberOfExtensions > 0.7*GPU_BUFFER_SIZE_PER_WARP) {
            *(d->d_bufferDrain) = 1;
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

        if(induce) {
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
    }

    __device__ void dm_backward() {
        dm_k(dm_k()-1);
        if(!dm_active())
            dm_popJob();
    }
}
