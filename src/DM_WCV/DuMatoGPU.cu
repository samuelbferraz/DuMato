#include <cuda_runtime.h>
#include "Structs.cu"

#ifndef __DUMATO_GPU_CU__
#define __DUMATO_GPU_CU__

class DuMatoGPU {
    public:
        DataGPU *dataGPU;
        GPULocalVariables variables;
        
        __device__ void start(DataGPU *dataGPU) {
            this->dataGPU = dataGPU;
            variables.tid = (blockIdx.x * blockDim.x) + threadIdx.x;
            variables.wid = variables.tid / *dataGPU->d_warpSize; 
            // variables.lane = threadIdx.x & 0x1f;
            // unsigned int warpSize = *dataGPU->d_warpSize;
            // unsigned int hardwareLane = threadIdx.x & 0x1f;
            // unsigned int subwarpId = (threadIdx.x & 0x1f) / (*dataGPU->d_warpSize);
            unsigned int shifts = ((threadIdx.x & 0x1f) / (*dataGPU->d_warpSize)) * *dataGPU->d_warpSize;
            unsigned int mask = pow(2, *dataGPU->d_warpSize)-1;
            variables.lane = threadIdx.x & (*dataGPU->d_warpSize-1);
            variables.mask = mask << shifts;
            // variables.mask = (threadIdx.x & 0x1f) < *dataGPU->d_warpSize ? 0x0000ffff : 0xffff0000;

            // variables.offsetWarp = variables.wid * 32;
            // variables.offsetWarp = variables.wid * *dataGPU->d_warpSize;
            variables.offsetWarp = variables.wid * *dataGPU->d_warpSize;
            variables.offsetExtensions = variables.wid * *(dataGPU->d_extensionsLength);
            variables.offsetHash = variables.wid * *(dataGPU->d_numberOfCgs);
            
            dataGPU->d_result[variables.wid] = 0;
            dataGPU->d_status[variables.wid] = 1;
            variables.k = dataGPU->d_currentPos[variables.wid];

            popJob();
        }

        // The only difference between start and start_induce is the popJob_induce function. 
        // A flag indicating the need of induction would allow merging both functions.
        __device__ void start_induce(DataGPU *dataGPU) {
            this->dataGPU = dataGPU;
            variables.tid = (blockIdx.x * blockDim.x) + threadIdx.x;
            variables.wid = variables.tid / *dataGPU->d_warpSize; 
            // variables.lane = threadIdx.x & 0x1f;
            // unsigned int warpSize = *dataGPU->d_warpSize;
            // unsigned int hardwareLane = threadIdx.x & 0x1f;
            // unsigned int subwarpId = (threadIdx.x & 0x1f) / (*dataGPU->d_warpSize);
            unsigned int shifts = ((threadIdx.x & 0x1f) / (*dataGPU->d_warpSize)) * *dataGPU->d_warpSize;
            unsigned int mask = pow(2, *dataGPU->d_warpSize)-1;
            variables.lane = threadIdx.x & (*dataGPU->d_warpSize-1);
            variables.mask = mask << shifts;
            // variables.mask = (threadIdx.x & 0x1f) < *dataGPU->d_warpSize ? 0x0000ffff : 0xffff0000;

            // variables.offsetWarp = variables.wid * 32;
            // variables.offsetWarp = variables.wid * *dataGPU->d_warpSize;
            variables.offsetWarp = variables.wid * *dataGPU->d_warpSize;
            variables.offsetExtensions = variables.wid * *(dataGPU->d_extensionsLength);
            variables.offsetHash = variables.wid * *(dataGPU->d_numberOfCgs);
            
            dataGPU->d_result[variables.wid] = 0;
            dataGPU->d_status[variables.wid] = 1;
            variables.k = dataGPU->d_currentPos[variables.wid];

            popJob_induce();
        }

        __device__ void end() {
            dataGPU->d_currentPos[variables.wid] = variables.k;
        }

        __device__ int roundToWarpSize(int value) {
            // return ((int)ceilf((float)value / (float)32)) * 32;
            // return ((int)ceilf((float)value / (float)*dataGPU->d_warpSize)) * *dataGPU->d_warpSize;
            return ((int)ceilf((float)value / (float)*dataGPU->d_warpSize)) * *dataGPU->d_warpSize;
        }

        __device__ int neighbour(int vertexId, int relativePosition) {
            return dataGPU->d_adjacencyList[dataGPU->d_vertexOffset[vertexId]+relativePosition];
        }

        __device__ int getCurrentJob() {
            return dataGPU->d_currentJob[variables.wid];
        }

        __device__ int getValidJobs() {
            return dataGPU->d_validJobs[variables.wid];
        }

        __device__ int getCurrentPosOfJob() {
            return dataGPU->d_currentPosOfJob[variables.wid*(*dataGPU->d_jobsPerWarp)+getCurrentJob()];
        }

        __device__ int getJob() {
            // return dataGPU->d_jobs[variables.wid*(*dataGPU->d_jobsPerWarp)*(32) + getCurrentJob()*(32) + variables.lane];
            // return dataGPU->d_jobs[variables.wid*(*dataGPU->d_jobsPerWarp)*(*dataGPU->d_warpSize) + getCurrentJob()*(*dataGPU->d_warpSize) + variables.lane];
            return dataGPU->d_jobs[variables.wid*(*dataGPU->d_jobsPerWarp)*(*dataGPU->d_warpSize) + getCurrentJob()*(*dataGPU->d_warpSize) + variables.lane];
        }

        __device__ int getInduction() {
            // return dataGPU->d_inductions[variables.wid*(*dataGPU->d_jobsPerWarp)*(32) + getCurrentJob()*(32) + variables.lane];
            // return dataGPU->d_inductions[variables.wid*(*dataGPU->d_jobsPerWarp)*(*dataGPU->d_warpSize) + getCurrentJob()*(*dataGPU->d_warpSize) + variables.lane];
            return dataGPU->d_inductions[variables.wid*(*dataGPU->d_jobsPerWarp)*(*dataGPU->d_warpSize) + getCurrentJob()*(*dataGPU->d_warpSize) + variables.lane];
        }

        __device__ int findMany(int *v, int size, int value) {
            int foundLocal = 0;
            for(int lockstep = 0 ; lockstep < size && !foundLocal ; lockstep++) {
                if(v[lockstep] == value)
                    foundLocal = 1;
            }
            return foundLocal;
        }

        __device__ int findOne(int *v, int size, int value) {
            int found = 0;
            // for(int warpPosition = variables.lane ; warpPosition < roundToWarpSize(size) && !found ; warpPosition += 32) {
            // for(int warpPosition = variables.lane ; warpPosition < roundToWarpSize(size) && !found ; warpPosition += *dataGPU->d_warpSize) {
            for(int warpPosition = variables.lane ; warpPosition < roundToWarpSize(size) && !found ; warpPosition += (*dataGPU->d_warpSize)) {
                // found = __any_sync(0xffffffff, warpPosition < size && v[warpPosition] == value ? 1 : 0);
                found = __any_sync(variables.mask, warpPosition < size && v[warpPosition] == value ? 1 : 0);
            }
            return found;
        }

        __device__ int write(int *v, int start, int value, bool valid) {
            unsigned long pos, actives, totalActives, activesOnMyRight, idlesOnMyRight;
                
            // actives = __ballot_sync(0xffffffff, !valid ? 0 : 1);
            actives = __ballot_sync(variables.mask, !valid ? 0 : 1);
            totalActives = __popc(actives);
            // actives = (actives << ((unsigned int)(32)-(unsigned int)variables.lane));
            // actives = (actives << ((unsigned int)(*dataGPU->d_warpSize)-(unsigned int)variables.lane));
            actives = (actives << ((unsigned int)(*dataGPU->d_warpSize)-(unsigned int)variables.lane));
            actives = actives & variables.mask;
            activesOnMyRight = __popc(actives);
            idlesOnMyRight = variables.lane - activesOnMyRight;
            pos = valid ? activesOnMyRight : totalActives + idlesOnMyRight;
            *(v+start+pos) = value;
            
            return totalActives;
        }

        __device__ int write_twice(int *v1, int start, int value1, int *v2, int value2, bool valid) {
            unsigned long pos, actives, totalActives, activesOnMyRight, idlesOnMyRight;
                
            // actives = __ballot_sync(0xffffffff, !valid ? 0 : 1);
            actives = __ballot_sync(variables.mask, !valid ? 0 : 1);
            totalActives = __popc(actives);
            // actives = (actives << ((unsigned int)(32)-(unsigned int)variables.lane));
            // actives = (actives << ((unsigned int)(*dataGPU->d_warpSize)-(unsigned int)variables.lane));
            actives = (actives << ((unsigned int)(*dataGPU->d_warpSize)-(unsigned int)variables.lane));
            actives = actives & variables.mask;
            activesOnMyRight = __popc(actives);
            idlesOnMyRight = variables.lane - activesOnMyRight;
            pos = valid ? activesOnMyRight : totalActives + idlesOnMyRight;
            *(v1+start+pos) = value1;
            *(v2+start+pos) = value2;
            
            return totalActives;
        }

        __device__ int getId(int targetK) {
            return dataGPU->d_id[variables.offsetWarp+targetK];
        }

        __device__ int* getAdjacency(int id) {
            return dataGPU->d_adjacencyList + dataGPU->d_vertexOffset[id];
        }

        __device__ int getExtension(int relativePosition) {
            int localOffsetExtensions = variables.offsetExtensions + dataGPU->d_extensionsOffset[variables.k];
            return dataGPU->d_extensions[localOffsetExtensions+relativePosition];
        }

        __device__ int getSource(int relativePosition) {
            int localOffsetExtensions = variables.offsetExtensions + dataGPU->d_extensionsOffset[variables.k];
            return dataGPU->d_extensionSources[localOffsetExtensions+relativePosition];
        }

        __device__ void setExtension(int relativePosition, int value) {
            int localOffsetExtensions = variables.offsetExtensions + dataGPU->d_extensionsOffset[variables.k];
            dataGPU->d_extensions[localOffsetExtensions+relativePosition] = value;
        }

        __device__ int* getExtensionsArray() {
            int localOffsetExtensions = variables.offsetExtensions + dataGPU->d_extensionsOffset[variables.k];
            return dataGPU->d_extensions + localOffsetExtensions;
        }

        __device__ int* getExtensionSourcesArray() {
            int localOffsetExtensions = variables.offsetExtensions + dataGPU->d_extensionsOffset[variables.k];
            return dataGPU->d_extensionSources + localOffsetExtensions;
        }

        __device__ long unsigned int getLocalSubgraphInduction(int k) {
            return dataGPU->d_localSubgraphInduction[variables.offsetWarp+k];
        }

        __device__ void setLocalSubgraphInduction(int k, long unsigned int value) {
            dataGPU->d_localSubgraphInduction[variables.offsetWarp+k] = value;
        }

        __device__ int degree(int id) {
            return dataGPU->d_degree[id];
        }

        __device__ void popJob() {
            if(variables.k != -1)
                return;
            
            if(getCurrentJob() >= getValidJobs()) {
                dataGPU->d_status[variables.wid] = 2;
            }
            else {
                variables.k = getCurrentPosOfJob();
                dataGPU->d_id[variables.offsetWarp+variables.lane] = getJob();
                dataGPU->d_numberOfExtensions[variables.offsetWarp+variables.lane] = 0; 
                dataGPU->d_numberOfExtensions[variables.offsetWarp+variables.k] = -1;
                dataGPU->d_currentJob[variables.wid]++;
                dataGPU->d_status[variables.wid] = 1;
            }
        }

        // The differences between popJob and popJob_induce are the manipulation of
        // subgraph inductions. The if/else structure is quite different.
        // Be careful when merging it to standard popJob.
        __device__ void popJob_induce() {
            if(variables.k != -1)
                return;
            
            if(getCurrentJob() >= getValidJobs()) {
                dataGPU->d_status[variables.wid] = 2;
            }
            else {
                variables.k = getCurrentPosOfJob();
                if(variables.k == 0) {
                    dataGPU->d_id[variables.offsetWarp+variables.lane] = getJob();
                    dataGPU->d_numberOfExtensions[variables.offsetWarp+variables.lane] = 0; 
                    dataGPU->d_numberOfExtensions[variables.offsetWarp+variables.k] = -1;
                    dataGPU->d_currentJob[variables.wid]++;
                    dataGPU->d_status[variables.wid] = 1;
                }
                else {
                    dataGPU->d_id[variables.offsetWarp+variables.lane] = getJob();
                    setLocalSubgraphInduction(variables.lane, getInduction());
                    dataGPU->d_numberOfExtensions[variables.offsetWarp+variables.lane] = 0;
                    dataGPU->d_numberOfExtensions[variables.offsetWarp+variables.k-1] = 1;
                    variables.k--;
                    setExtension(0, getId(variables.k+1));
                    dataGPU->d_currentJob[variables.wid]++;
                    dataGPU->d_status[variables.wid] = 1;
                }
            }
        }

        __device__ void backward() {
            variables.k = variables.k - 1;
            if(variables.k < 0) {
                popJob();
            } 
        }

        // The only difference between backward and backward_induce is the popJob_induce function. 
        // A flag indicating the need of induction would allow merging both functions.
        __device__ void backward_induce() {
            variables.k = variables.k - 1;
            if(variables.k < 0) {
                popJob_induce();
            } 
        }

        __device__ void forward() {
            int numberOfExtensions = dataGPU->d_numberOfExtensions[variables.offsetWarp+variables.k];
            int localOffsetExtensions = variables.offsetExtensions + dataGPU->d_extensionsOffset[variables.k];
            int nextEmbeddingID = dataGPU->d_extensions[localOffsetExtensions+numberOfExtensions-1];
            dataGPU->d_numberOfExtensions[variables.offsetWarp+variables.k]--;
            variables.k = variables.k + 1;
            dataGPU->d_id[variables.offsetWarp+variables.k] = nextEmbeddingID;
            dataGPU->d_numberOfExtensions[variables.offsetWarp+variables.k] = -1;
        }

        // The only difference between forward and forward_induce is the conditional (variables.k >= 2). 
        // A flag indicating the need of induction would allow merging both functions.
        __device__ void forward_induce() {
            // Original forward + induction
            int numberOfExtensions = getCurrentNumberOfExtensions();
            int nextEmbeddingId = getExtension(numberOfExtensions-1);
            int source = getSource(numberOfExtensions-1);
            setCurrentNumberOfExtensions(numberOfExtensions-1);
            variables.k = variables.k + 1;
            dataGPU->d_id[variables.offsetWarp+variables.k] = nextEmbeddingId;
            dataGPU->d_numberOfExtensions[variables.offsetWarp+variables.k] = -1;

            if(variables.k >= 2) {
                // (((dm_k()-2)*(2+dm_k()-1))/2) -> Offset created by previous inductions (sum of PA starting in 2)
                int localOffsetInduction = (((variables.k-2)*(2+variables.k-1))/2);

                unsigned long quickPattern = 0;
                for(int i = 0, currentPow = powf(2,localOffsetInduction), found, id, currentDegree ; i < variables.k ; i++, currentPow*=2) {
                    id = getId(i);
                    currentDegree = degree(id);
                    found = findOne(getAdjacency(id), currentDegree, nextEmbeddingId);
                    quickPattern += (found*currentPow);
                }
                dataGPU->d_localSubgraphInduction[variables.offsetWarp+variables.k] = dataGPU->d_localSubgraphInduction[variables.offsetWarp+variables.k-1] + quickPattern;
            }
        }

        __device__ void aggregate_pattern() {
            unsigned long quickPattern = 0;
            int localOffsetInduction = (((variables.k-1)*(2+variables.k))/2);
            int numberOfExtensions = getCurrentNumberOfExtensions();

            // for(int warpPosition = variables.lane, nextEmbeddingId ; warpPosition < roundToWarpSize(numberOfExtensions) ; warpPosition+=32) {
            // for(int warpPosition = variables.lane, nextEmbeddingId ; warpPosition < roundToWarpSize(numberOfExtensions) ; warpPosition+=*dataGPU->d_warpSize) {
            for(int warpPosition = variables.lane, nextEmbeddingId ; warpPosition < roundToWarpSize(numberOfExtensions) ; warpPosition+=*dataGPU->d_warpSize) {
                if(warpPosition < numberOfExtensions) {
                    nextEmbeddingId = getExtension(warpPosition);

                    // Final induction (quick pattern)
                    quickPattern = 0;
                    for(int j = 0, currentPow = powf(2,localOffsetInduction), found, id, currentDegree ; j <= variables.k ; j++, currentPow*=2) {
                        id = getId(j);
                        currentDegree = degree(id);
                        found = findMany(getAdjacency(id), currentDegree, nextEmbeddingId);
                        quickPattern += (found*currentPow);
                    }
                }
                // __syncwarp();
                __syncwarp(variables.mask);
                if(warpPosition < numberOfExtensions) {
                    quickPattern += dataGPU->d_localSubgraphInduction[variables.offsetWarp+variables.k];
                    // if(dataGPU->d_quickToCgLocal[quickPattern] == 0) {
                    //     printf("k = %d, ids: [%d][%d][%d][%d][%d], localSubgraphInduction: [%lu][%lu][%lu][%lu][%lu], quickPattern: %lu\n", variables.k, getId(0), getId(1), getId(2), getId(3), nextEmbeddingId, getLocalSubgraphInduction(0), getLocalSubgraphInduction(1), getLocalSubgraphInduction(2), getLocalSubgraphInduction(3), getLocalSubgraphInduction(4), quickPattern);
                    // }
                    atomicAdd_block(&(dataGPU->d_hashPerWarp[variables.offsetHash+dataGPU->d_quickToCgLocal[quickPattern]]), 1);
                }
                // __syncwarp();
                __syncwarp(variables.mask);
            }
        }

        __device__ void canonicalFilter() {
            // Canonical filtering
            int currentOffsetExtensions, numberOfExtensions = getCurrentNumberOfExtensions();
            bool validExtension;

            for(int i = 1, target ; i <= variables.k ; i++) {    
                target = getId(i);
                currentOffsetExtensions = 0;
                // for(int warpPosition = variables.lane, ext, src ; warpPosition < roundToWarpSize(numberOfExtensions) ; warpPosition += 32) {
                // for(int warpPosition = variables.lane, ext, src ; warpPosition < roundToWarpSize(numberOfExtensions) ; warpPosition += *dataGPU->d_warpSize) {
                for(int warpPosition = variables.lane, ext, src ; warpPosition < roundToWarpSize(numberOfExtensions) ; warpPosition += *dataGPU->d_warpSize) {
                    ext = warpPosition < numberOfExtensions ? getExtension(warpPosition) : -1;
                    src = warpPosition < numberOfExtensions ? getSource(warpPosition) : -1;                    
                    ext = (i > src && ext <= target) || (i < src && ext == target) ? -1 : ext; 
                    validExtension = ext != -1 ? true : false;
                    currentOffsetExtensions += write_twice(getExtensionsArray(), currentOffsetExtensions, ext, getExtensionSourcesArray(), src, validExtension);
                }
                numberOfExtensions = currentOffsetExtensions;
            }
            setCurrentNumberOfExtensions(numberOfExtensions);
        }

        __device__ void canonicalFilter2() {
            // Canonical filtering
            int currentOffsetExtensions, numberOfExtensions = getCurrentNumberOfExtensions();
            bool validExtension;

            for(int i = 1, target ; i <= variables.k ; i++) {    
                target = getId(i);
                // for(int warpPosition = variables.lane, ext, src ; warpPosition < roundToWarpSize(numberOfExtensions) ; warpPosition += 32) {
                // for(int warpPosition = variables.lane, ext, src ; warpPosition < roundToWarpSize(numberOfExtensions) ; warpPosition += *dataGPU->d_warpSize) {
                for(int warpPosition = variables.lane, ext, src ; warpPosition < roundToWarpSize(numberOfExtensions) ; warpPosition += *dataGPU->d_warpSize) {    
                    ext = warpPosition < numberOfExtensions ? getExtension(warpPosition) : -1;
                    src = warpPosition < numberOfExtensions ? getSource(warpPosition) : -1;                    
                    ext = (i > src && ext <= target) || (i < src && ext == target) ? -1 : ext; 
                    setExtension(warpPosition, ext);
                }
            }

            currentOffsetExtensions = 0;
            // for(int warpPosition = variables.lane, ext, src ; warpPosition < roundToWarpSize(numberOfExtensions) ; warpPosition += 32) {
            // for(int warpPosition = variables.lane, ext, src ; warpPosition < roundToWarpSize(numberOfExtensions) ; warpPosition += *dataGPU->d_warpSize) {
            for(int warpPosition = variables.lane, ext, src ; warpPosition < roundToWarpSize(numberOfExtensions) ; warpPosition += *dataGPU->d_warpSize) {    
                ext = warpPosition < numberOfExtensions ? getExtension(warpPosition) : -1;
                src = warpPosition < numberOfExtensions ? getSource(warpPosition) : -1;                    
                validExtension = ext != -1 ? true : false;
                currentOffsetExtensions += write_twice(getExtensionsArray(), currentOffsetExtensions, ext, getExtensionSourcesArray(), src, validExtension);
            }
            setCurrentNumberOfExtensions(currentOffsetExtensions);
        }

        __device__ void extend() {
            bool validExtension = false;
            int currentExtensionsSize = 0, v0 = getId(variables.k);
            int currentVertexDegree = degree(v0);
            
            // for(int warpPosition = variables.lane, currentNeighbour ; warpPosition < roundToWarpSize(currentVertexDegree) ; warpPosition += 32) {
            // for(int warpPosition = variables.lane, currentNeighbour ; warpPosition < roundToWarpSize(currentVertexDegree) ; warpPosition += *dataGPU->d_warpSize) {
            for(int warpPosition = variables.lane, currentNeighbour ; warpPosition < roundToWarpSize(currentVertexDegree) ; warpPosition += *dataGPU->d_warpSize) {
                currentNeighbour = neighbour(v0, warpPosition);
                currentNeighbour = warpPosition < currentVertexDegree && currentNeighbour > v0 ? currentNeighbour : -1;
                validExtension = currentNeighbour != -1 ? true : false;
                currentExtensionsSize += write(getExtensionsArray(), currentExtensionsSize, currentNeighbour, validExtension);
            }
            setCurrentNumberOfExtensions(currentExtensionsSize);
        }

        __device__ void extend(int begin, int end) {
            int v0 = getId(begin);
            int currentOffsetExtensions = 0;
            bool validExtension;
            
            for(int i = begin, currentVertexDegree, id ; i <= end ; i++) {
                id = getId(i);
                currentVertexDegree = degree(id);

                // for(int warpPosition = variables.lane, currentNeighbour, found ; warpPosition < roundToWarpSize(currentVertexDegree) ;  warpPosition += 32) {
                // for(int warpPosition = variables.lane, currentNeighbour, found ; warpPosition < roundToWarpSize(currentVertexDegree) ;  warpPosition += *dataGPU->d_warpSize) {
                for(int warpPosition = variables.lane, currentNeighbour, found ; warpPosition < roundToWarpSize(currentVertexDegree) ;  warpPosition += *dataGPU->d_warpSize) {
                    currentNeighbour = warpPosition < currentVertexDegree ? neighbour(id, warpPosition) : -1;
                    found = findMany(getExtensionsArray(), currentOffsetExtensions, currentNeighbour);
                    // __syncwarp();
                    __syncwarp(variables.mask);
                    currentNeighbour = warpPosition < currentVertexDegree && currentNeighbour > v0 && !found ? currentNeighbour : -1;
                    validExtension = currentNeighbour != -1 ? true : false;
                    currentOffsetExtensions += write_twice(getExtensionsArray(), currentOffsetExtensions, currentNeighbour, getExtensionSourcesArray(), i, validExtension);
                }
            }
            setCurrentNumberOfExtensions(currentOffsetExtensions);
        }

        __device__ int getCurrentNumberOfExtensions() {
            return dataGPU->d_numberOfExtensions[variables.offsetWarp+variables.k];
        }

        __device__ void setCurrentNumberOfExtensions(int value) {
            dataGPU->d_numberOfExtensions[variables.offsetWarp+variables.k] = value;
        }

        __device__ void filterClique() {
            int numberOfExtensions = getCurrentNumberOfExtensions(), currentExtensionsSize = 0;
            bool validExtension;

            // for(int warpPosition = variables.lane, ext; warpPosition < roundToWarpSize(numberOfExtensions) ; warpPosition+=32) {
            // for(int warpPosition = variables.lane, ext; warpPosition < roundToWarpSize(numberOfExtensions) ; warpPosition+=*dataGPU->d_warpSize) {
            for(int warpPosition = variables.lane, ext; warpPosition < roundToWarpSize(numberOfExtensions) ; warpPosition+=(*dataGPU->d_warpSize)) {
                validExtension = false;
                if(warpPosition < numberOfExtensions) {
                    validExtension = true;
                    ext = getExtension(warpPosition);

                    for(int targetK = 0, currentDegree, id ; targetK < variables.k && validExtension ; targetK++) {
                        id = getId(targetK);
                        currentDegree = degree(id);
                        validExtension = findMany(getAdjacency(id), currentDegree, ext);
                    }
                }
                // __syncwarp();
                __syncwarp(variables.mask);
                currentExtensionsSize += write(getExtensionsArray(), currentExtensionsSize, ext, validExtension);
            }
            setCurrentNumberOfExtensions(currentExtensionsSize);
        }

        __device__ void inspect() {
            __syncwarp(variables.mask);
            printf("wid: %d, k: %d, #extensions: %d, id[0]: %d, id[1]: %d, ext[%d]: %d\n", variables.wid, variables.k, getCurrentNumberOfExtensions(), getId(0), getId(1), variables.lane, getExtension(variables.lane));
        }

        __device__ void aggregate_counter() {
            dataGPU->d_result[variables.wid] += getCurrentNumberOfExtensions();
        }

        __device__ bool last_level() {
            return variables.k == *dataGPU->d_k-2;
        }

        __device__ int k() {
            return variables.k;
        } 

        __device__ bool active() {
            return variables.k >= 0;
        }

        __device__ bool balanced() {
            return !(*dataGPU->d_stop);
        }
};

#endif