#include <cuda_runtime.h>
#include "Structs.cu"

#ifndef __DUMATO_GPU_CU__
#define __DUMATO_GPU_CU__

class DuMatoGPU {
    public:
        DataGPU *dataGPU;
        GPULocalVariables variables;

        __device__ __inline__ uint32_t __mysmid() {
            uint32_t smid;
            asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
            return smid;
        }

        
        __device__ void start_clique(DataGPU *dataGPU) {
            this->dataGPU = dataGPU;
            variables.tid = (blockIdx.x * blockDim.x) + threadIdx.x;
            variables.wid = variables.tid / 32;
            variables.lane = threadIdx.x & 0x1f;
            variables.offsetWarp = variables.wid * 32;
            variables.offsetExtensions = variables.wid * *(dataGPU->d_extensionsLength);
            variables.offsetHash = variables.wid * *(dataGPU->d_numberOfCgs);
            
            dataGPU->d_result[variables.wid] = 0;
            dataGPU->d_status[variables.wid] = 1;
            dataGPU->d_smid[variables.wid] = __mysmid();
            variables.k = dataGPU->d_currentPos[variables.wid];                                              
            popJob_clique();
        }

        __device__ void start_q4(DataGPU *dataGPU) {
            this->dataGPU = dataGPU;
            variables.tid = (blockIdx.x * blockDim.x) + threadIdx.x;
            variables.wid = variables.tid / 32;
            variables.lane = threadIdx.x & 0x1f;
            variables.offsetWarp = variables.wid * 32;
            variables.offsetExtensions = variables.wid * *(dataGPU->d_extensionsLength);
            variables.offsetHash = variables.wid * *(dataGPU->d_numberOfCgs);
            
            dataGPU->d_result[variables.wid] = 0;
            dataGPU->d_status[variables.wid] = 1;
            dataGPU->d_smid[variables.wid] = __mysmid();
            variables.k = dataGPU->d_currentPos[variables.wid];                                              
            popJob_q4();
        }

        __device__ void start_q5(DataGPU *dataGPU) {
            this->dataGPU = dataGPU;
            variables.tid = (blockIdx.x * blockDim.x) + threadIdx.x;
            variables.wid = variables.tid / 32;
            variables.lane = threadIdx.x & 0x1f;
            variables.offsetWarp = variables.wid * 32;
            variables.offsetExtensions = variables.wid * *(dataGPU->d_extensionsLength);
            variables.offsetHash = variables.wid * *(dataGPU->d_numberOfCgs);
            
            dataGPU->d_result[variables.wid] = 0;
            dataGPU->d_status[variables.wid] = 1;
            //dataGPU->d_smid[variables.wid] = __mysmid();
            variables.k = dataGPU->d_currentPos[variables.wid];                                              
            popJob_q5();
        }

        __device__ void start_q6(DataGPU *dataGPU) {
            this->dataGPU = dataGPU;
            variables.tid = (blockIdx.x * blockDim.x) + threadIdx.x;
            variables.wid = variables.tid / 32;
            variables.lane = threadIdx.x & 0x1f;
            variables.offsetWarp = variables.wid * 32;
            variables.offsetExtensions = variables.wid * *(dataGPU->d_extensionsLength);
            variables.offsetHash = variables.wid * *(dataGPU->d_numberOfCgs);
            
            dataGPU->d_result[variables.wid] = 0;
            dataGPU->d_status[variables.wid] = 1;
            //dataGPU->d_smid[variables.wid] = __mysmid();
            variables.k = dataGPU->d_currentPos[variables.wid];                                              
            popJob_q6();
        }

        __device__ void start_q7(DataGPU *dataGPU) {
            this->dataGPU = dataGPU;
            variables.tid = (blockIdx.x * blockDim.x) + threadIdx.x;
            variables.wid = variables.tid / 32;
            variables.lane = threadIdx.x & 0x1f;
            variables.offsetWarp = variables.wid * 32;
            variables.offsetExtensions = variables.wid * *(dataGPU->d_extensionsLength);
            variables.offsetHash = variables.wid * *(dataGPU->d_numberOfCgs);
            
            dataGPU->d_result[variables.wid] = 0;
            dataGPU->d_status[variables.wid] = 1;
            //dataGPU->d_smid[variables.wid] = __mysmid();
            variables.k = dataGPU->d_currentPos[variables.wid];                                              
            popJob_q7();
        }

        __device__ void start_q8(DataGPU *dataGPU) {
            this->dataGPU = dataGPU;
            variables.tid = (blockIdx.x * blockDim.x) + threadIdx.x;
            variables.wid = variables.tid / 32;
            variables.lane = threadIdx.x & 0x1f;
            variables.offsetWarp = variables.wid * 32;
            variables.offsetExtensions = variables.wid * *(dataGPU->d_extensionsLength);
            variables.offsetHash = variables.wid * *(dataGPU->d_numberOfCgs);
            
            dataGPU->d_result[variables.wid] = 0;
            dataGPU->d_status[variables.wid] = 1;
            //dataGPU->d_smid[variables.wid] = __mysmid();
            variables.k = dataGPU->d_currentPos[variables.wid];                                              
            popJob_q8();
        }

        __device__ void start_q9(DataGPU *dataGPU) {
            this->dataGPU = dataGPU;
            variables.tid = (blockIdx.x * blockDim.x) + threadIdx.x;
            variables.wid = variables.tid / 32;
            variables.lane = threadIdx.x & 0x1f;
            variables.offsetWarp = variables.wid * 32;
            variables.offsetExtensions = variables.wid * *(dataGPU->d_extensionsLength);
            variables.offsetHash = variables.wid * *(dataGPU->d_numberOfCgs);
            
            dataGPU->d_result[variables.wid] = 0;
            dataGPU->d_status[variables.wid] = 1;
            //dataGPU->d_smid[variables.wid] = __mysmid();
            variables.k = dataGPU->d_currentPos[variables.wid];                                              
            popJob_q9();
        }

        __device__ void start_q10(DataGPU *dataGPU) {
            this->dataGPU = dataGPU;
            variables.tid = (blockIdx.x * blockDim.x) + threadIdx.x;
            variables.wid = variables.tid / 32;
            variables.lane = threadIdx.x & 0x1f;
            variables.offsetWarp = variables.wid * 32;
            variables.offsetExtensions = variables.wid * *(dataGPU->d_extensionsLength);
            variables.offsetHash = variables.wid * *(dataGPU->d_numberOfCgs);
            
            dataGPU->d_result[variables.wid] = 0;
            dataGPU->d_status[variables.wid] = 1;
            //dataGPU->d_smid[variables.wid] = __mysmid();
            variables.k = dataGPU->d_currentPos[variables.wid];                                              
            popJob_q10();
        }

        __device__ void start_q11(DataGPU *dataGPU) {
            this->dataGPU = dataGPU;
            variables.tid = (blockIdx.x * blockDim.x) + threadIdx.x;
            variables.wid = variables.tid / 32;
            variables.lane = threadIdx.x & 0x1f;
            variables.offsetWarp = variables.wid * 32;
            variables.offsetExtensions = variables.wid * *(dataGPU->d_extensionsLength);
            variables.offsetHash = variables.wid * *(dataGPU->d_numberOfCgs);
            
            dataGPU->d_result[variables.wid] = 0;
            dataGPU->d_status[variables.wid] = 1;
            //dataGPU->d_smid[variables.wid] = __mysmid();
            variables.k = dataGPU->d_currentPos[variables.wid];                                              
            popJob_q11();
        }

        __device__ void start_q12(DataGPU *dataGPU) {
            this->dataGPU = dataGPU;
            variables.tid = (blockIdx.x * blockDim.x) + threadIdx.x;
            variables.wid = variables.tid / 32;
            variables.lane = threadIdx.x & 0x1f;
            variables.offsetWarp = variables.wid * 32;
            variables.offsetExtensions = variables.wid * *(dataGPU->d_extensionsLength);
            variables.offsetHash = variables.wid * *(dataGPU->d_numberOfCgs);
            
            dataGPU->d_result[variables.wid] = 0;
            dataGPU->d_status[variables.wid] = 1;
            //dataGPU->d_smid[variables.wid] = __mysmid();
            variables.k = dataGPU->d_currentPos[variables.wid];                                              
            popJob_q12();
        }

        __device__ void start_q13(DataGPU *dataGPU) {
            this->dataGPU = dataGPU;
            variables.tid = (blockIdx.x * blockDim.x) + threadIdx.x;
            variables.wid = variables.tid / 32;
            variables.lane = threadIdx.x & 0x1f;
            variables.offsetWarp = variables.wid * 32;
            variables.offsetExtensions = variables.wid * *(dataGPU->d_extensionsLength);
            variables.offsetHash = variables.wid * *(dataGPU->d_numberOfCgs);
            
            dataGPU->d_result[variables.wid] = 0;
            dataGPU->d_status[variables.wid] = 1;
            //dataGPU->d_smid[variables.wid] = __mysmid();
            variables.k = dataGPU->d_currentPos[variables.wid];                                              
            popJob_q13();
        }

        __device__ void start_dm14(DataGPU *dataGPU) {
            this->dataGPU = dataGPU;
            variables.tid = (blockIdx.x * blockDim.x) + threadIdx.x;
            variables.wid = variables.tid / 32;
            variables.lane = threadIdx.x & 0x1f;
            variables.offsetWarp = variables.wid * 32;
            variables.offsetExtensions = variables.wid * *(dataGPU->d_extensionsLength);
            variables.offsetHash = variables.wid * *(dataGPU->d_numberOfCgs);
            
            dataGPU->d_result[variables.wid] = 0;
            dataGPU->d_status[variables.wid] = 1;
            //dataGPU->d_smid[variables.wid] = __mysmid();
            variables.k = dataGPU->d_currentPos[variables.wid];                                              
            popJob_dm14();
        }

        __device__ void start_induce(DataGPU *dataGPU) {
            this->dataGPU = dataGPU;
            variables.tid = (blockIdx.x * blockDim.x) + threadIdx.x;
            variables.wid = variables.tid / 32;
            variables.lane = threadIdx.x & 0x1f;
            variables.offsetWarp = variables.wid * 32;
            variables.offsetExtensions = variables.wid * *(dataGPU->d_extensionsLength);
            variables.offsetHash = variables.wid * *(dataGPU->d_numberOfCgs);
            
            dataGPU->d_result[variables.wid] = 0;
            dataGPU->d_status[variables.wid] = 1;
            //dataGPU->d_smid[variables.wid] = __mysmid();
            variables.k = dataGPU->d_currentPos[variables.wid];                                              
            popJob_induce();
        }

        __device__ void end() {
            dataGPU->d_currentPos[variables.wid] = variables.k;
        }

        __device__ int roundToWarpSize(int value) {
            return ((int)ceilf((float)value / (float)32)) * 32;
        }

        __device__ int roundToVirtualWarpSize(int value, int vWarpSize) {
            return ((int)ceilf((float)value / (float)vWarpSize)) * vWarpSize;
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
            return dataGPU->d_jobs[variables.wid*(*dataGPU->d_jobsPerWarp)*(32) + getCurrentJob()*(32) + variables.lane];
        }

        __device__ int getInduction() {
            return dataGPU->d_inductions[variables.wid*(*dataGPU->d_jobsPerWarp)*(32) + getCurrentJob()*(32) + variables.lane];
        }

        __device__ int findMany(int *v, int size, int value) {
            int foundLocal = 0;
            for(int lockstep = 0 ; lockstep < size && !foundLocal ; lockstep++) {
                if(v[lockstep] == value)
                    foundLocal = 1;
            }
            return value == -1 ? 0 : foundLocal;
        }

        __device__ int findOne(int *v, int size, int value) {
            int found = 0;
            for(int warpPosition = variables.lane ; warpPosition < roundToWarpSize(size) && !found ; warpPosition += 32) {
                found = __any_sync(0xffffffff, warpPosition < size && v[warpPosition] == value ? 1 : 0);
            }
            return found;
        }

        __device__ int findOneBinarySearch(int *v, int size, int value) {
            int start = 0, end = size, currentSize = size, v0;
            int found = 0;
            int offset = (currentSize / 2) - 16; // offset -> amount of shifted positions from start
            
            for(int warpPosition = start + offset + variables.lane ; currentSize > 0 && !found ; ) {
                found = __any_sync(0xffffffff, warpPosition >= 0 && warpPosition < size && v[warpPosition] == value ? 1 : 0);
                if(!found) {
                    v0 = start+offset < 0 ? v[0] : v[start+offset];
                    // Left-side
                    if(value < v0) {
                        end = start + offset;
                    }
                    // Right-side
                    else {
                        start = start + offset + 32;
                    }
                    currentSize = end - start;
                    offset = (currentSize / 2) - 16;
                    warpPosition = start + offset + variables.lane;
                    
                    /* Safety measurements to keep warpPosition with a safe value. */
                    /* warpPosition = warpPosition < 0 ? 0 : warpPosition;
                    warpPosition = warpPosition >= end ? end-1 : warpPosition; */
                }
            }
            return found;
        }

        __device__ int findOneVirtual(int *v, int size, int value, int vlane, int vWarpSize, unsigned mask) {
            int found = 0;
            for(int warpPosition = vlane ; warpPosition < roundToVirtualWarpSize(size, vWarpSize) && !found ; warpPosition += vWarpSize) {
                found = __any_sync(mask, warpPosition < size && v[warpPosition] == value ? 1 : 0);
            }
            return found;
        }

        __device__ int write(int *v, int start, int value, bool valid) {
            unsigned long pos, actives, totalActives, activesOnMyRight, idlesOnMyRight;
            actives = __ballot_sync(0xffffffff, !valid ? 0 : 1);
            totalActives = __popc(actives);
            actives = (actives << ((unsigned int)(32)-(unsigned int)variables.lane));
            activesOnMyRight = __popc(actives);
            idlesOnMyRight = variables.lane - activesOnMyRight;
            pos = valid ? activesOnMyRight : totalActives + idlesOnMyRight;
            *(v+start+pos) = value;
            
            return totalActives;
        }

        __device__ int write_twice(int *v1, int start, int value1, int *v2, int value2, bool valid) {
            unsigned long pos, actives, totalActives, activesOnMyRight, idlesOnMyRight;
                
            actives = __ballot_sync(0xffffffff, !valid ? 0 : 1);
            totalActives = __popc(actives);
            actives = (actives << ((unsigned int)(32)-(unsigned int)variables.lane));
            activesOnMyRight = __popc(actives);
            idlesOnMyRight = variables.lane - activesOnMyRight;
            pos = valid ? activesOnMyRight : totalActives + idlesOnMyRight;
            *(v1+start+pos) = value1;
            *(v2+start+pos) = value2;
            
            return totalActives;
        }

        __device__ bool greater(int x, int y) {
            return x > y;
        }
         
        __device__ bool lower(int x, int y) {
            return x < y;
        }
        
        __device__ void intersect(int *dst, int* amountDst, int *src1, int amount1, int *src2, int amount2, int symmetryValue) {
            int amount1Rounded = roundToWarpSize(amount1), found, currentSize = 0;
            bool validExtension = false;
            for(int warpPosition = variables.lane, currentSrc1 ; warpPosition < roundToWarpSize(amount1) ; warpPosition += 32) {
                currentSrc1 = src1[warpPosition];
                currentSrc1 = warpPosition < amount1 && currentSrc1 > symmetryValue ? currentSrc1 : -1;
                validExtension = currentSrc1 != -1 ? true : false;
                found = findMany(src2, amount2, currentSrc1);
                __syncwarp();
                validExtension = found == 1 ? true : false;
                currentSize += write(dst, currentSize, currentSrc1, validExtension);
            }
            *amountDst = currentSize;
        }

        __device__ void intersect_unique(int *dst, int* amountDst, int *src1, int amount1, int *src2, int amount2, int symmetryValue) {
            int amount1Rounded = roundToWarpSize(amount1), found1, found2, currentSize = 0;
            bool validExtension = false;
            for(int warpPosition = variables.lane, currentSrc1 ; warpPosition < roundToWarpSize(amount1) ; warpPosition += 32) {
                currentSrc1 = src1[warpPosition];
                currentSrc1 = warpPosition < amount1 && currentSrc1 > symmetryValue ? currentSrc1 : -1;
                validExtension = currentSrc1 != -1 ? true : false;
                found1 = findMany(src2, amount2, currentSrc1);
                __syncwarp();
                found2 = findMany(getIdArray(), variables.k+1, currentSrc1);
                __syncwarp();
                validExtension = found1 == 1 && found2 == 0 ? true : false;
                currentSize += write(dst, currentSize, currentSrc1, validExtension);
            }
            *amountDst = currentSize;
        }

        __device__ void intersect_unique(int *dst, int* amountDst, int *src1, int amount1, int *src2, int amount2, int *src3, int amount3, int symmetryValue) {
            int amount1Rounded = roundToWarpSize(amount1), found1, found2, found3, currentSize = 0;
            bool validExtension = false;
            for(int warpPosition = variables.lane, currentSrc1 ; warpPosition < roundToWarpSize(amount1) ; warpPosition += 32) {
                currentSrc1 = src1[warpPosition];
                currentSrc1 = warpPosition < amount1 && currentSrc1 > symmetryValue ? currentSrc1 : -1;
                validExtension = currentSrc1 != -1 ? true : false;
                found1 = findMany(src2, amount2, currentSrc1);
                found2 = found1 ? findMany(src3, amount3, currentSrc1) : 0;
                found3 = found1 && found2 ? findMany(getIdArray(), variables.k+1, currentSrc1) : 1;
                __syncwarp();
                validExtension = found1 == 1 && found2 == 1 && found3 == 0 ? true : false;
                currentSize += write(dst, currentSize, currentSrc1, validExtension);
            }
            *amountDst = currentSize;
        }

        __device__ void intersect(int *dst, int* amountDst, int *src1, int amount1, int *src2, int amount2, int symmetryValue, int unequalityValue) {
            int amount1Rounded = roundToWarpSize(amount1), found, currentSize = 0;
            bool validExtension = false;
            for(int warpPosition = variables.lane, currentSrc1 ; warpPosition < roundToWarpSize(amount1) ; warpPosition += 32) {
                currentSrc1 = src1[warpPosition];
                currentSrc1 = warpPosition < amount1 && currentSrc1 > symmetryValue && currentSrc1 != unequalityValue ? currentSrc1 : -1;
                validExtension = currentSrc1 != -1 ? true : false;
                found = findMany(src2, amount2, currentSrc1);
                __syncwarp();
                validExtension = found == 1 ? true : false;
                currentSize += write(dst, currentSize, currentSrc1, validExtension);
            }
            *amountDst = currentSize;
        }

        __device__ void intersect_difference(int *dst, int* amountDst, int *src1, int amount1, int *src2, int amount2, int* src3, int amount3, int symmetryValue, int unequalityValue) {
            int v0 = getId(0), v1 = getId(1), v2 = getId(2);

            int amount1Rounded = roundToWarpSize(amount1), found1, found2, currentSize = 0;
            bool validExtension = false;
            for(int warpPosition = variables.lane, currentSrc1 ; warpPosition < roundToWarpSize(amount1) ; warpPosition += 32) {
                currentSrc1 = src1[warpPosition];
                currentSrc1 = warpPosition < amount1 && currentSrc1 > symmetryValue && currentSrc1 != unequalityValue ? currentSrc1 : -1;
                validExtension = currentSrc1 != -1 ? true : false;
                found1 = findMany(src2, amount2, currentSrc1);
                __syncwarp();
                found2 = findMany(src3, amount3, currentSrc1);
                __syncwarp();
                validExtension = found1 == 1 && found2 == 0 ? true : false;
                currentSize += write(dst, currentSize, currentSrc1, validExtension);
            }
            *amountDst = currentSize;
        }

        __device__ void testIntersect() {
            intersect(getExtensionsArray(0), getCurrentNumberOfExtensionsPointer(), getAdjacency(0), degree(0), getAdjacency(2), degree(2), -1);
            
            // if(variables.lane == 0) {
            //     for(int i = 0 ; i < getCurrentNumberOfExtensions() ; i++)
            //         printf("[%d] ", getExtension(i));
            //     printf("\n");
            // }
            // __syncwarp();
        }

        __device__ int getId(int targetK) {
            return dataGPU->d_id[variables.offsetWarp+targetK];
        }
        __device__ int* getIdArray() {
            return dataGPU->d_id + variables.offsetWarp;
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

        __device__ int* getExtensionsArray(int k) {
            int localOffsetExtensions = variables.offsetExtensions + dataGPU->d_extensionsOffset[k];
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

        __device__ void popJob_clique() {
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
                prepare_job_clique();
            }
        }

        __device__ void popJob_q4() {
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
                prepare_job_q4();
            }
        }

        __device__ void popJob_q5() {
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
                prepare_job_q5();
            }
        }

        __device__ void popJob_q6() {
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
                prepare_job_q6();
            }
        }

        __device__ void popJob_q7() {
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
                prepare_job_q7();
            }
        }

        __device__ void popJob_q8() {
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
                prepare_job_q8();
            }
        }

        __device__ void popJob_q9() {
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
                prepare_job_q9();
            }
        }

        __device__ void popJob_q10() {
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
                prepare_job_q10();
            }
        }

        __device__ void popJob_q11() {
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
                prepare_job_q11();
            }
        }

        __device__ void popJob_q12() {
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
                prepare_job_q12();
            }
        }

        __device__ void popJob_q13() {
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
                prepare_job_q13();
            }
        }

        __device__ void popJob_dm14() {
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
                prepare_job_dm14();
            }
        }

        __device__ void backward_clique() {
            variables.k = variables.k - 1;
            if(variables.k < 0) {
                popJob_clique();
            } 
        }

        __device__ void backward_q4() {
            variables.k = variables.k - 1;
            if(variables.k < 0) {
                popJob_q4();
            } 
        }

        __device__ void backward_q5() {
            variables.k = variables.k - 1;
            if(variables.k < 0) {
                popJob_q5();
            } 
        }

        __device__ void backward_q6() {
            variables.k = variables.k - 1;
            if(variables.k < 0) {
                popJob_q6();
            } 
        }

        __device__ void backward_q7() {
            variables.k = variables.k - 1;
            if(variables.k < 0) {
                popJob_q7();
            } 
        }

        __device__ void backward_q8() {
            variables.k = variables.k - 1;
            if(variables.k < 0) {
                popJob_q8();
            } 
        }

        __device__ void backward_q9() {
            variables.k = variables.k - 1;
            if(variables.k < 0) {
                popJob_q9();
            } 
        }

        __device__ void backward_q10() {
            variables.k = variables.k - 1;
            if(variables.k < 0) {
                popJob_q10();
            } 
        }

        __device__ void backward_q11() {
            variables.k = variables.k - 1;
            if(variables.k < 0) {
                popJob_q11();
            } 
        }

        __device__ void backward_q12() {
            variables.k = variables.k - 1;
            if(variables.k < 0) {
                popJob_q12();
            } 
        }

        __device__ void backward_q13() {
            variables.k = variables.k - 1;
            if(variables.k < 0) {
                popJob_q13();
            } 
        }

        __device__ void backward_dm14() {
            variables.k = variables.k - 1;
            if(variables.k < 0) {
                popJob_dm14();
            } 
        }

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

        __device__ void forward_induce_virtual() {
            // Original forward + induction
            int numberOfExtensions = getCurrentNumberOfExtensions();
            int nextEmbeddingId = getExtension(numberOfExtensions-1);
            int source = getSource(numberOfExtensions-1);
            setCurrentNumberOfExtensions(numberOfExtensions-1);
            variables.k = variables.k + 1;
            dataGPU->d_id[variables.offsetWarp+variables.k] = nextEmbeddingId;
            dataGPU->d_numberOfExtensions[variables.offsetWarp+variables.k] = -1;

            if(variables.k >= 2) {
                unsigned virtualWarpSize = 16;
                unsigned amountVirtualWarps = 32 / virtualWarpSize;
                unsigned virtualWarpId = variables.lane / virtualWarpSize;
                unsigned virtualLane = variables.lane % virtualWarpSize;
                unsigned mask[2] = {0x0000ffff, 0xffff0000}; // criar máscara por warp virtual
                // printf("vwid -> %u, vlane -> %u, mask -> %u, amount -> %u\n", virtualWarpId, virtualLane, mask[virtualWarpId], amountVirtualWarps);
                
                unsigned localOffsetInduction = (((variables.k-2)*(2+variables.k-1))/2);
                unsigned long quickPattern = 0;
                for(int i = virtualWarpId, currentPow = powf(2,localOffsetInduction)*powf(2,virtualWarpId), found, id, currentDegree ; i < variables.k ; i+=amountVirtualWarps, currentPow*=powf(2,amountVirtualWarps)) {
                    id = getId(i);
                    currentDegree = degree(id);
                    found = findOneVirtual(getAdjacency(id), currentDegree, nextEmbeddingId, virtualLane, virtualWarpSize, mask[virtualWarpId]);
                    quickPattern += (found*currentPow);
                    // printf("wid -> %u, lane -> %u, vwid -> %u, vlane -> %u, mask -> %u, id -> %u, found -> %u, quickPattern -> %lu\n", variables.wid, variables.lane, virtualWarpId, virtualLane, mask[virtualWarpId], id, found, quickPattern);
                }
                __syncwarp();
                for(int i = 1 ; i < amountVirtualWarps ; i++)
                    quickPattern += __shfl_sync(0xffffffff, quickPattern, i*virtualWarpSize);
                quickPattern = __shfl_sync(0xffffffff, quickPattern, 0);
                dataGPU->d_localSubgraphInduction[variables.offsetWarp+variables.k] = dataGPU->d_localSubgraphInduction[variables.offsetWarp+variables.k-1] + quickPattern;

                // if(variables.lane == 0) {
                //     for(int k = 0 ; k <= variables.k ; k++) {
                //         printf("[%d]", getId(k));
                //     }
                //     printf("\n");
                //     for(int k = 0 ; k <= variables.k ; k++) {
                //         printf("[%lu]", dataGPU->d_localSubgraphInduction[variables.offsetWarp+k]);
                //     }
                //     printf("\n");
                // }
                // __syncwarp();
            }
        }
	
        __device__ void debug_subgraphs() {
            if(variables.lane == 0) {
                for(int ext = 0 ; ext < getCurrentNumberOfExtensions() ; ext++) {
                    for(int i = 0 ; i <= variables.k ; i++) {
                        printf("[%d]", getId(i));
                    }
                    printf("[%d]\n", getExtension(ext));
                }
            }
            __syncwarp();
        }

        __device__ void aggregate_pattern() {
            unsigned long quickPattern = 0;
            int localOffsetInduction = (((variables.k-1)*(2+variables.k))/2);
            int numberOfExtensions = getCurrentNumberOfExtensions();

            // if(variables.lane == 0) {
            //     // Print
            //     for(int i = 0 ; i <= variables.k ; i++) {
            //         printf("[%d]", getId(i));
            //     }
            //     printf(";");
            // }
            // __syncwarp();


            for(int warpPosition = variables.lane, nextEmbeddingId ; warpPosition < roundToWarpSize(numberOfExtensions) ; warpPosition+=32) {
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
                __syncwarp();
                if(warpPosition < numberOfExtensions) {
                    quickPattern += dataGPU->d_localSubgraphInduction[variables.offsetWarp+variables.k];
                    atomicAdd_block(&(dataGPU->d_hashPerWarp[variables.offsetHash+dataGPU->d_quickToCgLocal[quickPattern]]), 1);
                }
                __syncwarp();
                // if(variables.lane == 0)
                //     printf("\n");
            }
        }

        __device__ void canonicalFilter() {
            // Canonical filtering
            int currentOffsetExtensions, numberOfExtensions = getCurrentNumberOfExtensions();
            bool validExtension;

            for(int i = 1, target ; i <= variables.k ; i++) {    
                target = getId(i);
                currentOffsetExtensions = 0;
                for(int warpPosition = variables.lane, ext, src ; warpPosition < roundToWarpSize(numberOfExtensions) ; warpPosition += 32) {
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
                for(int warpPosition = variables.lane, ext, src ; warpPosition < roundToWarpSize(numberOfExtensions) ; warpPosition += 32) {
                    ext = warpPosition < numberOfExtensions ? getExtension(warpPosition) : -1;
                    src = warpPosition < numberOfExtensions ? getSource(warpPosition) : -1;                    
                    ext = (i > src && ext <= target) || (i < src && ext == target) ? -1 : ext; 
                    setExtension(warpPosition, ext);
                }
            }

            currentOffsetExtensions = 0;
            for(int warpPosition = variables.lane, ext, src ; warpPosition < roundToWarpSize(numberOfExtensions) ; warpPosition += 32) {
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
            
            for(int warpPosition = variables.lane, currentNeighbour ; warpPosition < roundToWarpSize(currentVertexDegree) ; warpPosition += 32) {
                currentNeighbour = neighbour(v0, warpPosition);
                currentNeighbour = warpPosition < currentVertexDegree && currentNeighbour > v0 ? currentNeighbour : -1;
                validExtension = currentNeighbour != -1 ? true : false;
                currentExtensionsSize += write(getExtensionsArray(), currentExtensionsSize, currentNeighbour, validExtension);
            }
            setCurrentNumberOfExtensions(currentExtensionsSize);
        }

        __device__ void extend_single(int v, int symmetryValue) {
            bool validExtension = false;
            int currentExtensionsSize = 0;
            int currentVertexDegree = degree(v);
            
            for(int warpPosition = variables.lane, currentNeighbour ; warpPosition < roundToWarpSize(currentVertexDegree) ; warpPosition += 32) {
                currentNeighbour = neighbour(v, warpPosition);
                currentNeighbour = warpPosition < currentVertexDegree && currentNeighbour > symmetryValue ? currentNeighbour : -1;
                validExtension = currentNeighbour != -1 ? true : false;
                currentExtensionsSize += write(getExtensionsArray(), currentExtensionsSize, currentNeighbour, validExtension);
            }
            setCurrentNumberOfExtensions(currentExtensionsSize);
            setCurrentNumberOfExtensionsFixed(currentExtensionsSize);
        }

        __device__ void copy_extensions(int pos, int symmetryValue) {
            bool validExtension = false;
            int currentExtensionsSize = 0;
            int previousExtensionsAmount = getCurrentNumberOfExtensionsFixed(pos);
            int *previousExtensions = getExtensionsArray(pos);
            
            for(int warpPosition = variables.lane, currentExtension ; warpPosition < roundToWarpSize(previousExtensionsAmount) ; warpPosition += 32) {
                currentExtension = previousExtensions[warpPosition];
                currentExtension = warpPosition < previousExtensionsAmount && currentExtension > symmetryValue ? currentExtension : -1;
                validExtension = currentExtension != -1 ? true : false;
                currentExtensionsSize += write(getExtensionsArray(), currentExtensionsSize, currentExtension, validExtension);
            }
            setCurrentNumberOfExtensions(currentExtensionsSize);
            setCurrentNumberOfExtensionsFixed(currentExtensionsSize);
        }

        __device__ void extend_single_unique(int v, int symmetryValue) {
            bool validExtension = false;
            int currentExtensionsSize = 0;
            int currentVertexDegree = degree(v);
            
            for(int warpPosition = variables.lane, currentNeighbour, found ; warpPosition < roundToWarpSize(currentVertexDegree) ; warpPosition += 32) {
                currentNeighbour = neighbour(v, warpPosition);
                currentNeighbour = warpPosition < currentVertexDegree && currentNeighbour > symmetryValue ? currentNeighbour : -1;
                found = findMany(getIdArray(), variables.k+1, currentNeighbour);
                __syncwarp();
                validExtension = currentNeighbour != -1 && found == 0 ? true : false;
                currentExtensionsSize += write(getExtensionsArray(), currentExtensionsSize, currentNeighbour, validExtension);
            }
            setCurrentNumberOfExtensions(currentExtensionsSize);
        }

        __device__ void prepare_job_clique() {
            int k = variables.k;
            for(int i = 0 ; i < k ; i++) {
                variables.k = i;
                extend_clique_prepare();
            }
            variables.k = k;
        }

        __device__ void prepare_job_q4() {
            int k = variables.k;
            for(int i = 0 ; i < k ; i++) {
                variables.k = i;
                extend_q4();
                setCurrentNumberOfExtensions(0);
            }
            variables.k = k;
        }

        __device__ void prepare_job_q5() {
            int k = variables.k;
            for(int i = 0 ; i < k ; i++) {
                variables.k = i;
                extend_q5();
                setCurrentNumberOfExtensions(0);
            }
            variables.k = k;
        }

        __device__ void prepare_job_q6() {
            int k = variables.k;
            for(int i = 0 ; i < k ; i++) {
                variables.k = i;
                extend_q6();
                setCurrentNumberOfExtensions(0);
            }
            variables.k = k;
        }

        __device__ void prepare_job_q7() {
            int k = variables.k;
            for(int i = 0 ; i < k ; i++) {
                variables.k = i;
                extend_q7();
                setCurrentNumberOfExtensions(0);
            }
            variables.k = k;
        }

        __device__ void prepare_job_q8() {
            int k = variables.k;
            for(int i = 0 ; i < k ; i++) {
                variables.k = i;
                extend_q8_prepare();
                setCurrentNumberOfExtensions(0);
            }
            variables.k = k;
        }

        __device__ void prepare_job_q9() {
            int k = variables.k;
            for(int i = 0 ; i < k ; i++) {
                variables.k = i;
                extend_q9();
                setCurrentNumberOfExtensions(0);
            }
            variables.k = k;
        }

        __device__ void prepare_job_q10() {
            int k = variables.k;
            for(int i = 0 ; i < k ; i++) {
                variables.k = i;
                extend_q10();
                setCurrentNumberOfExtensions(0);
            }
            variables.k = k;
        }

        __device__ void prepare_job_q11() {
            int k = variables.k;
            for(int i = 0 ; i < k ; i++) {
                variables.k = i;
                extend_q11();
                setCurrentNumberOfExtensions(0);
            }
            variables.k = k;
        }

        __device__ void prepare_job_q12() {
            int k = variables.k;
            for(int i = 0 ; i < k ; i++) {
                variables.k = i;
                extend_q12();
                setCurrentNumberOfExtensions(0);
            }
            variables.k = k;
        }

        __device__ void prepare_job_q13() {
            int k = variables.k;
            for(int i = 0 ; i < k ; i++) {
                variables.k = i;
                extend_q13();
                setCurrentNumberOfExtensions(0);
            }
            variables.k = k;
        }

        __device__ void prepare_job_dm14() {
            int k = variables.k;
            for(int i = 0 ; i < k ; i++) {
                variables.k = i;
                extend_dm14();
                setCurrentNumberOfExtensions(0);
            }
            variables.k = k;
        }

        __device__ void extend_clique() {
            bool validExtension = false;
            int currentExtensionsSize = 0, last = getId(variables.k);
            int currentVertexDegree = degree(last);
            int *previousExtensions, previousExtensionsAmount;
            if(variables.k > 0) {
                previousExtensions = getExtensionsArray(variables.k-1);
                previousExtensionsAmount = getCurrentNumberOfExtensionsFixed(variables.k-1);
            }
            
            for(int warpPosition = variables.lane, currentNeighbour ; warpPosition < roundToWarpSize(currentVertexDegree) ; warpPosition += 32) {
                currentNeighbour = neighbour(last, warpPosition);
                currentNeighbour = warpPosition < currentVertexDegree && currentNeighbour > last ? currentNeighbour : -1;
                validExtension = currentNeighbour != -1 ? true : false;
                
                if(variables.k > 0) {
                    int found = findMany(previousExtensions, previousExtensionsAmount, currentNeighbour);
                    // if(variables.k == 1 && getId(0) == 0) {
                    //     printf("previousExtensionsAmount: %d, %d,%d -> %d (degree %d), found: %d\n", previousExtensionsAmount, getId(0), getId(1), currentNeighbour, currentVertexDegree, found);
                    // }
                    validExtension = found == 1 ? true : false;
                }
                currentExtensionsSize += write(getExtensionsArray(), currentExtensionsSize, currentNeighbour, validExtension);
            }
            setCurrentNumberOfExtensions(currentExtensionsSize);
            setCurrentNumberOfExtensionsFixed(currentExtensionsSize);
        }

        __device__ void extend_q4() {
            if(variables.k == 0) {
                int v0 = getId(0);
                extend_single(v0, v0);
            }
            else if(variables.k == 1) {
                int last = getId(variables.k);
                extend_single(getId(0), -1);
                for(int i = 1 ; i <= variables.k ; i++)
                    intersect_unique(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k), getCurrentNumberOfExtensions(), getAdjacency(getId(i)), degree(getId(i)), -1);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 2) {
                int last = getId(variables.k);
                copy_extensions(variables.k-1, last);
            }
        }

        __device__ void extend_q5() {
            if(variables.k == 0) {
                int v0 = getId(0);
                extend_single(v0, v0);
            }
            else if(variables.k == 1) {
                int last = getId(variables.k);
                intersect(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k-1), getCurrentNumberOfExtensionsFixed(variables.k-1), getAdjacency(last), degree(last), last);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 2) {
                int last = getId(variables.k);
                extend_single(getId(0), -1);
                for(int i = 1 ; i <= variables.k ; i++)
                    intersect_unique(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k), getCurrentNumberOfExtensions(), getAdjacency(getId(i)), degree(getId(i)), -1);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 3) {
                int last = getId(variables.k);
                copy_extensions(variables.k-1, last);
            }
        }

        __device__ void extend_q6() {
            if(variables.k == 0) {
                int v0 = getId(0);
                extend_single(v0, v0);
            }
            else if(variables.k >= 1 && variables.k <= 2) {
                int last = getId(variables.k);
                intersect(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k-1), getCurrentNumberOfExtensionsFixed(variables.k-1), getAdjacency(last), degree(last), last);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 3) {
                int last = getId(variables.k);
                extend_single(getId(0), -1);
                for(int i = 1 ; i <= variables.k ; i++)
                    intersect_unique(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k), getCurrentNumberOfExtensions(), getAdjacency(getId(i)), degree(getId(i)), -1);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 4) {
                int last = getId(variables.k);
                copy_extensions(variables.k-1, last);
            }
        }

        __device__ void extend_q7() {
            if(variables.k == 0) {
                int v0 = getId(0);
                extend_single(v0, v0);
            }
            else if(variables.k >= 1 && variables.k <= 3) {
                int last = getId(variables.k);
                intersect(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k-1), getCurrentNumberOfExtensionsFixed(variables.k-1), getAdjacency(last), degree(last), last);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 4) {
                int last = getId(variables.k);
                extend_single(getId(0), -1);
                for(int i = 1 ; i <= variables.k ; i++)
                    intersect_unique(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k), getCurrentNumberOfExtensions(), getAdjacency(getId(i)), degree(getId(i)), -1);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 5) {
                int last = getId(variables.k);
                copy_extensions(variables.k-1, last);
            }
        }

        __device__ void extend_q8() {
            if(variables.k == 0) {
                int v0 = getId(0);
                extend_single(v0, v0);
            }
            else if(variables.k >= 1 && variables.k <= 4) {
                int last = getId(variables.k);
                intersect(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k-1), getCurrentNumberOfExtensionsFixed(variables.k-1), getAdjacency(last), degree(last), last);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 5) {
                int last = getId(variables.k);
                extend_single(getId(0), -1);
                for(int i = 1 ; i <= variables.k ; i++)
                    intersect_unique(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k), getCurrentNumberOfExtensions(), getAdjacency(getId(i)), degree(getId(i)), -1);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 6) {
                int last = getId(variables.k);
                copy_extensions(variables.k-1, last);
            }
        }

        __device__ void extend_q9() {
            if(variables.k == 0) {
                int v0 = getId(0);
                extend_single(v0, v0);
            }
            else if(variables.k >= 1 && variables.k <= 5) {
                int last = getId(variables.k);
                intersect(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k-1), getCurrentNumberOfExtensionsFixed(variables.k-1), getAdjacency(last), degree(last), last);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 6) {
                int last = getId(variables.k);
                extend_single(getId(0), -1);
                for(int i = 1 ; i <= variables.k ; i++)
                    intersect_unique(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k), getCurrentNumberOfExtensions(), getAdjacency(getId(i)), degree(getId(i)), -1);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 7) {
                int last = getId(variables.k);
                copy_extensions(variables.k-1, last);
            }
        }

        __device__ void extend_q10() {
            if(variables.k == 0) {
                int v0 = getId(0);
                extend_single(v0, v0);
            }
            else if(variables.k >= 1 && variables.k <= 6) {
                int last = getId(variables.k);
                intersect(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k-1), getCurrentNumberOfExtensionsFixed(variables.k-1), getAdjacency(last), degree(last), last);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 7) {
                int last = getId(variables.k);
                extend_single(getId(0), -1);
                for(int i = 1 ; i <= variables.k ; i++)
                    intersect_unique(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k), getCurrentNumberOfExtensions(), getAdjacency(getId(i)), degree(getId(i)), -1);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 8) {
                int last = getId(variables.k);
                copy_extensions(variables.k-1, last);
            }
        }

        __device__ void extend_q11() {
            if(variables.k == 0) {
                int v0 = getId(0);
                extend_single(v0, v0);
            }
            else if(variables.k >= 1 && variables.k <= 7) {
                int last = getId(variables.k);
                intersect(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k-1), getCurrentNumberOfExtensionsFixed(variables.k-1), getAdjacency(last), degree(last), last);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 8) {
                int last = getId(variables.k);
                extend_single(getId(0), -1);
                for(int i = 1 ; i <= variables.k ; i++)
                    intersect_unique(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k), getCurrentNumberOfExtensions(), getAdjacency(getId(i)), degree(getId(i)), -1);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 9) {
                int last = getId(variables.k);
                copy_extensions(variables.k-1, last);
            }
        }

        __device__ void extend_q12() {
            if(variables.k == 0) {
                int v0 = getId(0);
                extend_single(v0, v0);
            }
            else if(variables.k >= 1 && variables.k <= 8) {
                int last = getId(variables.k);
                intersect(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k-1), getCurrentNumberOfExtensionsFixed(variables.k-1), getAdjacency(last), degree(last), last);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 9) {
                int last = getId(variables.k);
                extend_single(getId(0), -1);
                for(int i = 1 ; i <= variables.k ; i++)
                    intersect_unique(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k), getCurrentNumberOfExtensions(), getAdjacency(getId(i)), degree(getId(i)), -1);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 10) {
                int last = getId(variables.k);
                copy_extensions(variables.k-1, last);
            }
        }

        __device__ void extend_q13() {
            if(variables.k == 0) {
                int v0 = getId(0);
                extend_single(v0, v0);
            }
            else if(variables.k >= 1 && variables.k <= 9) {
                int last = getId(variables.k);
                intersect(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k-1), getCurrentNumberOfExtensionsFixed(variables.k-1), getAdjacency(last), degree(last), last);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 10) {
                int last = getId(variables.k);
                extend_single(getId(0), -1);
                for(int i = 1 ; i <= variables.k ; i++)
                    intersect_unique(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k), getCurrentNumberOfExtensions(), getAdjacency(getId(i)), degree(getId(i)), -1);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 11) {
                int last = getId(variables.k);
                copy_extensions(variables.k-1, last);
            }
        }

        __device__ void extend_dm14() {
            if(variables.k == 0) {
                int v0 = getId(0);
                extend_single(v0, v0);
            }
            else if(variables.k >= 1 && variables.k <= 10) {
                int last = getId(variables.k);
                intersect(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k-1), getCurrentNumberOfExtensionsFixed(variables.k-1), getAdjacency(last), degree(last), last);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 11) {
                int last = getId(variables.k);
                extend_single(getId(0), -1);
                for(int i = 1 ; i <= variables.k ; i++)
                    intersect_unique(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k), getCurrentNumberOfExtensions(), getAdjacency(getId(i)), degree(getId(i)), -1);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 12) {
                int last = getId(variables.k);
                copy_extensions(variables.k-1, last);
            }
        }

        __device__ void extend_wedge() {
            bool validExtension = false;
            int currentExtensionsSize = 0, last = getId(variables.k);
            int currentVertexDegree = degree(last);
            int *previousExtensions, previousExtensionsAmount;
            int found;
            if(variables.k > 0) {
                previousExtensions = getExtensionsArray(variables.k-1);
                previousExtensionsAmount = getCurrentNumberOfExtensionsFixed(variables.k-1);
            }
            
            for(int warpPosition = variables.lane, currentNeighbour ; warpPosition < roundToWarpSize(currentVertexDegree) ; warpPosition += 32) {
                currentNeighbour = neighbour(last, warpPosition);
                currentNeighbour = warpPosition < currentVertexDegree && currentNeighbour > last ? currentNeighbour : -1;
                validExtension = currentNeighbour != -1 ? true : false;
                
                if(variables.k > 0) {
                    found = findMany(previousExtensions, previousExtensionsAmount, currentNeighbour);
                    validExtension = currentNeighbour != -1 && found == 0 ? true : false;
                }
                currentExtensionsSize += write(getExtensionsArray(), currentExtensionsSize, currentNeighbour, validExtension);
                // printf("lane: %d ; k: %d ; id[0]: %d, id[1]: %d, id[2]: %d, extension: %d, valid: %d, currentExtensionsSize: %d\n", variables.lane, variables.k, getId(0), getId(1), getId(2), currentNeighbour, validExtension, currentExtensionsSize);
            }
            setCurrentNumberOfExtensions(currentExtensionsSize);
            setCurrentNumberOfExtensionsFixed(currentExtensionsSize);
        }

        __device__ void extend_chordal_4_EI() {   
            if(variables.k == 0) {
                extend_single(getId(0), -1);
            } else if(variables.k == 1) {
                int v0 = getId(0), v1 = getId(1);
                intersect(getExtensionsArray(1), getCurrentNumberOfExtensionsPointer(), getAdjacency(v0), degree(v0), getAdjacency(v1), degree(v1), v0);
            }
            else if(variables.k == 2) {
                int v0 = getId(0), v1 = getId(1), v2 = getId(2);
                intersect_unique(getExtensionsArray(2), getCurrentNumberOfExtensionsPointer(), getAdjacency(v0), degree(v0), getAdjacency(v2), degree(v2), v1);
            }            
        }

        __device__ void extend_q4_seed() {   
            if(variables.k == 0) {
                extend_single(getId(0), -1);
            } else if(variables.k == 1) {
                extend_single_unique(getId(1), -1);
            }
            else if(variables.k == 2) {
                extend_single_unique(getId(2), -1);
            }
            else if(variables.k == 3) {
                int v0 = getId(0), v1 = getId(1), v3 = getId(3);
                intersect_unique(getExtensionsArray(3), getCurrentNumberOfExtensionsPointer(), getAdjacency(v0), degree(v0), getAdjacency(v1), degree(v1), getAdjacency(v3), degree(v3), v1);
            }            
        }

        __device__ void extend_q5_seed() {   
            if(variables.k == 0) {
                int currentExtensionsSize = 0;
                bool validExtension = false;
                int v0 = getId(0);
                int currentVertexDegree = degree(v0);
                for(int warpPosition = variables.lane, v1 ; warpPosition < roundToWarpSize(currentVertexDegree) ; warpPosition += 32) {
                    v1 = neighbour(v0, warpPosition);
                    v1 = warpPosition < currentVertexDegree ? v1 : -1;
                    validExtension = v1 != -1 ? true : false;
                    currentExtensionsSize += write(getExtensionsArray(), currentExtensionsSize, v1, validExtension);
                    // printf("lane: %d ; k: %d ; id[0]: %d, id[1]: %d, id[2]: %d, extension: %d, valid: %d, currentExtensionsSize: %d\n", variables.lane, variables.k, getId(0), getId(1), getId(2), currentNeighbour, validExtension, currentExtensionsSize);
                }
                setCurrentNumberOfExtensions(currentExtensionsSize);
                
                // if(variables.lane == 0 && v0 == 1011) {
                //     printf("v0: [%d]", v0);
                //     for(int i = 0 ; i < getCurrentNumberOfExtensions(variables.k) ; i++) {
                //         printf(", v1: [%d]", getExtension(i));
                //     }
                //     printf("\n");
                // }
                // __syncwarp();

            } else if(variables.k == 1) {
                int v0 = getId(0), v1 = getId(1);
                intersect(getExtensionsArray(1), getCurrentNumberOfExtensionsPointer(), getAdjacency(v0), degree(v0), getAdjacency(v1), degree(v1), -1);
            }
            else if(variables.k == 2) {
                int v0 = getId(0), v2 = getId(2);
                intersect_unique(getExtensionsArray(2), getCurrentNumberOfExtensionsPointer(), getAdjacency(v0), degree(v0), getAdjacency(v2), degree(v2), -1);
            }
            else if(variables.k == 3) {
                int v0 = getId(0), v2 = getId(2), v3 = getId(3);
                intersect_unique(getExtensionsArray(2), getCurrentNumberOfExtensionsPointer(), getAdjacency(v0), degree(v0), getAdjacency(v3), degree(v3), v2);
            }
            else if(variables.k == 4) {
                int v0 = getId(0), v4 = getId(4);
                intersect_unique(getExtensionsArray(2), getCurrentNumberOfExtensionsPointer(), getAdjacency(v0), degree(v0), getAdjacency(v4), degree(v4), -1);
            }            
        }

        __device__ void extend_chordal_4_VI() {   
            if(variables.k == 0) {
                int currentExtensionsSize = 0;
                bool validExtension = false;
                int v0 = getId(0);
                int currentVertexDegree = degree(v0);
                for(int warpPosition = variables.lane, v1 ; warpPosition < roundToWarpSize(currentVertexDegree) ; warpPosition += 32) {
                    v1 = neighbour(v0, warpPosition);
                    v1 = warpPosition < currentVertexDegree ? v1 : -1;
                    validExtension = v1 != -1 ? true : false;
                    currentExtensionsSize += write(getExtensionsArray(), currentExtensionsSize, v1, validExtension);
                    // printf("lane: %d ; k: %d ; id[0]: %d, id[1]: %d, id[2]: %d, extension: %d, valid: %d, currentExtensionsSize: %d\n", variables.lane, variables.k, getId(0), getId(1), getId(2), currentNeighbour, validExtension, currentExtensionsSize);
                }
                setCurrentNumberOfExtensions(currentExtensionsSize);
                
                // if(variables.lane == 0 && v0 == 1011) {
                //     printf("v0: [%d]", v0);
                //     for(int i = 0 ; i < getCurrentNumberOfExtensions(variables.k) ; i++) {
                //         printf(", v1: [%d]", getExtension(i));
                //     }
                //     printf("\n");
                // }
                // __syncwarp();

            } else if(variables.k == 1) {
                int v0 = getId(0), v1 = getId(1);
                
                intersect(getExtensionsArray(1), getCurrentNumberOfExtensionsPointer(), getAdjacency(v0), degree(v0), getAdjacency(v1), degree(v1), v0);

                // if(variables.lane == 0 && v0 == 1011 && v1 == 1424) {
                //     printf("v0: [%d], v1: [%d] -> ", v0, v1);
                //     for(int i = 0 ; i < getCurrentNumberOfExtensions() ; i++) {
                //         printf("v2: [%d] ", getExtension(i));
                //     }
                //     printf("\n");
                // }
                // __syncwarp();
            }
            else if(variables.k == 2) {
                int v0 = getId(0), v1 = getId(1), v2 = getId(2);
                
                intersect_difference(getExtensionsArray(2), getCurrentNumberOfExtensionsPointer(), getAdjacency(v0), degree(v0), getAdjacency(v2), degree(v2), getAdjacency(v1), degree(v1), v1, v1);
                
                // if(variables.lane == 0 && v0 == 1011 && v1 == 1424 && v2 == 1541) {
                //     printf("v0: [%d], v1: [%d], v2: [%d] -> ", v0, v1, v2);
                //     for(int i = 0 ; i < getCurrentNumberOfExtensions() ; i++) {
                //         printf("v3: [%d] ", getExtension(i));
                //     }
                //     printf("\n");
                // }
                // __syncwarp();
            }            
        }

        __device__ void extend_clique_binary_search() {
            bool validExtension = false;
            int currentExtensionsSize = 0, last = getId(variables.k);
            int currentVertexDegree = degree(last), *adjacency = getAdjacency(last);;
            int *previousExtensions, previousExtensionsAmount;
            

            if(variables.k == 0) {
                for(int warpPosition = variables.lane, currentNeighbour ; warpPosition < roundToWarpSize(currentVertexDegree) ; warpPosition += 32) {
                    currentNeighbour = neighbour(last, warpPosition);
                    currentNeighbour = warpPosition < currentVertexDegree && currentNeighbour > last ? currentNeighbour : -1;
                    validExtension = currentNeighbour != -1 ? true : false;
                    currentExtensionsSize += write(getExtensionsArray(), currentExtensionsSize, currentNeighbour, validExtension);
                }
            }
            else if(variables.k > 0) {
                previousExtensions = getExtensionsArray(variables.k-1);
                previousExtensionsAmount = getCurrentNumberOfExtensionsFixed(variables.k-1);

                for(int i = 0 ; i < previousExtensionsAmount ; i++) {
                    int ext = previousExtensions[i];
                    int found = findOneBinarySearch(adjacency, currentVertexDegree, ext);
                    if(found)
                        setExtension(currentExtensionsSize++, ext);
                }
            }
            
            setCurrentNumberOfExtensions(currentExtensionsSize);
            setCurrentNumberOfExtensionsFixed(currentExtensionsSize);
        }

        __device__ void extend_clique_prepare() {
            bool validExtension = false;
            int currentExtensionsSize = 0, last = getId(variables.k);
            int currentVertexDegree = degree(last);
            int *previousExtensions, previousExtensionsAmount;
            if(variables.k > 0) {
                previousExtensions = getExtensionsArray(variables.k-1);
                previousExtensionsAmount = getCurrentNumberOfExtensionsFixed(variables.k-1);
            }
            
            for(int warpPosition = variables.lane, currentNeighbour ; warpPosition < roundToWarpSize(currentVertexDegree) ; warpPosition += 32) {
                currentNeighbour = neighbour(last, warpPosition);
                currentNeighbour = warpPosition < currentVertexDegree && currentNeighbour > last ? currentNeighbour : -1;
                validExtension = currentNeighbour != -1 ? true : false;
                
                if(variables.k > 0) {
                    int found = findMany(previousExtensions, previousExtensionsAmount, currentNeighbour);
                    // if(variables.k == 1 && getId(0) == 0) {
                    //     printf("previousExtensionsAmount: %d, %d,%d -> %d (degree %d), found: %d\n", previousExtensionsAmount, getId(0), getId(1), currentNeighbour, currentVertexDegree, found);
                    // }
                    validExtension = found == 1 ? true : false;
                }
                currentExtensionsSize += write(getExtensionsArray(), currentExtensionsSize, currentNeighbour, validExtension);
            }
            setCurrentNumberOfExtensionsFixed(currentExtensionsSize);
        }

        __device__ void extend_q8_prepare() {
            if(variables.k == 0) {
                int v0 = getId(0);
                extend_single(v0, v0);
            }
            else if(variables.k >= 1 && variables.k <= 4) {
                int last = getId(variables.k);
                intersect(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k-1), getCurrentNumberOfExtensionsFixed(variables.k-1), getAdjacency(last), degree(last), last);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 5) {
                int last = getId(variables.k);
                extend_single(getId(0), -1);
                for(int i = 1 ; i <= 5 ; i++)
                    intersect_unique(getExtensionsArray(variables.k), getCurrentNumberOfExtensionsPointer(), getExtensionsArray(variables.k), getCurrentNumberOfExtensions(), getAdjacency(getId(i)), degree(getId(i)), -1);
                setCurrentNumberOfExtensionsFixed(getCurrentNumberOfExtensions());
            }
            else if(variables.k == 6) {
                int last = getId(variables.k);
                copy_extensions(variables.k-1, last);
            }
        }

        __device__ void extend(int begin, int end) {
            int v0 = getId(begin);
            int currentOffsetExtensions = 0;
            bool validExtension;
            
            for(int i = begin, currentVertexDegree, id ; i <= end ; i++) {
                id = getId(i);
                currentVertexDegree = degree(id);

                for(int warpPosition = variables.lane, currentNeighbour, found ; warpPosition < roundToWarpSize(currentVertexDegree) ;  warpPosition += 32) {
                    currentNeighbour = warpPosition < currentVertexDegree ? neighbour(id, warpPosition) : -1;
                    found = findMany(getExtensionsArray(), currentOffsetExtensions, currentNeighbour);
                    __syncwarp();
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

        __device__ int* getCurrentNumberOfExtensionsPointer() {
            return dataGPU->d_numberOfExtensions + variables.offsetWarp + variables.k;
        }

        __device__ int getCurrentNumberOfExtensions(int k) {
            return dataGPU->d_numberOfExtensions[variables.offsetWarp+k];
        }

        __device__ int getCurrentNumberOfExtensionsFixed(int k) {
            return dataGPU->d_numberOfExtensionsFixed[variables.offsetWarp+k];
        }

        __device__ void setCurrentNumberOfExtensions(int value) {
            dataGPU->d_numberOfExtensions[variables.offsetWarp+variables.k] = value;
        }

        __device__ void setCurrentNumberOfExtensionsFixed(int value) {
            dataGPU->d_numberOfExtensionsFixed[variables.offsetWarp+variables.k] = value;
        }

        __device__ void filterClique() {
            int numberOfExtensions = getCurrentNumberOfExtensions(), currentExtensionsSize = 0;
            bool validExtension;

            for(int warpPosition = variables.lane, ext; warpPosition < roundToWarpSize(numberOfExtensions) ; warpPosition+=32) {
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
                __syncwarp();
                currentExtensionsSize += write(getExtensionsArray(), currentExtensionsSize, ext, validExtension);
            }
            setCurrentNumberOfExtensions(currentExtensionsSize);
        }

        __device__ void aggregate_counter() {
            dataGPU->d_result[variables.wid] += getCurrentNumberOfExtensions();
        }

        __device__ void aggregate_counter_chordal_4() {
            dataGPU->d_result[variables.wid] += (getCurrentNumberOfExtensions()*2);
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
