#include "Graph.h"
#include "Timer.h"
#include "QuickMapping.h"
#include <cuda_runtime.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

typedef struct {
    int *d_vertexOffset;
    int *d_adjacencyList;
    int *d_degree;
} GraphGPU;

typedef struct {
    int *h_vertexOffset;
    int *h_adjacencyList;
    int *h_degree;
} GraphCPU;

typedef struct {
    int *d_k;                   // TODO Migrate to __const__
    int *d_id;
    int *d_jobsPerWarp;         // TODO Migrate to __const__
    int *d_jobs;
    int *d_currentJob;
    int *d_currentPosOfJob;
    int *d_validJobs;
    int *d_numberOfExtensions;
    int *d_extensions;
    int *d_extensionSources;
    int *d_currentPos;
    int *d_warpSize;            
    int *d_extensionsOffset;
    int *d_extensionsLength;
    unsigned long* d_result;
    unsigned long long *d_hashPerWarp;
    long unsigned int *d_quickToCgLocal;
    int *d_numberOfCgs;
    long unsigned int *d_localSubgraphInduction;
} EnumerationGPU;

typedef struct {
    unsigned long* h_result;
    int *h_jobs;
    int *h_currentJob;
    int *h_currentPosOfJob;
    int *h_validJobs;
    int *h_extensionsOffset;
    unsigned long long *h_hashPerWarp;
    long unsigned int *h_hashGlobal;
} EnumerationCPU;


typedef struct {
    // Input
    int h_k;
    int h_numberOfActiveThreads;
    int h_blockSize;
    int h_numberOfSMs;
    int h_jobsPerWarp;

    // Others
    int h_warpSize;
    int h_numberOfBlocks;
    int h_warpsPerBlock;
    int h_numberOfWarps;
    
    int h_maxVertexId;
    int h_maxDegree;
    int h_numberOfEdges;
    int h_extensionsLength;
    int h_initialJobsPerWarp;
    int h_theoreticalJobsPerWarp;
} ConstantsCPU;


void initializeCpuDataStructures(Graph *graphReader, ConstantsCPU *constantsCPU, GraphCPU *graphCPU, EnumerationCPU *enumerationCPU, QuickMapping *quickMapping);
void initializeConstantsCpu(ConstantsCPU *constantsCPU, Graph *graphReader);
void releaseCpuDataStructures(GraphCPU *graphCPU, EnumerationCPU *enumerationCPU);
void initializeGpuDataStructures(ConstantsCPU *constantsCPU, GraphCPU* graphCPU, GraphGPU *graphGPU, EnumerationCPU *enumerationCPU, EnumerationGPU *enumerationGPU, QuickMapping *quickMapping);
void releaseGpuDataStructures(GraphGPU *graphGPU, EnumerationGPU *enumerationGPU);


typedef struct {
    int tid;
    int wid;
    int lane;
    int k;
    int offsetWarp;
    int offsetExtensions;
    int offsetInductions;
    int offsetHash;
} GPULocalVariables;


__device__ int roundToWarpSize(int value, int warpSize) {
    return ((int)ceilf((float)value / (float)warpSize)) * warpSize;
}

__device__ int neighbour(int vertexId, int relativePosition, GraphGPU *graph) {
    return graph->d_adjacencyList[graph->d_vertexOffset[vertexId]+relativePosition];
}

__device__ int getCurrentJob(GPULocalVariables *variables, EnumerationGPU *enumerationGPU) {
    return enumerationGPU->d_currentJob[variables->wid];
}

__device__ int getValidJobs(GPULocalVariables *variables, EnumerationGPU *enumerationGPU) {
    return enumerationGPU->d_validJobs[variables->wid];
}

__device__ int getCurrentPosOfJob(GPULocalVariables *variables, EnumerationGPU *enumerationGPU) {
    return enumerationGPU->d_currentPosOfJob[variables->wid*(*enumerationGPU->d_jobsPerWarp)+getCurrentJob(variables, enumerationGPU)];
}

__device__ int getJob(GPULocalVariables *variables, EnumerationGPU *enumerationGPU) {
    return enumerationGPU->d_jobs[variables->wid*(*enumerationGPU->d_jobsPerWarp)*(32) + getCurrentJob(variables, enumerationGPU)*(32) + variables->lane];
}


__device__ void popJob(GPULocalVariables *variables, EnumerationGPU *enumerationGPU) {
    if(variables->k != -1)
        return;
    
    if(getCurrentJob(variables, enumerationGPU) >= getValidJobs(variables, enumerationGPU)) {
        // No more jobs... warp is going to quit.
        // Update status and smid variables, when load-balancing comes up.
    }
    else {
        variables->k = getCurrentPosOfJob(variables, enumerationGPU);
        enumerationGPU->d_id[variables->offsetWarp+variables->lane] = getJob(variables, enumerationGPU);
        enumerationGPU->d_localSubgraphInduction[variables->offsetInductions+variables->lane] = 0;
        enumerationGPU->d_numberOfExtensions[variables->offsetWarp+variables->lane] = 0; // initializeNumberOfExtensions(0)
        enumerationGPU->d_numberOfExtensions[variables->offsetWarp+variables->k] = -1; // setNumberOfExtensions(-1)
        enumerationGPU->d_currentJob[variables->wid]++; // increaseJob()
    }
}

__device__ void backward(GPULocalVariables *variables, EnumerationGPU *enumerationGPU) {
    variables->k = variables->k - 1;
    if(variables->k < 0) {
        popJob(variables, enumerationGPU);
    } 
}

__device__ void forward(GPULocalVariables *variables, EnumerationGPU *enumerationGPU, GraphGPU *graph) {
    int numberOfExtensions = enumerationGPU->d_numberOfExtensions[variables->offsetWarp+variables->k];
    int localOffsetExtensions = variables->offsetExtensions + enumerationGPU->d_extensionsOffset[variables->k];
    int nextEmbeddingID = enumerationGPU->d_extensions[localOffsetExtensions+numberOfExtensions-1];
    enumerationGPU->d_numberOfExtensions[variables->offsetWarp+variables->k]--;
    variables->k = variables->k + 1;
    enumerationGPU->d_id[variables->offsetWarp+variables->k] = nextEmbeddingID;
    enumerationGPU->d_numberOfExtensions[variables->offsetWarp+variables->k] = -1;

    if(variables->k >= 2) {
        // (((dm_k()-2)*(2+dm_k()-1))/2) -> Offset created by previous inductions (sum of PA starting in 2)
        int localOffsetInduction = (((variables->k-2)*(2+variables->k-1))/2);

        unsigned long quickPattern = 0;
        for(int i = 0, currentPow = powf(2,localOffsetInduction), found = 0, currentDegree,  vertexId ; i < variables->k ; i++, currentPow*=2) {
            vertexId = enumerationGPU->d_id[variables->offsetWarp+i];
            currentDegree = graph->d_degree[vertexId];

            for(int warpPosition = variables->lane ; warpPosition < roundToWarpSize(currentDegree, 32) && !found ; warpPosition += 32)
                found = __any_sync(0xffffffff, warpPosition < currentDegree && neighbour(vertexId, warpPosition, graph) == nextEmbeddingID ? 1 : 0);
            
            quickPattern += (found*currentPow);
        }
        enumerationGPU->d_localSubgraphInduction[variables->offsetInductions+variables->k] = enumerationGPU->d_localSubgraphInduction[variables->offsetInductions+variables->k-1] + quickPattern;
    }
}

__global__ void motifs(GraphGPU *graph, EnumerationGPU *enumerationGPU) {
    GPULocalVariables variables;

    variables.tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    variables.wid = variables.tid / 32;
    variables.lane = threadIdx.x & 0x1f;
    variables.offsetWarp = variables.wid * 32;
    variables.offsetExtensions = variables.wid * *(enumerationGPU->d_extensionsLength);
    variables.offsetInductions = variables.wid * 32;
    variables.offsetHash = variables.wid * *(enumerationGPU->d_numberOfCgs);
    
    /*********************/
    /****  dm_start() ****/
    enumerationGPU->d_result[variables.wid] = 0;                                                    // result[dm_getWid()] = 0;
    variables.k = -1;                                                                               // dm_k(currentPos[dm_getWid()]);
    popJob(&variables, enumerationGPU);                                                             // dm_popJob();
    /*********************/

    while(variables.k >= 0) {                                                                       // while(dm_active() && dm_gpuIsBalanced())
        if(enumerationGPU->d_numberOfExtensions[variables.offsetWarp+variables.k] == -1) {          // if(dm_numberOfExtensions() == -1)
            /**************************************************/
            /** [BEGIN] generateUniqueExtensions(0, dm_k()) **/
            int localOffsetExtensions = variables.offsetExtensions + enumerationGPU->d_extensionsOffset[variables.k];
            int currentOffsetExtensions = 0;
            int v0 = enumerationGPU->d_id[variables.offsetWarp];
            unsigned long pos, actives, totalActives, activesOnMyRight, idlesOnMyRight;

            enumerationGPU->d_numberOfExtensions[variables.offsetWarp+variables.k] = 0;

            for(int i = 0, currentVertexDegree, currentVertexDegreeRounded ; i <= variables.k ; i++) {
                int id = enumerationGPU->d_id[variables.offsetWarp+i];
                currentVertexDegree = graph->d_degree[id];
                currentVertexDegreeRounded = roundToWarpSize(currentVertexDegree, 32);

                for(int warpPosition = variables.lane, currentNeighbour ; warpPosition < currentVertexDegreeRounded ;  warpPosition += 32) {
                    currentNeighbour = neighbour(id, warpPosition, graph);
                    currentNeighbour = warpPosition < currentVertexDegree && currentNeighbour > v0 ? currentNeighbour : -1;
                    bool found = false;
                    for(int warpPosition = 0 ; warpPosition < currentOffsetExtensions && !found && currentNeighbour != -1 ; warpPosition++)
                        found = enumerationGPU->d_extensions[localOffsetExtensions+warpPosition] == currentNeighbour;
                    __syncwarp();
                    currentNeighbour = found ? -1 : currentNeighbour;
                    
                    actives = __ballot_sync(0xffffffff, currentNeighbour == -1 ? 0 : 1);
                    totalActives = __popc(actives);
                    actives = (actives << ((unsigned int)32-(unsigned int)variables.lane));
                    activesOnMyRight = __popc(actives);
                    idlesOnMyRight = variables.lane - activesOnMyRight;

                    pos = currentNeighbour != -1 ? activesOnMyRight : totalActives + idlesOnMyRight;
                    enumerationGPU->d_extensions[localOffsetExtensions+currentOffsetExtensions+pos] = currentNeighbour;
                    enumerationGPU->d_extensionSources[localOffsetExtensions+currentOffsetExtensions+pos] = i;
                    currentOffsetExtensions += totalActives;
                }
            }
            enumerationGPU->d_numberOfExtensions[variables.offsetWarp+variables.k] = currentOffsetExtensions;
            /** [END] generateUniqueExtensions(0, dm_k())   **/

            /**************************************************/
            /********* [BEGIN] canonicalFilter() ***************/
            if(variables.k > 0) {
                int currentOffsetExtensionsNew;
                for(int i = 1, target ; i <= variables.k ; i++) {
                    target = enumerationGPU->d_id[variables.offsetWarp+i];;
                    currentOffsetExtensionsNew = 0;
                    for(int warpPosition = variables.lane, ext, src ; warpPosition < roundToWarpSize(currentOffsetExtensions, 32) ; warpPosition += 32) {
                        ext = warpPosition < currentOffsetExtensions ? enumerationGPU->d_extensions[localOffsetExtensions+warpPosition] : -1;
                        src = warpPosition < currentOffsetExtensions ? enumerationGPU->d_extensionSources[localOffsetExtensions+warpPosition] : -1;
                        
                        ext = (i > src && ext <= target) || (i < src && ext == target) ? -1 : ext; 

                        actives = __ballot_sync(0xffffffff, ext == -1 ? 0 : 1);
                        totalActives = __popc(actives);
                        actives = (actives << ((unsigned int)32-(unsigned int)variables.lane));
                        activesOnMyRight = __popc(actives);
                        idlesOnMyRight = variables.lane - activesOnMyRight;

                        pos = ext != -1 ? activesOnMyRight : totalActives + idlesOnMyRight;
                        enumerationGPU->d_extensions[localOffsetExtensions+currentOffsetExtensionsNew+pos] = ext;
                        enumerationGPU->d_extensionSources[localOffsetExtensions+currentOffsetExtensionsNew+pos] = src;
                        currentOffsetExtensionsNew += totalActives;
                    }
                    currentOffsetExtensions = currentOffsetExtensionsNew;
                }
                enumerationGPU->d_numberOfExtensions[variables.offsetWarp+variables.k] = currentOffsetExtensions;
            }
            /********** [END] canonicalFilter() ***************/
        }

        int numberOfExtensions = enumerationGPU->d_numberOfExtensions[variables.offsetWarp+variables.k];
        if(numberOfExtensions != 0) {
            if(variables.k == *enumerationGPU->d_k-2) {
                /************************************/
                /* [BEGIN] accumulateValidSubgraphs */
                enumerationGPU->d_result[variables.wid] += numberOfExtensions;
                /************************************/

                /*********************************/ 
                /*[BEGIN] aggregateValidSubgraphs*/
                int localOffsetExtensions = variables.offsetExtensions + enumerationGPU->d_extensionsOffset[variables.k];
                unsigned long quickPattern = 0;
                int localOffsetInduction = (((variables.k-1)*(2+variables.k))/2);

                for(int warpPosition = variables.lane, nextEmbeddingId ; warpPosition < roundToWarpSize(numberOfExtensions, 32) ; warpPosition+=32) {
                    if(warpPosition < numberOfExtensions) {
                        nextEmbeddingId = enumerationGPU->d_extensions[localOffsetExtensions+warpPosition];

                        // Final induction (quick pattern)
                        quickPattern = 0;
                        for(int j = 0, currentPow = powf(2,localOffsetInduction), found, currentDegree, vertexId ; j <= variables.k ; j++, currentPow*=2) {
                            found = 0; 
                            vertexId = enumerationGPU->d_id[variables.offsetWarp+j];
                            currentDegree = graph->d_degree[vertexId];
                            for(int warpPosition = 0 ; warpPosition < currentDegree && !found ; warpPosition++) {
                                if(neighbour(vertexId, warpPosition, graph) == nextEmbeddingId)
                                    found = 1;
                            }
                            quickPattern += (found*currentPow);
                        }
                    }
                    __syncwarp();
                    if(warpPosition < numberOfExtensions) {
                        quickPattern += enumerationGPU->d_localSubgraphInduction[variables.offsetInductions+variables.k];
                        atomicAdd_block(&(enumerationGPU->d_hashPerWarp[variables.offsetHash+enumerationGPU->d_quickToCgLocal[quickPattern]]), 1);
                    }
                    __syncwarp();
                }
                /*[END] aggregateValidSubgraphs*/

                backward(&variables, enumerationGPU);
            }
            else {
                forward(&variables, enumerationGPU, graph);
            }
        }
        else {
            backward(&variables, enumerationGPU);
        }
    }

    // Silly condition just to check the kernel finished without errors 
    if(variables.tid == 0)
        printf("After all, we are only human...\n");
}

int main(int argc, const char** argv) {
    printf("Usage: ./motifs_HAND_WC graphFile k threads blockSize\n");
    printf("\t graphFile: \t url of graph dataset\n");
    printf("\t k: \t\t clique size\n");
    printf("\t threads: \t amount of GPU threads (recommended: 102400)\n");
    printf("\t blockSize: \t amount of threads per block (recommended: 256)\n");
    if(argc != 5) {
        printf("\nWrong amount of parameters!\n");
        printf("Exiting...\n");
        exit(1);
    }

    ConstantsCPU *constantsCPU = (ConstantsCPU*)malloc(sizeof(ConstantsCPU));
    Timer timer;

    Graph *graphReader = new Graph(argv[1]);
    constantsCPU->h_k = atoi(argv[2]);
    constantsCPU->h_numberOfActiveThreads = atoi(argv[3]);
    constantsCPU->h_blockSize = atoi(argv[4]);
    constantsCPU->h_numberOfSMs = 80;
    constantsCPU->h_jobsPerWarp = 16;    
    initializeConstantsCpu(constantsCPU, graphReader);

    QuickMapping *quickMapping = new QuickMapping(constantsCPU->h_k);
    
    GraphCPU *graphCPU = (GraphCPU*)malloc(sizeof(GraphCPU));
    EnumerationCPU *enumerationCPU = (EnumerationCPU*)malloc(sizeof(EnumerationCPU));
    initializeCpuDataStructures(graphReader, constantsCPU, graphCPU, enumerationCPU, quickMapping);

    GraphGPU *graphGPU = (GraphGPU*)malloc(sizeof(GraphGPU));
    EnumerationGPU *enumerationGPU = (EnumerationGPU*)malloc(sizeof(EnumerationGPU));
    initializeGpuDataStructures(constantsCPU, graphCPU, graphGPU, enumerationCPU, enumerationGPU, quickMapping);

    // printf("Number of blocks: %d, block size: %d\n", constantsCPU->h_numberOfBlocks, constantsCPU->h_blockSize);

    GraphGPU *d_graphGPU;
    gpuErrchk(cudaMalloc((void**)&d_graphGPU, sizeof(GraphGPU)));
    gpuErrchk(cudaMemcpy(d_graphGPU, graphGPU, sizeof(GraphGPU), cudaMemcpyHostToDevice));

    EnumerationGPU *d_enumerationGPU;
    gpuErrchk(cudaMalloc((void**)&d_enumerationGPU, sizeof(EnumerationGPU)));
    gpuErrchk(cudaMemcpy(d_enumerationGPU, enumerationGPU, sizeof(EnumerationGPU), cudaMemcpyHostToDevice));

    timer.play();
    motifs<<<constantsCPU->h_numberOfBlocks, constantsCPU->h_blockSize>>>(d_graphGPU, d_enumerationGPU);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    /***********************/
    /* Organizing results. */
    gpuErrchk(cudaMemcpy(enumerationCPU->h_result, enumerationGPU->d_result, constantsCPU->h_numberOfWarps*sizeof(unsigned long), cudaMemcpyDeviceToHost));
    unsigned long result = 0;
    for(int i = 0 ; i < constantsCPU->h_numberOfWarps ; i++) {
        result += enumerationCPU->h_result[i];
    }
    gpuErrchk(cudaMemcpy(enumerationCPU->h_hashPerWarp, enumerationGPU->d_hashPerWarp, constantsCPU->h_numberOfWarps * quickMapping->numberOfCgs * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    for(int i = 0 ; i < constantsCPU->h_numberOfWarps ; i++) {
        for(int j = 0 ; j < quickMapping->numberOfCgs ; j++) {
            enumerationCPU->h_hashGlobal[j] += enumerationCPU->h_hashPerWarp[i*quickMapping->numberOfCgs+j];
        }
    }
    long unsigned int validSubgraphs = 0;
    for(int i = 0 ; i < quickMapping->numberOfCgs ; i++) {
        if(enumerationCPU->h_hashGlobal[i] > 0) {
            validSubgraphs += enumerationCPU->h_hashGlobal[i];
        }
    }
    /************************/
    timer.pause();

    printf("Result: %lu (Aggregation: %s), tempo: %f\n", result, result == validSubgraphs ? "MATCH" : "MISMATCH", timer.getElapsedTimeInSeconds());

    /*************************************************/
    /************** Memory release *******************/
    releaseGpuDataStructures(graphGPU, enumerationGPU);
    releaseCpuDataStructures(graphCPU, enumerationCPU);
    delete graphReader;
    delete quickMapping;
    free(constantsCPU);
    /*************************************************/


    return 0;
}

void initializeGpuDataStructures(ConstantsCPU *constantsCPU, GraphCPU *graphCPU, GraphGPU *graphGPU, EnumerationCPU *enumerationCPU, EnumerationGPU *enumerationGPU, QuickMapping *quickMapping) {
    /***************************************/
    /************ Graph related ************/
    gpuErrchk(cudaMalloc((void**)&graphGPU->d_vertexOffset, (constantsCPU->h_maxVertexId+2)*sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&graphGPU->d_adjacencyList, (constantsCPU->h_numberOfEdges*2 + (constantsCPU->h_maxVertexId+1)) * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&graphGPU->d_degree, (constantsCPU->h_maxVertexId+1)*sizeof(int)));

    gpuErrchk(cudaMemcpy(graphGPU->d_vertexOffset, graphCPU->h_vertexOffset, (constantsCPU->h_maxVertexId+2)*sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(graphGPU->d_adjacencyList, graphCPU->h_adjacencyList, (constantsCPU->h_numberOfEdges*2 + (constantsCPU->h_maxVertexId+1)) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(graphGPU->d_degree, graphCPU->h_degree, (constantsCPU->h_maxVertexId+1)*sizeof(int), cudaMemcpyHostToDevice));
    /***************************************/

    /***************************************/
    /******** Enumeration related **********/
    gpuErrchk(cudaMalloc((void**)&(enumerationGPU->d_k), sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&(enumerationGPU->d_extensionsLength), sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&(enumerationGPU->d_id), constantsCPU->h_numberOfWarps * constantsCPU->h_warpSize * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&(enumerationGPU->d_jobsPerWarp), sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&(enumerationGPU->d_jobs), constantsCPU->h_numberOfWarps * constantsCPU->h_theoreticalJobsPerWarp * constantsCPU->h_warpSize * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&(enumerationGPU->d_currentJob), constantsCPU->h_numberOfWarps * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&(enumerationGPU->d_currentPosOfJob), constantsCPU->h_numberOfWarps * constantsCPU->h_theoreticalJobsPerWarp * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&(enumerationGPU->d_validJobs), constantsCPU->h_numberOfWarps * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&(enumerationGPU->d_numberOfExtensions), constantsCPU->h_numberOfWarps * constantsCPU->h_warpSize * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&(enumerationGPU->d_extensions), constantsCPU->h_numberOfWarps * constantsCPU->h_extensionsLength * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&(enumerationGPU->d_extensionSources), constantsCPU->h_numberOfWarps * constantsCPU->h_extensionsLength * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&(enumerationGPU->d_currentPos), constantsCPU->h_numberOfWarps * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&(enumerationGPU->d_result), constantsCPU->h_numberOfWarps * sizeof(unsigned long)));
    gpuErrchk(cudaMalloc((void**)&(enumerationGPU->d_warpSize), sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&(enumerationGPU->d_extensionsOffset), (constantsCPU->h_k-1) * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&(enumerationGPU->d_hashPerWarp), constantsCPU->h_numberOfWarps * quickMapping->numberOfCgs * sizeof(unsigned long long)));
    gpuErrchk(cudaMalloc((void**)&(enumerationGPU->d_quickToCgLocal), quickMapping->numberOfQuicks * sizeof(long unsigned int)));
    gpuErrchk(cudaMalloc((void**)&(enumerationGPU->d_numberOfCgs), sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&(enumerationGPU->d_localSubgraphInduction), constantsCPU->h_numberOfWarps * constantsCPU->h_warpSize * sizeof(long unsigned int)));

    gpuErrchk(cudaMemcpy(enumerationGPU->d_jobsPerWarp, &(constantsCPU->h_theoreticalJobsPerWarp), sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(enumerationGPU->d_jobs, enumerationCPU->h_jobs, constantsCPU->h_numberOfWarps * constantsCPU->h_theoreticalJobsPerWarp * constantsCPU->h_warpSize * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(enumerationGPU->d_currentJob, enumerationCPU->h_currentJob, constantsCPU->h_numberOfWarps * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(enumerationGPU->d_validJobs, enumerationCPU->h_validJobs, constantsCPU->h_numberOfWarps * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(enumerationGPU->d_currentPosOfJob, enumerationCPU->h_currentPosOfJob, constantsCPU->h_numberOfWarps * constantsCPU->h_theoreticalJobsPerWarp * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(enumerationGPU->d_warpSize, &(constantsCPU->h_warpSize), sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(enumerationGPU->d_extensionsOffset, enumerationCPU->h_extensionsOffset, (constantsCPU->h_k-1) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(enumerationGPU->d_k, &constantsCPU->h_k, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(enumerationGPU->d_extensionsLength, &constantsCPU->h_extensionsLength, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(enumerationGPU->d_quickToCgLocal, quickMapping->quickToCgLocal, quickMapping->numberOfQuicks * sizeof(long unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(enumerationGPU->d_numberOfCgs, &(quickMapping->numberOfCgs), sizeof(int), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemset(enumerationGPU->d_hashPerWarp, 0, constantsCPU->h_numberOfWarps * quickMapping->numberOfCgs * sizeof(unsigned long long)));
    gpuErrchk(cudaMemset(enumerationGPU->d_localSubgraphInduction, 0, constantsCPU->h_numberOfWarps * constantsCPU->h_warpSize * sizeof(long unsigned int)));
    /***************************************/
}

void releaseGpuDataStructures(GraphGPU *graphGPU, EnumerationGPU *enumerationGPU) {
    // Graph related
    cudaFree(graphGPU->d_vertexOffset);
    cudaFree(graphGPU->d_adjacencyList);
    cudaFree(graphGPU->d_degree); 
    // free(graphGPU);

    // Enumeration related
    cudaFree(enumerationGPU->d_k);
    cudaFree(enumerationGPU->d_extensionsLength);
    cudaFree(enumerationGPU->d_id);
    cudaFree(enumerationGPU->d_jobsPerWarp);
    cudaFree(enumerationGPU->d_jobs);
    cudaFree(enumerationGPU->d_currentJob);
    cudaFree(enumerationGPU->d_currentPosOfJob); 
    cudaFree(enumerationGPU->d_validJobs);
    cudaFree(enumerationGPU->d_numberOfExtensions);
    cudaFree(enumerationGPU->d_extensions); 
    cudaFree(enumerationGPU->d_currentPos); 
    cudaFree(enumerationGPU->d_result);
    cudaFree(enumerationGPU->d_warpSize);
    cudaFree(enumerationGPU->d_extensionsOffset);
    cudaFree(enumerationGPU->d_hashPerWarp);
    cudaFree(enumerationGPU->d_quickToCgLocal);
    cudaFree(enumerationGPU->d_numberOfCgs);
    cudaFree(enumerationGPU->d_extensionSources);
    cudaFree(enumerationGPU->d_localSubgraphInduction);
    // free(enumerationGPU);
}

void initializeCpuDataStructures(Graph *graphReader, ConstantsCPU *constantsCPU, GraphCPU *graphCPU, EnumerationCPU *enumerationCPU, QuickMapping *quickMapping) {
    graphCPU->h_vertexOffset = (int*)malloc((constantsCPU->h_maxVertexId) * sizeof(int));
    graphCPU->h_adjacencyList = (int*)malloc(((constantsCPU->h_numberOfEdges*2) + (constantsCPU->h_maxVertexId+1)) * sizeof(int));
    graphCPU->h_degree = (int*)malloc((constantsCPU->h_maxVertexId+1)*sizeof(int));

    // Initialize CSR graph data structures for GPU
    int offset = 0;
    for(int vertexId = 0 ; vertexId <= graphReader->getMaxVertexId() ; vertexId++) {
        graphCPU->h_vertexOffset[vertexId] = offset;
        // printf("%d, vertexOffset: %d\n", vertexId, h_vertexOffset[vertexId]);
        for(std::set<int>::iterator itEdges = graphReader->getNeighbours(vertexId)->begin() ; itEdges != graphReader->getNeighbours(vertexId)->end() ; itEdges++)
            graphCPU->h_adjacencyList[offset++] = *itEdges;
        graphCPU->h_adjacencyList[offset++] = -1;

        graphCPU->h_degree[vertexId] = graphReader->getNeighbours(vertexId)->size();
        // printf("%d, vertexOffset: %d, degree: %d\n", vertexId, h_vertexOffset[vertexId], h_degree[vertexId]);
    }
    graphCPU->h_vertexOffset[graphReader->getMaxVertexId()+1] = graphCPU->h_vertexOffset[graphReader->getMaxVertexId()]+graphCPU->h_degree[graphReader->getMaxVertexId()]+1;

    enumerationCPU->h_hashPerWarp = (unsigned long long*)malloc(constantsCPU->h_numberOfWarps*quickMapping->numberOfCgs * sizeof(unsigned long long));
    enumerationCPU->h_hashGlobal = (long unsigned int*)malloc(quickMapping->numberOfCgs * sizeof(long unsigned int));
    enumerationCPU->h_result = (unsigned long*)malloc(constantsCPU->h_numberOfWarps * sizeof(unsigned long));
    enumerationCPU->h_currentJob = (int*)malloc(constantsCPU->h_numberOfWarps * sizeof(int));
    enumerationCPU->h_currentPosOfJob = (int*)malloc(constantsCPU->h_numberOfWarps * constantsCPU->h_theoreticalJobsPerWarp * sizeof(int));
    enumerationCPU->h_validJobs = (int*)malloc(constantsCPU->h_numberOfWarps * sizeof(int));
    enumerationCPU->h_jobs = (int*)malloc(constantsCPU->h_numberOfWarps * constantsCPU->h_theoreticalJobsPerWarp * constantsCPU->h_warpSize * sizeof(int));

    for(int i = 0 ; i < constantsCPU->h_numberOfWarps ; i++) {
        enumerationCPU->h_currentJob[i] = 0;
        enumerationCPU->h_validJobs[i] = 0;
        for(int j = 0 ; j < quickMapping->numberOfCgs ; j++)
            enumerationCPU->h_hashPerWarp[i*quickMapping->numberOfCgs + j] = 0;
    }
    for(int i = 0 ; i < quickMapping->numberOfCgs ; i++)
        enumerationCPU->h_hashGlobal[i] = 0;

    for(int round = 0 ; round < constantsCPU->h_initialJobsPerWarp ; round++) {
        for(int i = 0 ; i < constantsCPU->h_numberOfWarps ; i++) {
            int jobId = round*constantsCPU->h_numberOfWarps+i;
            if(jobId <= constantsCPU->h_maxVertexId) {
                enumerationCPU->h_validJobs[i]++;
                enumerationCPU->h_jobs[i*constantsCPU->h_theoreticalJobsPerWarp*constantsCPU->h_warpSize + round*constantsCPU->h_warpSize + 0] = jobId;
                enumerationCPU->h_currentPosOfJob[i*constantsCPU->h_theoreticalJobsPerWarp+round] = 0;
            } 
        }
    }

    constantsCPU->h_extensionsLength = 0;
    enumerationCPU->h_extensionsOffset = (int*)malloc(constantsCPU->h_k * sizeof(int));
    enumerationCPU->h_extensionsOffset[0] = 0;
    for(int k = 1, length ; k <= constantsCPU->h_k - 1 ; k++) {
        length = (int)ceilf(pow(2, ceilf(log2((float)(k * constantsCPU->h_maxDegree)))));
        constantsCPU->h_extensionsLength += length;
        if(k < constantsCPU->h_k - 1)
            enumerationCPU->h_extensionsOffset[k] = constantsCPU->h_extensionsLength;
    }
}

void releaseCpuDataStructures(GraphCPU *graphCPU, EnumerationCPU *enumerationCPU) {
    // free(graphCPU->h_vertexOffset);
    // free(graphCPU->h_adjacencyList);
    // free(graphCPU->h_degree);
    // free(graphCPU);

    // free(enumerationCPU->h_result);
    // free(enumerationCPU->h_currentJob);
    // free(enumerationCPU->h_currentPosOfJob);
    // free(enumerationCPU->h_validJobs);
    // free(enumerationCPU->h_jobs);
    // free(enumerationCPU->h_extensionsOffset);
    // free(enumerationCPU->h_hashPerWarp)
    // free(enumerationCPU->h_hashGlobal);
    // free(enumerationCPU);
}

void initializeConstantsCpu(ConstantsCPU* constantsCPU, Graph* graphReader) {
    constantsCPU->h_maxVertexId = graphReader->getMaxVertexId();
    constantsCPU->h_numberOfEdges = graphReader->getNumberOfEdges();
    constantsCPU->h_maxDegree = graphReader->getMaxDegree();

    constantsCPU->h_warpSize = 32;
    constantsCPU->h_numberOfBlocks = ceil(constantsCPU->h_numberOfActiveThreads/(float)constantsCPU->h_blockSize);
    constantsCPU->h_warpsPerBlock = constantsCPU->h_blockSize / constantsCPU->h_warpSize;
    constantsCPU->h_numberOfWarps = constantsCPU->h_numberOfBlocks * constantsCPU->h_warpsPerBlock;

    constantsCPU->h_initialJobsPerWarp = ceil((constantsCPU->h_maxVertexId+1)/(float)constantsCPU->h_numberOfWarps);    
    constantsCPU->h_theoreticalJobsPerWarp = std::max(constantsCPU->h_initialJobsPerWarp, constantsCPU->h_jobsPerWarp);
}
