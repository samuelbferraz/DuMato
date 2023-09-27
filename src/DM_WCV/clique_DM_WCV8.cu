#include "Graph.h"
#include "Timer.h"
#include "Structs.cu"
#include "DuMatoCPU.h"
#include "DuMatoGPU.cu"
#include <cuda_runtime.h>

__global__ void clique(DataGPU *dataGPU) {
    DuMatoGPU DM_GPU;
    DM_GPU.start(dataGPU);

    while(DM_GPU.active() && DM_GPU.balanced()) {
        if(DM_GPU.getCurrentNumberOfExtensions() == -1) {
            DM_GPU.extend();
            if(DM_GPU.k() > 0) {
                DM_GPU.filterClique();                
            }
        }
        if(DM_GPU.getCurrentNumberOfExtensions() != 0) {
            if(DM_GPU.last_level()) {
                DM_GPU.aggregate_counter();
                DM_GPU.backward();
            }
            else {
                DM_GPU.forward();
            }
        }
        else {
            DM_GPU.backward();
        }
    }

    DM_GPU.end();
}

int main(int argc, const char** argv) {
    printf("Usage: %s graphFile k threads blockSize donationsPerWarp threshold\n", argv[0]);
    printf("\t graphFile: \t\t url of graph dataset\n");
    printf("\t k: \t\t\t clique size\n");
    printf("\t threads: \t\t amount of GPU threads (recommended: 102400)\n");
    printf("\t blockSize: \t\t amount of threads per block (recommended: 256)\n");
    printf("\t donationsPerWarp: \t amount of donations during load-balancing (recommended: 16)\n");
    printf("\t threshold: \t\t load-balancing threshold (recommended: 30)\n");

    if(argc != 7) {
        printf("\nWrong amount of parameters!\n");
        printf("Exiting...\n");
        exit(1);
    }

    Timer timerTOTAL;
    timerTOTAL.play("timerTOTAL");

    /*************************************************/
    /***************     Input    ********************/
    const char *datasetName = argv[1];
    int k = atoi(argv[2]);
    int numberOfActiveThreads = atoi(argv[3]);
    int blockSize = atoi(argv[4]);
    int numberOfSMs = 80;
    int jobsPerWarp = atoi(argv[5]);
    int globalThreshold = atoi(argv[6]);
    int validateResults = 0;
    int virtualWarpSize = 8;
    bool relabeling = false;
    /*************************************************/

    Timer timerIO;

    timerIO.play("IO");
    DuMatoCPU *DM_CPU = new DuMatoCPU(datasetName, k, numberOfActiveThreads, blockSize, numberOfSMs, jobsPerWarp, clique, globalThreshold, relabeling, virtualWarpSize);
    cudaDeviceSynchronize();
    timerIO.pause("IO");

    Timer timerGPU;

    // Without load balancing
    if(globalThreshold >= 100) {
        timerGPU.play();    
        DM_CPU->runKernel();
        DM_CPU->waitKernel();
        timerGPU.pause();
    }
    // With load balancing
    else {
        timerGPU.play();
        DM_CPU->runKernel();
        // report->start();
        while(true) {
            if(DM_CPU->gpuIsIdle()) {
                DM_CPU->stopKernel();
                if(DM_CPU->rebalance()) {
                    DM_CPU->runKernel();
                }
                else
                    break;
            }
            DM_CPU->sleepFor(100);
        }
        DM_CPU->waitKernel();
        // report->stop();
        timerGPU.pause();
    }
    printf("[END] Running kernel\n");
    DM_CPU->outputAggregateCounter();
    timerTOTAL.pause("timerTOTAL");

    if(validateResults) { 
        DM_CPU->validateAggregateCounter();
    }

    printf("Time (GPU): %f secs.\n", timerGPU.getElapsedTimeInSeconds());
    printf("Time (IO): %f secs\n", timerIO.getElapsedTimeInSeconds());
    printf("Time (TOTAL): %f secs\n", timerTOTAL.getElapsedTimeInSeconds());
    // /*************************************************/
    /************** Memory release *******************/
    delete DM_CPU;
    /*************************************************/

    return 0;
}
