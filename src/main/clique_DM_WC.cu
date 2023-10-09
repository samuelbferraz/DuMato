#include "Graph.h"
#include "Timer.h"
#include "Structs.cu"
#include "DuMatoCPU.h"
#include "DuMatoGPU.cu"
#include "Report.h"
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
    printf("Usage: %s graphFile k threads blockSize\n", argv[0]);
    printf("\t graphFile: \t\t url of graph dataset\n");
    printf("\t k: \t\t\t clique size\n");
    printf("\t threads: \t\t amount of GPU threads (recommended: 102400)\n");
    printf("\t blockSize: \t\t amount of threads per block (recommended: 256)\n");

    if(argc != 5) {
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
    int jobsPerWarp = 16;
    int globalThreshold = 101;
    bool relabeling = false;
    /*************************************************/

    Timer timerIO;

    timerIO.play("IO");
    DuMatoCPU *DM_CPU = new DuMatoCPU(datasetName, k, numberOfActiveThreads, blockSize, numberOfSMs, jobsPerWarp, clique, globalThreshold, relabeling);
    // Report* report = new Report(DM_CPU, 100);
    cudaDeviceSynchronize();
    timerIO.pause("IO");

    Timer timerGPU;
    if(globalThreshold >= 100) {
        timerGPU.play("Kernel");
        DM_CPU->runKernel();
        // report->start();
        DM_CPU->waitKernel();
        // report->stop();
        timerGPU.pause("Kernel");
    }
    else {
        timerGPU.play("Kernel");
        DM_CPU->runKernel();
        // report->start();
        while(true) {
            if(DM_CPU->gpuIsIdle()) {
                printf("[gpuIsIdle] %.2f.\n", DM_CPU->dataCPU->h_percentageWarpsIdle);
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
        timerGPU.pause("Kernel");
    }

    DM_CPU->outputAggregateCounter();
    timerTOTAL.pause("timerTOTAL");

    printf("Time (GPU): %f secs.\n", timerGPU.getElapsedTimeInSeconds());
    printf("Time (IO): %f secs\n", timerIO.getElapsedTimeInSeconds());
    printf("Time (TOTAL): %f secs\n", timerTOTAL.getElapsedTimeInSeconds());
    // /*************************************************/
    /************** Memory release *******************/
    delete DM_CPU;
    // delete report;
    /*************************************************/

    return 0;
}
