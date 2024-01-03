#include "Graph.h"
#include "Timer.h"
#include "Structs.cu"
#include "DuMatoCPU.h"
#include "DuMatoGPU.cu"
#include "Report.h"
#include <cuda_runtime.h>

__global__ void q9_PA(DataGPU *dataGPU) {
    DuMatoGPU DM_GPU;
    DM_GPU.start_q9(dataGPU);

    while(DM_GPU.active() && DM_GPU.balanced()) {
        if(DM_GPU.getCurrentNumberOfExtensions() == -1) {
            DM_GPU.extend_q9();
        }
        if(DM_GPU.getCurrentNumberOfExtensions() != 0) {
            if(DM_GPU.last_level()) {
                DM_GPU.aggregate_counter();
                DM_GPU.backward_q9();
            }
            else {
                DM_GPU.forward();
            }
        }
        else {
            DM_GPU.backward_q9();
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
    bool relabeling = false;
    bool patternAware = true;
    int rep = 0;
    /*************************************************/

    Timer timerIO;
    double timeLB = 0;

    timerIO.play("IO");
    DuMatoCPU *DM_CPU = new DuMatoCPU(datasetName, k, numberOfActiveThreads, blockSize, numberOfSMs, jobsPerWarp, q9_PA, globalThreshold, relabeling, patternAware);
    Report* report;
    if(rep)
        report = new Report(DM_CPU, 100);
    cudaDeviceSynchronize();
    timerIO.pause("IO");

    Timer timerGPU;
    if(globalThreshold >= 100) {
        timerGPU.play("Kernel");
        DM_CPU->runKernel();
        if(rep)
            report->start();
        DM_CPU->waitKernel();
        if(rep)
            report->stop();
        timerGPU.pause("Kernel");
    }
    else {
        timerGPU.play("Kernel");
        DM_CPU->runKernel();
        if(rep)
            report->start();
        while(true) {
            if(DM_CPU->gpuIsIdle()) {
                printf("[gpuIsIdle] %.2f.\n", DM_CPU->dataCPU->h_percentageWarpsIdle);
                DM_CPU->stopKernel();
                Timer timerLB;
                timerLB.play();
                if(DM_CPU->rebalance()) {
                    timerLB.pause();
                    timeLB += timerLB.getElapsedTimeInSeconds();
                    DM_CPU->runKernel();
                }
                else
                    break;
            }
            DM_CPU->sleepFor(100);
        } 
        DM_CPU->waitKernel();
        if(rep)
            report->stop();
        timerGPU.pause("Kernel");
    }

    DM_CPU->outputAggregateCounter();
    timerTOTAL.pause("timerTOTAL");

    printf("Time (GPU): %f secs.\n", timerGPU.getElapsedTimeInSeconds());
    printf("Time (LB): %f secs,%f%%\n", timeLB, (timeLB/timerGPU.getElapsedTimeInSeconds())*100);
    printf("Time (IO): %f secs\n", timerIO.getElapsedTimeInSeconds());
    printf("Time (TOTAL): %f secs\n", timerTOTAL.getElapsedTimeInSeconds());
    // /*************************************************/
    /************** Memory release *******************/
    delete DM_CPU;
    if(rep)
        delete report;
    /*************************************************/

    return 0;
}
