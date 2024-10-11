#include "Graph.h"
#include "Timer.h"
#include "Structs.cu"
#include "DuMatoCPU.h"
#include "DuMatoGPU.cu"
#include "Report.h"
#include <cuda_runtime.h>

__global__ void motifs(DataGPU *dataGPU) {
    DuMatoGPU DM_GPU;
    DM_GPU.start_induce(dataGPU);

    while(DM_GPU.active() && DM_GPU.balanced()) {
        if(DM_GPU.getCurrentNumberOfExtensions() == -1) {
            DM_GPU.extend(0, DM_GPU.k());
            if(DM_GPU.k() > 0) {
                DM_GPU.canonicalFilter2();                
            }
        }
        if(DM_GPU.getCurrentNumberOfExtensions() != 0) {
            if(DM_GPU.last_level()) {
		        // DM_GPU.debug_subgraphs();
                DM_GPU.aggregate_pattern();
                DM_GPU.backward_induce();
            }
            else {
                DM_GPU.forward_induce();
            }
        }
        else {
            DM_GPU.backward_induce();
        }
    }

    DM_GPU.end();
}

int main(int argc, const char** argv) {
    
    if(argc != 8) {
        printf("Usage: %s graphFile k threads blockSize donationsPerWarp threshold\n", argv[0]);
        printf("\t graphFile: \t\t url of graph dataset\n");
        printf("\t k: \t\t\t motifs size\n");
        printf("\t threads: \t\t amount of GPU threads (recommended: 102400)\n");
        printf("\t blockSize: \t\t amount of threads per block (recommended: 256)\n");
        printf("\t donations: \t amount of donations during load-balancing (recommended: 16)\n");
        printf("\t threshold: \t\t load-balancing threshold (recommended: 30)\n");
        printf("\t report: \t\t 0 (no) or 1 (yes)\n");

        printf("\nWrong amount of parameters!\n");
        printf("Exiting...\n");
        exit(1);
    }

    Timer timerTOTAL;
    timerTOTAL.play();

    /*************************************************/
    /***************     Input    ********************/
    const char *datasetName = argv[1];
    int k = atoi(argv[2]);
    int numberOfActiveThreads = atoi(argv[3]);
    int blockSize = atoi(argv[4]);
    int numberOfSMs = 80;
    int jobsPerWarp = atoi(argv[5]);
    int globalThreshold = atoi(argv[6]);
    int rep = atoi(argv[7]);
    int validateResults = 0;
    bool relabeling = true;
    bool patternAware = false;
    
    /*************************************************/

    printf("Usage: %s graphFile k threads blockSize donationsPerWarp threshold report\n", argv[0]);
    printf("\t graphFile: \t %s\n", datasetName);
    printf("\t k: \t\t %d\n", k);
    printf("\t #threads: \t %d\n", numberOfActiveThreads);
    printf("\t blockSize: \t %d\n", blockSize);
    printf("\t donationsPerWarp: \t %d\n", jobsPerWarp);
    printf("\t threshold: \t %d\n", globalThreshold);
    printf("\t report: \t %s\n", rep ? "Yes" : "No");

    Timer timerIO;
    Timer timerLB;

    timerIO.play();
    DuMatoCPU *DM_CPU = new DuMatoCPU(datasetName, k, numberOfActiveThreads, blockSize, numberOfSMs, jobsPerWarp, motifs, globalThreshold, relabeling, patternAware);
    cudaDeviceSynchronize();
    timerIO.pause();

    Report* report;
    if(rep)
        report = new Report(DM_CPU, 100);
    
    Timer timerGPU;

    if(globalThreshold >= 100) {
        timerGPU.play("Kernel");
        DM_CPU->runKernel();
        if(rep)
            report->start();
        DM_CPU->waitKernel();
        timerGPU.pause("Kernel");
        if(rep)
            report->stop();
    }
    else {
        timerGPU.play("Kernel");
        DM_CPU->runKernel();
        if(rep)
            report->start();
        while(true) {
            if(DM_CPU->gpuIsIdle()) {
                printf("[gpuIsIdle] %.2f.\n", DM_CPU->dataCPU->h_percentageWarpsIdle);
                timerLB.play();
                DM_CPU->stopKernel();
                if(DM_CPU->rebalance()) {
                    timerLB.pause();
                    DM_CPU->runKernel();
                }
                else {
                    timerLB.pause();
                    break;
                }
            }
            DM_CPU->sleepFor(100);
        } 
        DM_CPU->waitKernel();
        if(rep)
            report->stop();
        timerGPU.pause("Kernel");
    }

    DM_CPU->outputAggregatePattern();
    // DM_CPU->outputAggregateCounter();
    timerTOTAL.pause();

    if(validateResults) { 
        DM_CPU->validateAggregatePattern();
    }
    printf("%.2f seconds.\n", timerGPU.getElapsedTimeInSeconds());
    // printf("Time (GPU): %f secs.\n", timerGPU.getElapsedTimeInSeconds());
    // printf("Time (LB): %f secs.\n", timerLB.getElapsedTimeInSeconds());
    // printf("Time (IO): %f secs\n", timerIO.getElapsedTimeInSeconds());
    // printf("Time (TOTAL): %f secs\n", timerTOTAL.getElapsedTimeInSeconds());
    /*************************************************/
    /************** Memory release *******************/
    delete DM_CPU;
    if(rep)
        delete report;
    /*************************************************/

    return 0;
}
