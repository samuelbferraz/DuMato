#include "Report.h"
#include <cuda_runtime.h>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cstring>

Report::Report(DuMatoCPU* DM_CPU, int reportInterval) {
    this->DM_CPU = DM_CPU;
    this->activeReport = true;
    this->reportInterval = reportInterval;
    this->output = new std::ofstream("report.csv", std::ios::out);
}

void Report::start() {
    reportThread = new std::thread(reportFunction, DM_CPU, &activeReport, reportInterval, output);
}

void Report::stop() {
    activeReport = false;
}

void* Report::reportFunction(DuMatoCPU* DM_CPU, bool* activeReport, int reportInterval, std::ofstream *output) { 
    std::cout << std::fixed;
    std::cout << std::setprecision(2);

    int *status;
    cudaStream_t stream;
    long unsigned instant = 0;

    int virtualWarpsPerWarp = 32 / DM_CPU->dataCPU->h_warpSize;
    int *virtualWarpActivity = new int[virtualWarpsPerWarp];
    int numberOfPhysicalWarps = DM_CPU->dataCPU->h_numberOfActiveThreads/32;
    int *physicalWarps = new int[numberOfPhysicalWarps];

    gpuErrorCheck(cudaStreamCreate(&stream));
    gpuErrorCheck(cudaMallocHost((void**)&status, DM_CPU->dataCPU->h_numberOfWarps * sizeof(int)));
    
    while(*activeReport) {
        memset(physicalWarps, 0, numberOfPhysicalWarps*sizeof(int));
        memset(virtualWarpActivity, 0, virtualWarpsPerWarp*sizeof(int));
        gpuErrorCheck(cudaMemcpyAsync(status, DM_CPU->dataGPU->d_status, DM_CPU->dataCPU->h_numberOfWarps * sizeof(int), cudaMemcpyDeviceToHost, stream));
        gpuErrorCheck(cudaStreamSynchronize(stream));

        // Idleness
        int amountVirtualWarpsIdle = 0, amountVirtualWarpsActive = 0;
        float percentageVirtualWarpsIdle = 0;
        float percentageVirtualWarpsActive = 0;
        
        for(int i = 0, warpId ; i < DM_CPU->dataCPU->h_numberOfWarps ; i++) {
            warpId = (i * DM_CPU->dataCPU->h_warpSize) / 32;
            if(status[i] == 2)
                amountVirtualWarpsIdle++;
            else if(status[i] == 1)
                physicalWarps[warpId] = 1;
        }
        amountVirtualWarpsActive = DM_CPU->dataCPU->h_numberOfWarps-amountVirtualWarpsIdle;
        percentageVirtualWarpsIdle = (amountVirtualWarpsIdle/(double)DM_CPU->dataCPU->h_numberOfWarps)*100;
        percentageVirtualWarpsActive = 100 - percentageVirtualWarpsIdle;

        int amountPhysicalWarpsActive = 0;
        float percentagePhysicalWarpsActive = 0;
        for(int i = 0 ; i < numberOfPhysicalWarps ; i++) {
            if(physicalWarps[i] == 1) 
                amountPhysicalWarpsActive++;
        }
        percentagePhysicalWarpsActive = ((float)amountPhysicalWarpsActive / (float)numberOfPhysicalWarps)*100;


        for(int i = 0 ; i < numberOfPhysicalWarps ; i++) {
            int amountActiveVirtualWarps = 0;
            for(int j = 0 ; j < virtualWarpsPerWarp ; j++) {
                if(status[i*virtualWarpsPerWarp+j] == 1)
                    amountActiveVirtualWarps++;
            }
            if(amountActiveVirtualWarps >= 1 && amountActiveVirtualWarps <= virtualWarpsPerWarp)
                virtualWarpActivity[amountActiveVirtualWarps-1]++;
        }
        
        // (*output) << "[REPORT][" << instant << "]active virtual warps:" << percentageVirtualWarpsActive << "%,active physical warps:" << percentagePhysicalWarpsActive << "%\n";
        std::cout << "[REPORT][" << instant << "]%active virtual warps:" << percentageVirtualWarpsActive << ",%active physical warps:" << percentagePhysicalWarpsActive << ",#active virtual warps:" << amountVirtualWarpsActive << ",#active physical warps:" << amountPhysicalWarpsActive;

        for(int i = 0 ; i < virtualWarpsPerWarp ; i++)
            std::cout << "," << i+1 << ":" << virtualWarpActivity[i]; 
        std::cout << "\n";

        std::this_thread::sleep_for(std::chrono::milliseconds(reportInterval));

        instant+=reportInterval;
    }

    cudaFreeHost(status);
    delete[] physicalWarps;
    delete[] virtualWarpActivity;

    return 0;
}

Report::~Report() {
    reportThread->join();
    output->close();
    delete output;
}