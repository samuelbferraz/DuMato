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

    int *status, *smid;
    cudaStream_t stream;
    long unsigned instant = 0;
    int numberOfPhysicalWarps = DM_CPU->dataCPU->h_numberOfActiveThreads/32;
    int *physicalWarps = new int[DM_CPU->dataCPU->h_numberOfActiveThreads/32];
    int *warpsPerSM = new int[DM_CPU->dataCPU->h_numberOfSMs+1];

    gpuErrorCheck(cudaStreamCreate(&stream));
    gpuErrorCheck(cudaMallocHost((void**)&status, DM_CPU->dataCPU->h_numberOfWarps * sizeof(int)));
    gpuErrorCheck(cudaMallocHost((void**)&smid, DM_CPU->dataCPU->h_numberOfWarps * sizeof(int)));
    
    while(*activeReport) {
        memset(physicalWarps, 0, (DM_CPU->dataCPU->h_numberOfActiveThreads/32)*sizeof(int));
        memset(warpsPerSM, 0, (DM_CPU->dataCPU->h_numberOfSMs+1) * sizeof(int));
        gpuErrorCheck(cudaMemcpyAsync(status, DM_CPU->dataGPU->d_status, DM_CPU->dataCPU->h_numberOfWarps * sizeof(int), cudaMemcpyDeviceToHost, stream));
        gpuErrorCheck(cudaMemcpyAsync(smid, DM_CPU->dataGPU->d_smid, DM_CPU->dataCPU->h_numberOfWarps * sizeof(int), cudaMemcpyDeviceToHost, stream));
        gpuErrorCheck(cudaStreamSynchronize(stream));

        // Idleness
        int amountVirtualWarpsIdle = 0;
        float percentageVirtualWarpsIdle = 0;
        float percentageVirtualWarpsActive = 0;
        
        for(int i = 0, warpId ; i < DM_CPU->dataCPU->h_numberOfWarps ; i++) {
            warpId = (i * DM_CPU->dataCPU->h_warpSize) / 32;
            if(status[i] == 2) 
                amountVirtualWarpsIdle++;
            else if(status[i] == 1) {
                physicalWarps[i] = 1;
                warpsPerSM[smid[i]]++;
            }
        }
        percentageVirtualWarpsIdle = (amountVirtualWarpsIdle/(double)DM_CPU->dataCPU->h_numberOfWarps)*100;
        percentageVirtualWarpsActive = 100 - percentageVirtualWarpsIdle;

        int amountPhysicalWarpsActive = 0;
        float percentagePhysicalWarpsActive = 0;
        for(int i = 0 ; i < numberOfPhysicalWarps ; i++) {
            if(physicalWarps[i] == 1) 
                amountPhysicalWarpsActive++;
        }
        percentagePhysicalWarpsActive = ((float)amountPhysicalWarpsActive / (float)numberOfPhysicalWarps)*100;
        
        std::cout << "REPORT," << instant << ",active warps," << amountPhysicalWarpsActive << ",";
        // std::cout << "[REPORT][" << instant << "]active warps:" << percentagePhysicalWarpsActive <<"%,active warps:" << amountPhysicalWarpsActive << ",avg warps per sm:" << (float)amountPhysicalWarpsActive/80 << ",SMs:";
        // for(int i = 0 ; i < DM_CPU->dataCPU->h_numberOfSMs ; i++) {
        //    std::cout << warpsPerSM[i] << ",";
        // }
        std::cout << "\n";
        // std::cout << "[REPORT][" << instant << "]active warps:" << percentagePhysicalWarpsActive << "%,avg warps per sm:" << amountPhysicalWarpsActive/80 << "\n";

        std::this_thread::sleep_for(std::chrono::milliseconds(reportInterval));

        instant+=reportInterval;
    }

    cudaFreeHost(status);
    cudaFreeHost(smid);
    delete []physicalWarps;
    delete []warpsPerSM;

    return NULL;
}

Report::~Report() {
    reportThread->join();
    output->close();
    delete output;
}