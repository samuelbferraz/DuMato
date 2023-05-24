#include "Manager.h"
#include "DuMato.h"

using namespace DuMato;

__global__ void motifs(Device*);

int main(int argc, const char** argv)
{
    const char* graphFile = argv[1];
    int k = atoi(argv[2]);
    int numberOfActiveThreads = atoi(argv[3]);
    int blockSize = atoi(argv[4]);
    int numberOfSMs = atoi(argv[5]);
    int globalThreshold = atoi(argv[6]);
    int jobsPerWarp = atoi(argv[7]);
    int checkFlag = atoi(argv[8]);
    int induce = atoi(argv[9]);
    int balancingInterval = 200;

    Manager* manager = new Manager(graphFile, k, numberOfActiveThreads, blockSize, motifs, numberOfSMs, 1000, jobsPerWarp, induce);

    manager->startTimer();

    manager->runKernel();
    while(true) {
        manager->loadGpuThreadStatus();
        if(manager->gpuIsIdle(globalThreshold)) {
            manager->stopKernel();
            manager->loadGpuThreadStatus();
            manager->shortIdlenessReport("[REBALANCING]");
            if(manager->rebalance()) {
                manager->runKernel();
                manager->sleepFor(balancingInterval);
            }
            else
                break;
        }
    }
    manager->waitKernel();
    manager->copyResult();
    manager->stopTimer();
    manager->printResult();
    manager->check(checkFlag);

    printf("%f\n", manager->getRuntimeInSeconds());

    delete(manager);

    return 0;
}


__global__ void motifs(Device* device) {

    dm_start(device);

    while(dm_active() && dm_gpuIsBalanced()) {
        if(dm_numberOfExtensions() == -1) {
            dm_generateUniqueExtensions(0, dm_k());
            if(dm_k() > 0)
                dm_canonicalFilter();
        }

        if(dm_numberOfExtensions() != 0) {
            if(dm_k() == dm_globalK()-2) {
                dm_accumulateValidSubgraphs();
                dm_agregateValidSubgraphs();
                dm_backward();
            }
            else {
                dm_forward();
            }
        }
        else
            dm_backward();
    }

    dm_end();
}
