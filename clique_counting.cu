#include "Manager.h"
#include "DuMato.h"

using namespace DuMato;

__global__ void clique(Device*);

int main(int argc, const char** argv)
{
    const char* graphFile = argv[1];
    int k = atoi(argv[2]);
    int numberOfActiveThreads = atoi(argv[3]);
    int blockSize = atoi(argv[4]);
    int numberOfSMs = atoi(argv[5]);
    int globalThreshold = atoi(argv[6]);
    int reportInterval = atoi(argv[7]);

    Manager* manager = new Manager(graphFile, k, numberOfActiveThreads, blockSize, clique, false, 1, numberOfSMs, reportInterval);

    manager->startTimer();
    manager->runKernel();

    // Load balancing
    while(true) {
        manager->loadGpuThreadStatus();
        if(manager->gpuIsIdle(globalThreshold)) {
            manager->shortIdlenessReport();
            manager->stopKernel();
            if(manager->rebalance()) {
                manager->runKernel();
            }
            else
                break;
        }
    }

    manager->copyResult();
    manager->stopTimer();
    manager->printCount();
    printf("Execution runtime (CPU+GPU): %f\n", manager->getRuntimeInSeconds());

    delete(manager);

    return 0;
}


__global__ void clique(Device* device) {

    dm_start(device);

    while(dm_active() && dm_gpuIsBalanced()) { // Control

        if(dm_numberOfExtensions() == -1) {

            dm_extend_all(dm_k(), dm_k()); // Extend

            dm_filterExtensionsLowerThan(dm_id(dm_k())); // Filter

            dm_compact(); // Compact

            if(dm_k() > 0) {
                // Filter (clique)
                bool found;
                for(int warpPosition = dm_getLane(), extension ; warpPosition < roundToWarpSize(dm_numberOfExtensions()) ; warpPosition+=32) {
                    found = true;
                    if(warpPosition < dm_numberOfExtensions()) {
                        extension = dm_getExtension(warpPosition);
                        if(extension != -1)
                            for(int j = 0 ; j < dm_k() && found ; j++)
                                found = dm_findNeighbourhood2(dm_id(j), extension);
                    }
                    __syncwarp();
                    if(!found)
                        dm_invalidateExtension(warpPosition);
                    __syncwarp();
                }
            }
            dm_compact(); // Compact
        }

        // Move
        if(dm_numberOfExtensions() != 0) {
            if(dm_k() == dm_globalK()-2) {
                dm_aggregate_count(); // Aggregate

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
