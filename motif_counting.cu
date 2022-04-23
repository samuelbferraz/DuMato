#include "Manager.h"
#include "DuMato.h"

using namespace DuMato;

__global__ void motif_counting(Device*);

int main(int argc, const char** argv)
{
    const char* graphFile = argv[1];
    int k = atoi(argv[2]);
    int numberOfActiveThreads = atoi(argv[3]);
    int blockSize = atoi(argv[4]);
    int numberOfSMs = atoi(argv[5]);
    int globalThreshold = atoi(argv[6]);
    int reportInterval = atoi(argv[7]);

    Manager* manager = new Manager(graphFile, k, numberOfActiveThreads, blockSize, motif_counting, false, 1, numberOfSMs, reportInterval);

    manager->startTimer();

    //Load balancing
    manager->runKernel();
    while(true) {
        manager->loadGpuThreadStatus();
        if(manager->gpuIsIdle(globalThreshold)) {
            manager->stopKernel();
            if(manager->rebalance()) {
                manager->runKernel();
                manager->loadGpuThreadStatus();
                manager->shortIdlenessReport();
            }
            else
                break;
        }
    }

    manager->copyResult();
    manager->stopTimer();
    manager->printResult();
    printf("Execution runtime (CPU+GPU): %f\n", manager->getRuntimeInSeconds());

    delete(manager);

    return 0;
}


__global__ void motif_counting(Device* device) {

    dm_start(device);

    while(dm_active() && dm_gpuIsBalanced()) { // Control
        if(dm_numberOfExtensions() == -1) {
            // Extend
            dm_extend(0, dm_k());

            // Filter
            dm_filterExtensionsLowerThan(dm_id(0));
            if(dm_k() > 0) {
                dm_compact(); // Compact

                // Canonical filtering
                for(int i = 1, target ; i <= dm_k() ; i++) {
                    target = dm_id(i);
                    if(i != 0) {
                        dm_filterExtensionsLowerOrEqualThan(0, dm_extensionSourcesOffsetK(i), target);
                    }
                    if(i != dm_k()) {
                        dm_filterExtensionsEqual(dm_extensionSourcesOffsetK(i+1), dm_numberOfExtensions(), target);
                    }
                }
            }
            dm_compact(); // Compact
        }

        // Move
        if(dm_numberOfExtensions() != 0) {
            if(dm_k() == dm_globalK()-2) {
                dm_agregate_pattern(); // Aggregate

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
