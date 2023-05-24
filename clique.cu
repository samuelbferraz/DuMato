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
    int jobsPerWarp = atoi(argv[7]);
    int checkFlag = atoi(argv[8]);
    int balancingInterval = 200;

    Manager* manager = new Manager(graphFile, k, numberOfActiveThreads, blockSize, clique, numberOfSMs, 100, jobsPerWarp, 0);

    manager->startTimer();

    manager->runKernel();
    while(true) {
        manager->loadGpuThreadStatus();
        if(manager->gpuIsIdle(globalThreshold)) {
            manager->stopKernel();
            manager->loadGpuThreadStatus();
            // manager->shortIdlenessReport("[REBALANCING]");
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
    manager->printCount();
    manager->check(checkFlag);

    printf("%f\n", manager->getRuntimeInSeconds());

    delete(manager);

    return 0;
}


__global__ void clique(Device* device) {

    dm_start(device);

    while(dm_active() && dm_gpuIsBalanced()) {
        if(dm_numberOfExtensions() == -1) {
            dm_generateExtensionsSingleSource(dm_k());

            if(dm_k() > 0) {
                bool found;
                unsigned long pos, actives, totalActives, activesOnMyRight, idlesOnMyRight;
                int localOffsetExtensions = dm_offsetExtensions() + extensionsOffset[dm_k()], currentOffsetExtensions;
                currentOffsetExtensions = 0;

                for(int warpPosition = dm_getLane(), ext; warpPosition < roundToWarpSize(dm_numberOfExtensions()) ; warpPosition+=32) {
                    found = false;
                    if(warpPosition < dm_numberOfExtensions()) {
                        found = true;
                        ext = dm_getExtension(warpPosition);
                        for(int j = 0 ; j < dm_k() && found ; j++)
                            found = dm_findNeighbourhood2(dm_id(j), ext);
                    }
                    __syncwarp();
                    ext = found ? ext : -1;

                    actives = __ballot_sync(0xffffffff, ext == -1 ? 0 : 1);
                    totalActives = __popc(actives);
                    actives = (actives << ((unsigned int)32-(unsigned int)dm_getLane()));
                    activesOnMyRight = __popc(actives);
                    idlesOnMyRight = dm_getLane() - activesOnMyRight;
                    pos = ext != -1 ? activesOnMyRight : totalActives + idlesOnMyRight;

                    extensions[localOffsetExtensions+currentOffsetExtensions+pos] = ext;
                    currentOffsetExtensions += totalActives;
                }
                numberOfExtensions[dm_offsetWarp()+dm_k()] = currentOffsetExtensions;
            }
        }

        if(dm_numberOfExtensions() != 0) {
            if(dm_k() == dm_globalK()-2) {
                dm_accumulateValidSubgraphs();
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
