#include "Manager.h"
#include "DuMato.h"

using namespace DuMato;

__global__ void motifs(Device*);

int main(int argc, const char** argv)
{
    printf("Usage: ./motifs_DM_DFS graphFile k threads blockSize\n");
    printf("\t graphFile: \t url of graph dataset\n");
    printf("\t k: \t\t clique size\n");
    printf("\t threads: \t amount of GPU threads (recommended: 102400)\n");
    printf("\t blockSize: \t amount of threads per block (recommended: 256)\n");
    if(argc != 5) {
        printf("\nWrong amount of parameters!\n");
        printf("Exiting...\n");
        exit(1);
    }

    const char* graphFile = argv[1];
    int k = atoi(argv[2]);
    int numberOfActiveThreads = atoi(argv[3]);
    int blockSize = atoi(argv[4]);
    int numberOfSMs = 80;
    int canonicalize = 0;
    int numberOfWorkerThreads = 1;

    Manager* manager = new Manager(graphFile, k, numberOfActiveThreads, blockSize, motifs, (canonicalize == 1 ? true : false), numberOfWorkerThreads, numberOfSMs, 1000);

    manager->startTimer();
    manager->runKernel();
    manager->waitKernel();
    manager->stopTimer();
    manager->copyResult();

    manager->printResult();
    printf("%f\n", manager->getRuntimeInSeconds());

    delete(manager);

    return 0;
}


__global__ void motifs(Device* device) {

    dm_start(device);

    while(dm_active() && dm_gpuIsBalanced()) {
        if(dm_numberOfExtensions() == -1) {
            dm_generateUniqueExtensions(0, dm_k());

            // if(dm_k() == 1 && dm_id(0) == 20 && dm_id(1) == 661) {
            //     printf("[GENERATE_UNIQUE]*** %d ***\n", dm_numberOfExtensions());
            //     for(int i = 0 ; i < dm_numberOfExtensions() ; i++) {
            //         printf("** %d **\n", dm_getExtension(i));
            //     }
            // }

            dm_filterExtensionsLowerThan(dm_id(0));

            // if(dm_k() == 1 && dm_id(0) == 20 && dm_id(1) == 661) {
            //     printf("[FILTER_LOWER_THAN]*** %d ***\n", dm_numberOfExtensions());
            //     for(int i = 0 ; i < dm_numberOfExtensions() ; i++) {
            //         printf("** %d **\n", dm_getExtension(i));
            //     }
            // }

            if(dm_k() > 0) {
                dm_compactExtensions();

                // if(dm_k() == 1 && dm_id(0) == 20 && dm_id(1) == 661) {
                //     printf("[COMPACT 1]*** %d ***\n", dm_numberOfExtensions());
                //     for(int i = 0 ; i < dm_numberOfExtensions() ; i++) {
                //         printf("** %d **\n", dm_getExtension(i));
                //     }
                // }

                // Canonical filtering
                for(int i = 1, target ; i <= dm_k() ; i++) {
                    target = dm_id(i);
                    if(i != 0) {
                        dm_filterExtensionsLowerOrEqualThan(0, dm_extensionSourcesOffsetK(i), target);

                        // if(dm_k() == 1 && dm_id(0) == 20 && dm_id(1) == 661) {
                        //     printf("[FILTER_LOWER_EQUAL]*** %d,%d,%d ***\n", dm_numberOfExtensions(), dm_extensionSourcesOffsetK(i), target);
                        //     for(int i = 0 ; i < dm_numberOfExtensions() ; i++) {
                        //         printf("** %d **\n", dm_getExtension(i));
                        //     }
                        // }
                    }
                    if(i != dm_k()) {
                        dm_filterExtensionsEqual(dm_extensionSourcesOffsetK(i+1), dm_numberOfExtensions(), target);

                        // if(dm_k() == 1 && dm_id(0) == 20 && dm_id(1) == 661) {
                        //     printf("[FILTER_EQUAL]*** %d ***\n", dm_numberOfExtensions());
                        //     for(int i = 0 ; i < dm_numberOfExtensions() ; i++) {
                        //         printf("** %d **\n", dm_getExtension(i));
                        //     }
                        // }
                    }
                }

                // Induction
                // bool found;
                // int localOffsetInduction = (((dm_k()-1)*(2+dm_k()))/2);
                // long unsigned int previousQuickPattern = localSubgraphInduction[dm_offsetCurrentInduction()+dm_k()-1], quickPattern;
                // for(int warpPosition = dm_getLane(), extension ; warpPosition < roundToWarpSize(dm_numberOfExtensions()) ; warpPosition+=32) {
                //     quickPattern = previousQuickPattern;
                //     if(warpPosition < dm_numberOfExtensions()) {
                //         extension = dm_getExtension(warpPosition);
                //
                //         if(extension != -1) {
                //             for(int j = 0, currentPow = powf(2,localOffsetInduction) ; j < dm_k() ; j++, currentPow*=2) {
                //                 found = dm_findNeighbourhood2(dm_id(j), extension) ? 1 : 0;
                //                 quickPattern += (found*currentPow);
                //             }
                //             dm_setExtensionQuick(warpPosition, quickPattern);
                //         }
                //     }
                //     __syncwarp();
                // }
            }
            dm_compactExtensions();
            // if(dm_k() == 1 && dm_id(0) == 20 && dm_id(1) == 661) {
            //     printf("[COMPACT 2]*** %d ***\n", dm_numberOfExtensions());
            //     for(int i = 0 ; i < dm_numberOfExtensions() ; i++) {
            //         printf("** %d **\n", dm_getExtension(i));
            //     }
            // }
        }

        if(dm_numberOfExtensions() != 0) {
            if(dm_k() == dm_globalK()-2) {
                dm_accumulateValidSubgraphs();
                dm_agregateValidSubgraphs();
                dm_backward();
            }
            else {
                dm_forward();
                // if com backward
            }
        }
        else
            dm_backward();
    }

    dm_end();
}
