#include "Manager.h"
#include "DuMato.h"

using namespace DuMato;

__global__ void clique(Device*);

int main(int argc, const char** argv)
{
    printf("Usage: ./clique_DM_DFS graphFile k threads blockSize\n");
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

    Manager* manager = new Manager(graphFile, k, numberOfActiveThreads, blockSize, clique);


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


__global__ void clique(Device* device) {

    dm_start(device);

    while(dm_active()) {
        // Reached desired length of valid subgraph: count++.
        if(dm_k() >= dm_globalK()-1) {
            dm_validSubgraphs();
            dm_backward();
        }
        // There are no neighbours to be traversed in current enumeration level
        else if(dm_traversedExtensions() >= dm_numberOfExtensions()) {
            dm_backward();
        }
        // There are neighbours to be traversed in current enumeration level
        else {
            // Canonical checking
            int extension = dm_generateExtension();

            if(extension > dm_id(dm_k())) {
                bool isClique = true;
                for(int i = 0 ; i < dm_k() && isClique ; i++)
                    isClique = dm_findNeighbourhood2(dm_id(i), extension);

                if(isClique)
                    dm_forward(extension);
            }

        }
    }

    dm_end();
}
