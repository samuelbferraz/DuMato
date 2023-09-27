#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    char* filename = argv[1];

    FILE* fp = fopen(filename, "r");

    if(fp == NULL) {
        printf("Quick mappings doesn't exist! Generate them first...\n");
        exit(1);
    }

    unsigned int quick, cgLocal, cgGlobal;
    unsigned int numberOfQuicks;
    unsigned int *quickToCgLocal;
    unsigned int numberOfCgs;
    unsigned int *cgLocalToGlobal;

    fscanf(fp,"%u", &numberOfQuicks);
    quickToCgLocal = new unsigned int[numberOfQuicks];

    for(int i = 0 ; i < numberOfQuicks ; i++) {
        fscanf(fp,"%u,%u", &quick, &cgLocal);
        quickToCgLocal[quick] = cgLocal;
    }

    fscanf(fp,"%u", &numberOfCgs);
    cgLocalToGlobal = new unsigned int[numberOfCgs];

    for(int i = 0 ; i < numberOfCgs ; i++) {
        fscanf(fp,"%u,%u", &cgLocal, &cgGlobal);
        cgLocalToGlobal[cgLocal] = cgGlobal;
    }

    // while(fscanf(fp,"%u,%u", &g, &cg) != EOF) {
        // printf("%u,%u\n", g, cg);
        // map[g] = cg;
    // }

    printf("%u\n", numberOfQuicks);
    for(int i = 0 ; i < numberOfQuicks ; i++) {
        printf("%u,%u\n", i, quickToCgLocal[i]);
    }

    printf("%u\n", numberOfCgs);
    for(int i = 0 ; i < numberOfCgs ; i++) {
        printf("%u,%u\n", i, cgLocalToGlobal[i]);
    }


    // for(auto it = map.begin() ; it != map.end() ; it++)
    //     printf("%u,%u\n", it->first, it->second);

    // printf("%u,%u\n", g, cg);

    fclose(fp);

    delete[] quickToCgLocal;
    delete[] cgLocalToGlobal;

    return 0;
}
