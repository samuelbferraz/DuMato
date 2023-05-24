#include "QuickMapping.h"
#include <stdlib.h>
#include <string>

QuickMapping::QuickMapping(int k) {
    if(k == -1) {
        quick = cgLocal = cgGlobal = numberOfQuicks = numberOfCgs = 1;
        quickToCgLocal = new long unsigned int[numberOfQuicks];
        cgLocalToGlobal = new long unsigned int[numberOfCgs];
        return;
    }

    printf("[QuickMapping] Reading quick mappings...\n");
    std::string filename = "./dictionaries/" + std::to_string(k) + ".csv";

    FILE* fp = fopen(filename.c_str(), "r");
    char line[100];

    if(fp == NULL) {
        printf("Generate file %s first... aborting\n", filename.c_str());
        exit(1);
    }

    fscanf(fp,"%s\n", line); // Header
    fscanf(fp,"%lu", &numberOfQuicks);
    quickToCgLocal = new long unsigned int[numberOfQuicks];

    for(int i = 0 ; i < numberOfQuicks ; i++) {
        fscanf(fp,"%lu,%lu", &quick, &cgLocal);
        quickToCgLocal[quick] = cgLocal;
    }

    fscanf(fp,"%s\n", line); // Header
    fscanf(fp,"%lu", &numberOfCgs);
    cgLocalToGlobal = new long unsigned int[numberOfCgs];

    for(int i = 0 ; i < numberOfCgs ; i++) {
        fscanf(fp,"%lu,%lu", &cgLocal, &cgGlobal);
        cgLocalToGlobal[cgLocal] = cgGlobal;
    }

    printf("[QuickMapping] Finished!\n");

    fclose(fp);
}

QuickMapping::~QuickMapping() {
    delete[] quickToCgLocal;
    delete[] cgLocalToGlobal;
}
