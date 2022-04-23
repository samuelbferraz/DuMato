#ifndef QUICKMAP_H
#define QUICKMAP_H

#include <stdio.h>

class QuickMapping {
    public:
        long unsigned int quick, cgLocal, cgGlobal;
        long unsigned int numberOfQuicks;
        long unsigned int *quickToCgLocal;
        long unsigned int numberOfCgs;
        long unsigned int *cgLocalToGlobal;
        bool canonical_relabeling;

        FILE* fp;

        QuickMapping(int k, bool canonical_relabeling);
        ~QuickMapping();
};



#endif
