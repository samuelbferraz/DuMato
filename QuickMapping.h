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

        FILE* fp;

        QuickMapping(int k);
        ~QuickMapping();
};



#endif
