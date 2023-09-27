#include "DuMatoCPU.h"
#include <thread>
#include <fstream>

#ifndef REPORT_H
#define REPORT_H

class Report {
    public:
        DuMatoCPU* DM_CPU;
        std::thread *reportThread;
        bool activeReport;
        int reportInterval;
        std::ofstream *output;

        
        Report(DuMatoCPU* DM_CPU, int reportInterval);
        void start();
        void stop();
        ~Report();

        static void* reportFunction(DuMatoCPU *manager, bool *activeReport, int reportInterval, std::ofstream *output);
};



#endif