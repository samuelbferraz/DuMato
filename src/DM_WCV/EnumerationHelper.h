#include "Structs.cu"

#ifndef ENUMERATION_HELPER_H
#define ENUMERATION_HELPER_H

class EnumerationHelper {
	private:
        int h_warpId;
        int h_k;
        int h_warpSize;
        int h_extensionsLength;
        int h_jobsPerWarp;

        int *h_id;
		int *h_extensions;
        int *h_extensionsOffset;
        int *h_numberOfExtensions;
        int *h_currentPos;
        int *h_currentPosOfJob;
        long unsigned int *h_localSubgraphInduction;
        int *h_inductions;
        int *h_validJobs;
        int *h_currentJob;
        int *h_jobs;
        unsigned long *h_result;

        // Offsets
        int offsetId;
        int offsetExtensions;
        int offsetNumberOfExtensions;
        int offsetLocalSubgraphInduction;

	public:
		EnumerationHelper(int h_warpId, int h_k, int h_warpSize, int h_jobsPerWarp, int h_extensionsLength, int *h_id, int *h_extensions, int *h_extensionsOffset, int *h_numberOfExtensions, int *h_currentPos, int *h_currentPosOfJob, long unsigned int *h_localSubgraphInduction, int *h_inductions, int *h_validJobs, int *h_currentJob, int *h_jobs, unsigned long *h_result);
        EnumerationHelper(int h_warpId, DataCPU *dataCPU);
        int getWarpId();
        void setId(int k, int value);
        int getId(int k);
        void setLocalSubgraphInduction(int k, long unsigned int value);
        long unsigned int getLocalSubgraphInduction(int k);
        void setNumberOfExtensions(int k, int value);
        int getNumberOfExtensions(int k);
        void setExtension(int level, int pos, int value);
        int getExtension(int level, int pos);
        int popLastExtension(int level);
        void setCurrentPos(int value);
        int getCurrentPos();
        void report();
        int getTargetLevel();
        int getWeight();
        void setValidJobs(int amount);
        int getValidJobs();
        void setCurrentJob(int pos);
        int getCurrentJob();
        void increaseJob();
        void setJob(int job, int k, int value);
        int getJob(int job, int k);
        void setLocalSubgraphInduction(int job, int k, int value);
        void setCurrentPosOfJob(int job, int k);
        void setInductions(int job, int k, int value);
        long unsigned int getInductions(int job, int k);
        int getCurrentPosOfJob(int job);
        unsigned long getResult();
        bool jobQueueIsEmpty();

        bool isDonator();
		~EnumerationHelper();
};

#endif