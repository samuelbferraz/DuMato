#include "EnumerationHelper.h"
#include <stdio.h>

EnumerationHelper::EnumerationHelper(int h_warpId, int h_k, int h_warpSize, int h_jobsPerWarp, int h_extensionsLength, int *h_id, int *h_extensions, int *h_extensionsOffset, int *h_numberOfExtensions, int *h_currentPos, int *h_currentPosOfJob, long unsigned int *h_localSubgraphInduction, int *h_inductions, int *h_validJobs, int *h_currentJob, int *h_jobs, unsigned long *h_result) {
    this->h_warpId = h_warpId;
    this->h_k = h_k;
    this->h_warpSize = h_warpSize;
    this->h_jobsPerWarp = h_jobsPerWarp;
    this->h_extensionsLength = h_extensionsLength;
    this->h_id = h_id;
    this->h_extensions = h_extensions;
    this->h_extensionsOffset = h_extensionsOffset;
    this->h_numberOfExtensions = h_numberOfExtensions;
    this->h_currentPos = h_currentPos;        
    this->h_currentPosOfJob = h_currentPosOfJob;
    this->h_localSubgraphInduction = h_localSubgraphInduction;
    this->h_inductions = h_inductions;
    this->h_validJobs = h_validJobs;
    this->h_currentJob = h_currentJob;
    this->h_jobs = h_jobs;
    this->h_result = h_result;

    offsetId = h_warpId*h_warpSize;
    offsetNumberOfExtensions = h_warpId*h_warpSize;
    offsetLocalSubgraphInduction = h_warpId*h_warpSize;
    offsetExtensions = h_warpId * h_extensionsLength;
}

EnumerationHelper::EnumerationHelper(int h_warpId, DataCPU *dataCPU) {
    this->h_warpId = h_warpId;
    this->h_k = dataCPU->h_k;
    this->h_warpSize = dataCPU->h_warpSize;
    this->h_jobsPerWarp = dataCPU->h_theoreticalJobsPerWarp;
    this->h_extensionsLength = dataCPU->h_extensionsLength;
    this->h_id = dataCPU->h_id;
    this->h_extensions = dataCPU->h_extensions;
    this->h_extensionsOffset = dataCPU->h_extensionsOffset;
    this->h_numberOfExtensions = dataCPU->h_numberOfExtensions;
    this->h_currentPos = dataCPU->h_currentPos;        
    this->h_currentPosOfJob = dataCPU->h_currentPosOfJob;
    this->h_localSubgraphInduction = dataCPU->h_localSubgraphInduction;
    this->h_inductions = dataCPU->h_inductions;
    this->h_validJobs = dataCPU->h_validJobs;
    this->h_currentJob = dataCPU->h_currentJob;
    this->h_jobs = dataCPU->h_jobs;
    this->h_result = dataCPU->h_result;

    offsetId = h_warpId*dataCPU->h_warpSize;
    offsetNumberOfExtensions = h_warpId*dataCPU->h_warpSize;
    offsetLocalSubgraphInduction = h_warpId*dataCPU->h_warpSize;
    offsetExtensions = h_warpId * dataCPU->h_extensionsLength;
}

int EnumerationHelper::getWarpId() {
    return h_warpId;
}

void EnumerationHelper::setId(int k, int value) {
    h_id[offsetId+k] = value;
}

int EnumerationHelper::getId(int k) {
    return h_id[offsetId+k];
}

void EnumerationHelper::setLocalSubgraphInduction(int k, long unsigned int value) {
    h_localSubgraphInduction[offsetLocalSubgraphInduction+k] = value;
}

long unsigned int EnumerationHelper::getLocalSubgraphInduction(int k) {
    return h_localSubgraphInduction[offsetLocalSubgraphInduction+k];
}

void EnumerationHelper::setNumberOfExtensions(int k, int value) {
    h_numberOfExtensions[offsetNumberOfExtensions+k] = value;
}

int EnumerationHelper::getNumberOfExtensions(int k) {
    return h_numberOfExtensions[offsetNumberOfExtensions+k];
}

void EnumerationHelper::setExtension(int level, int pos, int value) {
    h_extensions[offsetExtensions + h_extensionsOffset[level] + pos] = value;
}

int EnumerationHelper::getExtension(int level, int pos) {
    return h_extensions[offsetExtensions + h_extensionsOffset[level] + pos];
}

int EnumerationHelper::popLastExtension(int level) {
    int numberOfExtensions = getNumberOfExtensions(level);
    int extension = getExtension(level, numberOfExtensions-1);
    setNumberOfExtensions(level, numberOfExtensions-1);
    return extension;
}

void EnumerationHelper::setCurrentPos(int value) {
    h_currentPos[h_warpId] = value;
}

int EnumerationHelper::getCurrentPos() {
    return h_currentPos[h_warpId];
}

void EnumerationHelper::report() {
    int count = 0;
    printf("wid: %d, currentPos: %d, targetLevel: %d, weight: %d, result:%lu\n", h_warpId, getCurrentPos(), getTargetLevel(), getWeight(), getResult());
    for(int i = 0 ; i <= getCurrentPos() ; i++) {
        printf("id: %d, #extensions: %d, localSubgraphInduction: %lu:  ", getId(i), getNumberOfExtensions(i), getLocalSubgraphInduction(i));
        for(int j = 0 ; j < getNumberOfExtensions(i) ; j++)
            printf("%d ", getExtension(i, j));
        printf("\n");
    }
    printf("jobs: %d, currentJob: %d, \n", getValidJobs(), getCurrentJob());
    for(int i = getCurrentJob() ; i < getValidJobs() ; i++) {
        printf("job: %d, currentPosOfJob: %d -> ", i, getCurrentPosOfJob(i));
        for(int j = 0 ; j <= getCurrentPosOfJob(i) ; j++) {
            printf("id: [%d], inductions: [%lu] ", getJob(i, j), getInductions(i, j));
        }
        printf("\n");
    }
    printf("\n\n");
}

int EnumerationHelper::getWeight() {
    int weight = 0;
    for(int i = getTargetLevel() ; i <= getCurrentPos() ; i++) {
        if(getNumberOfExtensions(i) != -1)
            weight += getNumberOfExtensions(i);
    }
    return weight;
}

int EnumerationHelper::getTargetLevel() {
    for(int k = 0 ; k <= getCurrentPos() && k < h_k - 2 ; k++) {
        if(getNumberOfExtensions(k) > 0)
            return k;
    }    
    return -1;
}

void EnumerationHelper::setValidJobs(int amount) {
    h_validJobs[getWarpId()] = amount;
}

int EnumerationHelper::getValidJobs() {
    return h_validJobs[getWarpId()];
}

void EnumerationHelper::setCurrentJob(int pos) {
    h_currentJob[getWarpId()] = pos;
}

int EnumerationHelper::getCurrentJob() {
    return h_currentJob[getWarpId()];
}

void EnumerationHelper::increaseJob() {
    h_validJobs[getWarpId()]++;
}

bool EnumerationHelper::isDonator() {
    return getTargetLevel() != -1;
}

void EnumerationHelper::setJob(int job, int k, int value) {
    h_jobs[getWarpId()*h_jobsPerWarp*h_warpSize + job*h_warpSize + k] = value;
}

int EnumerationHelper::getJob(int job, int k) {
    return h_jobs[getWarpId()*h_jobsPerWarp*h_warpSize + job*h_warpSize + k];
}

long unsigned int EnumerationHelper::getInductions(int job, int k) {
    return h_inductions[getWarpId()*h_jobsPerWarp*h_warpSize + job*h_warpSize + k];
}

void EnumerationHelper::setCurrentPosOfJob(int job, int k) {
    h_currentPosOfJob[getWarpId()*h_jobsPerWarp + job] = k;
}

void EnumerationHelper::setInductions(int job, int k, int value) {
    h_inductions[getWarpId()*h_jobsPerWarp*h_warpSize + job*h_warpSize + k] = value;
}

int EnumerationHelper::getCurrentPosOfJob(int job) {
    return h_currentPosOfJob[getWarpId()*h_jobsPerWarp+job];
}

unsigned long EnumerationHelper::getResult() {
    return h_result[getWarpId()];
}

bool EnumerationHelper::jobQueueIsEmpty() {
    return getCurrentJob() >= getValidJobs();
}

EnumerationHelper::~EnumerationHelper() {

}
