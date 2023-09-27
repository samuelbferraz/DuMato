#ifndef STRUCTS_H
#define STRUCTS_H

#define CHUNK_SIZE 10000000

#define GPU_BUFFER_SIZE 7700000
#define CHUNKS_GPU 100

#define CPU_BUFFER_SIZE 7700000
#define CHUNKS_CPU 200


//#define GPU_BUFFER_SIZE_PER_WARP 10000
#define GPU_BUFFER_SIZE_PER_WARP 650000 // Around 7 Gb of memory for 102400 threads

typedef struct {
    int *id;
    int *numberOfExtensions;
    int *currentPosOfJob;
    int *currentJob;
    int *jobs; 
} Embeddings;

typedef struct {
    int wid;
    int targetLevel;
    int weight;
} Donator;

#endif
