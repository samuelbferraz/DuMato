#ifndef STRUCTS_H
#define STRUCTS_H

#define CHUNK_SIZE 10000000

#define GPU_BUFFER_SIZE 7700000
#define CHUNKS_GPU 100

#define CPU_BUFFER_SIZE 7700000
#define CHUNKS_CPU 200

typedef struct {
    int *id;
    int *numberOfExtensions;
} Embeddings;

typedef struct {
    int* extensions;
    // long unsigned int *extensionsQuick;
    int* extensionSources;
    int* extensionSourcesOffset;
} Extensions;

typedef struct {
    int wid;
    int targetLevel;
} Donator;

#endif
