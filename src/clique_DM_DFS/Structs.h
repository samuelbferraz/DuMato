#ifndef STRUCTS_H
#define STRUCTS_H

#define DEBUG_BUFFER_SIZE 1

typedef struct {
    int *id;
    int *numberOfExtensions;
    int *traversedExtensions;
    int *currentOffsetNeighbours;
} Embeddings;

typedef struct {
    int* extensions;
    int* extensionSources;
    int* neighbours;
    int* neighbourSources;
    bool* readAllNeighbours;
} Extensions;

typedef struct {
    int wid;
    int targetLevel;
} Donator;

#endif
