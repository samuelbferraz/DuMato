// Host

dm_createDataStructures();
dm_run("clique");

// GPU

__global__ void clique() {
    dm_retrieveThreadInformation();

    int k = 0, firstEmbeddingVertexId;

    while(true) {
        if(!dm_active(warpId)) {
            firstEmbeddingVertexId = dm_popJob();
            if(firstEmbeddingVertexId == -1)
                break;
        }

        if(dm_numberOfValidExtensions(warpId, k) == -1) {
            dm_generateAllExtensions(warpId, k, begin=k, end=k);
            foreach(i < k) {
                dm_filterExtensionsLowerThan(warpId, k, dm_id(i));
            }
            foreach(extension in dm_extensions(k)) {
                foreach(i < k) {
                    bool found = dm_findNeighbourhood(warpId, dm_id(i), extension);
                    if(!found) {
                        dm_invalidateExtension(warpId, k, extension)
                        break;
                    }
                }
            }
            dm_compactExtensions(warpId, k);
        }

        if(dm_numberOfValidExtensions(warpId, k) != 0) {
            if(k == dm_k()-1)
                dm_validSubgraphs(warpId);
            else
                dm_forward(warpId, &k);
        }
        else
            dm_backward(warpId, &k);
    }

}
