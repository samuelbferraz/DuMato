namespace DuMato {
    __device__ Device* d;

    __device__ int *vertexOffset;
    __device__ int *adjacencyList;
    __device__ int *degree;

    // Enumeration data structures
    __device__ int *id;
    __device__ int *numberOfExtensions;
    __device__ int *traversedExtensions;
    __device__ unsigned long* result;


    /*------------------------------------------------------------------------*/

    // __shared__ int k[1024];
    __device__ int k[409600];

    __device__ int dm_globalK() {
        return *(d->d_k);
    }

    __device__ int neighbour(int vertexId, int relativePosition) {
        return adjacencyList[vertexOffset[vertexId]+relativePosition];
    }

    __device__ int roundToWarpSize(int value) {
        return ((int)ceilf((float)value / (float)(*(d->d_virtualWarpSize)))) * (*(d->d_virtualWarpSize));
    }

    __device__ int dm_getTid() {
        return (blockIdx.x * blockDim.x) + threadIdx.x;
    }

    __device__ int dm_getWid() {
        return dm_getTid() / *(d->d_virtualWarpSize);
    }

    __device__ int dm_getLocalWid() {
        return threadIdx.x / *(d->d_virtualWarpSize);
    }

    __device__ int dm_getLane() {
        return threadIdx.x & 0x1f;
    }

    __device__ int dm_offsetWarp() {
        return dm_getWid() * dm_globalK();
    }

    __device__ int dm_offsetExtensions() {
        return dm_getWid() * dm_globalK();
    }

    __device__ int dm_k() {
        return k[dm_getTid()];
        // return k[threadIdx.x];
    }

    __device__ void dm_k(int value) {
        k[dm_getTid()] = value;
        // k[threadIdx.x] = value;
    }

    __device__ void dm_popJob() {
        if(dm_k() != -1)
            return;

        int vertex = atomicAdd((int*)d->d_globalVertexId, 1);
        // job = 0;

        if(vertex > *(d->d_maxVertexId)) {
            vertex = -1;
            // status[dm_getTid()] = 2;
            // smid[dm_getTid()] = -1;
        }
        else {
            dm_k(0);
            id[dm_offsetWarp()] = vertex;
            for(int i = 0 ; i < dm_globalK() ; i++) {
                numberOfExtensions[dm_offsetWarp()+i] = degree[vertex];
                traversedExtensions[dm_offsetWarp()+i] = 0;
            }
            // status[dm_getTid()] = 1;
            // smid[dm_getTid()] = __mysmid();
        }
    }

    __device__ void dm_start(Device* device) {
        d = device;

        vertexOffset = device->d_vertexOffset;
        adjacencyList = device->d_adjacencyList;
        degree = device->d_degree;

        id = device->d_id;
        result = device->d_result;
        numberOfExtensions = device->d_numberOfExtensions;
        traversedExtensions = device->d_traversedExtensions;


        result[dm_getWid()] = 0;
        dm_k(-1);
        dm_popJob();
    }

    __device__ void dm_end() {
        // currentPos[dm_getWid()] = dm_k();
    }

    __device__ int dm_id(int position) {
        return id[dm_offsetWarp()+position];
    }

    __device__ bool dm_active() {
        return dm_k() >= 0;
    }

    __device__ int dm_numberOfExtensions() {
        return numberOfExtensions[dm_offsetWarp()+dm_k()];
    }

    __device__ int dm_traversedExtensions() {
        return traversedExtensions[dm_offsetWarp()+dm_k()];
    }

    __device__ int dm_generateExtension() {
        if(dm_traversedExtensions() < dm_numberOfExtensions()) {
            int extension = neighbour(dm_id(dm_k()),dm_traversedExtensions());
            traversedExtensions[dm_offsetWarp()+dm_k()]++;
            return extension;
        }
        else
            return -1;
    }

    // Careful: don't use it in a virtual warp size != 32
    __device__ bool dm_findNeighbourhood(int vertexId, int possibleNeighbour) {
        int found = 0, currentDegree = degree[vertexId];
        for(int warpPosition = dm_getLane() ; warpPosition < roundToWarpSize(currentDegree) && !found ; warpPosition += 32) {
            found = __any_sync(0xffffffff, warpPosition < currentDegree && neighbour(vertexId, warpPosition) == possibleNeighbour ? 1 : 0);
        }
        return found == 1 ? true : false;
    }

    // Careful: works only for virtual warp size == 1
    __device__ bool dm_findNeighbourhood2(int vertexId, int possibleNeighbour) {
        int found = 0, currentDegree = degree[vertexId];
        for(int threadPosition = 0 ; threadPosition < currentDegree && !found ; threadPosition++) {
            if(neighbour(vertexId, threadPosition) == possibleNeighbour)
                found = 1;
        }
        return found == 1 ? true : false;
    }

    __device__ void dm_validSubgraphs() {
        // for(int i = dm_getLane() ; i < dm_numberOfExtensions() ; i += 32) {
        //     printf("[%d,%d,%d,%d]: %d %d %d\n", dm_k(), dm_getWid(), dm_offsetWarp(), dm_offsetExtensions(), dm_id(0), dm_id(1), dm_getExtension(i));
        // }
        // __syncwarp();

        result[dm_getWid()]++;
    }

    __device__ void dm_forward(int extension) {
        dm_k(dm_k()+1);
        id[dm_offsetWarp()+dm_k()] = extension;
        numberOfExtensions[dm_offsetWarp()+dm_k()] = degree[extension];
        traversedExtensions[dm_offsetWarp()+dm_k()] = 0;
    }

    // TODO Keep currentVertexDegree consistent, as well as localOffsetExtensions
    __device__ void dm_backward() {
        dm_k(dm_k()-1);
        if(!dm_active())
            dm_popJob();
    }
}
