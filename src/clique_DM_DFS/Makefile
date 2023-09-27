ALL_CCFLAGS :=
GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)

all:build

build:clique_DM_DFS

Graph.o:Graph.cpp Graph.h
	g++ -c $<

Timer.o:Timer.cpp Timer.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

Manager.o:Manager.cpp Manager.h Device.h Structs.h CudaHelperFunctions.h Graph.h Timer.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

clique_DM_DFS.o:clique_DM_DFS.cu Graph.h Device.h Manager.h DuMato.h Timer.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

clique_DM_DFS: Graph.o clique_DM_DFS.o Manager.o Timer.o
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) $^ -o $@

clean:
	rm clique_DM_DFS *.o
