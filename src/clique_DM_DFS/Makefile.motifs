ALL_CCFLAGS :=
GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)

all:build

build:motifs_DM_DFS

Graph.o:Graph.cpp Graph.h
	g++ -c $<

Timer.o:Timer.cpp Timer.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

Manager.o:Manager.cpp Manager.h Graph.h CudaHelperFunctions.h Structs.h Device.h Timer.h QuickMapping.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

QuickMapping.o:QuickMapping.cpp QuickMapping.h nauty.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

gpm.o:gpm.cu Manager.h DuMato.h nauty.h Structs.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

motifs_DM_DFS.o:motifs_DM_DFS.cu Manager.h DuMato.h nauty.h Structs.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

motifs_DM_DFS:motifs_DM_DFS.o Manager.o Timer.o Graph.o QuickMapping.o nauty.a
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) $^ -o $@

clean:
	rm motifs_DM_DFS *.o
