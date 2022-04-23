ALL_CCFLAGS :=
GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)

all:build

build:motif_counting clique_counting

Graph.o:Graph.cpp Graph.h
	g++ -c $<

Timer.o:Timer.cpp Timer.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

Manager.o:Manager.cpp Manager.h Graph.h CudaHelperFunctions.h Structs.h Device.h Timer.h QuickMapping.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

QuickMapping.o:QuickMapping.cpp QuickMapping.h nauty.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

motif_counting.o:motif_counting.cu Manager.h DuMato.h nauty.h Structs.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

motif_counting:motif_counting.o Manager.o Timer.o Graph.o QuickMapping.o nauty.a
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) $^ -o $@

clique_counting.o:clique_counting.cu Manager.h DuMato.h nauty.h Structs.h
		nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

clique_counting:clique_counting.o Manager.o Timer.o Graph.o QuickMapping.o nauty.a
		nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) $^ -o $@

clean:
	rm *.o
