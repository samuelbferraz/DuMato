ALL_CCFLAGS :=
GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)

all:build

build:generateQuickMap motifs clique

Graph.o:Graph.cpp Graph.h
	g++ -c $<

Timer.o:Timer.cpp Timer.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

Manager.o:Manager.cpp Manager.h Graph.h CudaHelperFunctions.h Structs.h Device.h Timer.h QuickMapping.h EnumerationHelper.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

QuickMapping.o:QuickMapping.cpp QuickMapping.h nauty.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

motifs.o:motifs.cu Manager.h DuMato.h nauty.h Structs.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

clique.o:clique.cu Manager.h DuMato.h nauty.h Structs.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

generateQuickMap.o:generateQuickMap.cpp Graph.h nauty.h
	g++ -c -pthread $<

generateQuickMap:generateQuickMap.o Graph.o nauty.a
	g++ -pthread $^ -o $@

EnumerationHelper.o: EnumerationHelper.cpp EnumerationHelper.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

motifs:motifs.o Manager.o Timer.o Graph.o QuickMapping.o EnumerationHelper.o nauty.a
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) $^ -o $@

clique:clique.o Manager.o Timer.o Graph.o QuickMapping.o EnumerationHelper.o nauty.a
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) $^ -o $@

clean:
	rm *.o
