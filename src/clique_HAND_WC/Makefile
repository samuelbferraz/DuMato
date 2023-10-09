ALL_CCFLAGS :=
GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)

all:build

build:clique_HAND_WC

Graph.o:Graph.cpp Graph.h
	g++ -c $<

Timer.o:Timer.cpp Timer.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

Manager.o:Manager.cpp Manager.h Graph.h CudaHelperFunctions.h Structs.h Device.h Timer.h QuickMapping.h EnumerationHelper.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

clique_HAND_WC.o:clique_HAND_WC.cu Manager.h DuMato.h nauty.h Structs.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

EnumerationHelper.o: EnumerationHelper.cpp EnumerationHelper.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

clique_HAND_WC:clique_HAND_WC.o Manager.o Timer.o Graph.o QuickMapping.o EnumerationHelper.o nauty.a
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) $^ -o $@

clean:
	rm *.o
	rm clique_HAND_WC
