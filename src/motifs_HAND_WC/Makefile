ALL_CCFLAGS :=
GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)

all:build

build:motifs_HAND_WC

Graph.o:Graph.cpp Graph.h
	g++ -c $<

Timer.o:Timer.cpp Timer.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

Manager.o:Manager.cpp Manager.h Graph.h CudaHelperFunctions.h Structs.h Device.h Timer.h QuickMapping.h EnumerationHelper.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

QuickMapping.o:QuickMapping.cpp QuickMapping.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

motifs_HAND_WC.o:motifs_HAND_WC.cu Manager.h DuMato.h Structs.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

motifs_HAND_WC:motifs_HAND_WC.o Manager.o Timer.o Graph.o QuickMapping.o EnumerationHelper.o nauty.a
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) $^ -o $@

clean:
	rm *.o
