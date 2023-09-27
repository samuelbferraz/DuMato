ALL_CCFLAGS :=
GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)

all:build

build:motifs_DM_WCV8 motifs_DM_WCV16 clique_DM_WCV8 clique_DM_WCV16

Graph.o:Graph.cpp Graph.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

Timer.o:Timer.cpp Timer.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

Manager.o:Manager.cpp Manager.h Graph.h CudaHelperFunctions.h Structs.h Device.h Timer.h QuickMapping.h EnumerationHelper.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

QuickMapping.o:QuickMapping.cpp QuickMapping.h nauty.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

EnumerationHelper.o: EnumerationHelper.cpp EnumerationHelper.h Structs.cu
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

Report.o: Report.cpp Report.h Structs.cu DuMatoCPU.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

motifs_DM_WCV8.o:motifs_DM_WCV8.cu DuMato.h Structs.cu DuMatoCPU.h DuMatoGPU.cu Timer.h Report.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

motifs_DM_WCV8:motifs_DM_WCV8.o Timer.o Graph.o EnumerationHelper.o DuMatoCPU.o QuickMapping.o Report.o
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) $^ -o $@

motifs_DM_WCV16.o:motifs_DM_WCV16.cu DuMato.h Structs.cu DuMatoCPU.h DuMatoGPU.cu Timer.h Report.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

motifs_DM_WCV16:motifs_DM_WCV16.o Timer.o Graph.o EnumerationHelper.o DuMatoCPU.o QuickMapping.o Report.o
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) $^ -o $@

DuMatoCPU.o:DuMatoCPU.cu DuMatoCPU.h Structs.cu Graph.h EnumerationHelper.h Timer.h QuickMapping.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

clique_DM_WCV8.o:clique_DM_WCV8.cu DuMato.h Structs.cu DuMatoCPU.h DuMatoGPU.cu Timer.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

clique_DM_WCV8:clique_DM_WCV8.o Timer.o Graph.o EnumerationHelper.o DuMatoCPU.o QuickMapping.o
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) $^ -o $@

clique_DM_WCV16.o:clique_DM_WCV16.cu DuMato.h Structs.cu DuMatoCPU.h DuMatoGPU.cu Timer.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<

clique_DM_WCV16:clique_DM_WCV16.o Timer.o Graph.o EnumerationHelper.o DuMatoCPU.o QuickMapping.o
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) $^ -o $@

clean:
	rm *.o
	rm clique_DM_WCV8 clique_DM_WCV16 motifs_DM_WCV8 motifs_DM_WCV16
