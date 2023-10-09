ALL_CCFLAGS :=
GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)
OBJECT_DIR := obj
MAIN_SRC_DIR := src/main
MOTIFS_HAND_WC_DIR := src/motifs_HAND_WC
MOTIFS_DM_DFS_DIR := src/motifs_DM_DFS
CLIQUE_HAND_WC_DIR := src/clique_HAND_WC
CLIQUE_DM_DFS_DIR := src/clique_DM_DFS
DM_WCV_DIR := src/DM_WCV
EXEC_DIR := exec

all:build motifs_HAND_WC motifs_DM_DFS clique_HAND_WC clique_DM_DFS DM_WCV

build:$(EXEC_DIR)/clique_DM_WC $(EXEC_DIR)/clique_DM_WCLB $(EXEC_DIR)/clique $(EXEC_DIR)/motifs_DM_WC $(EXEC_DIR)/motifs_DM_WCLB $(EXEC_DIR)/motifs

$(OBJECT_DIR)/clique_DM_WC.o:$(MAIN_SRC_DIR)/clique_DM_WC.cu $(MAIN_SRC_DIR)/DuMato.h $(MAIN_SRC_DIR)/Structs.cu $(MAIN_SRC_DIR)/DuMatoCPU.h $(MAIN_SRC_DIR)/DuMatoGPU.cu $(MAIN_SRC_DIR)/Timer.h $(MAIN_SRC_DIR)/Report.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<
	mv clique_DM_WC.o $(OBJECT_DIR)/clique_DM_WC.o

$(OBJECT_DIR)/clique_DM_WCLB.o:$(MAIN_SRC_DIR)/clique_DM_WCLB.cu $(MAIN_SRC_DIR)/DuMato.h $(MAIN_SRC_DIR)/Structs.cu $(MAIN_SRC_DIR)/DuMatoCPU.h $(MAIN_SRC_DIR)/DuMatoGPU.cu $(MAIN_SRC_DIR)/Timer.h $(MAIN_SRC_DIR)/Report.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<
	mv clique_DM_WCLB.o $(OBJECT_DIR)/clique_DM_WCLB.o

$(OBJECT_DIR)/motifs_DM_WC.o:$(MAIN_SRC_DIR)/motifs_DM_WC.cu $(MAIN_SRC_DIR)/DuMato.h $(MAIN_SRC_DIR)/Structs.cu $(MAIN_SRC_DIR)/DuMatoCPU.h $(MAIN_SRC_DIR)/DuMatoGPU.cu $(MAIN_SRC_DIR)/Timer.h $(MAIN_SRC_DIR)/Report.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<
	mv motifs_DM_WC.o $(OBJECT_DIR)/motifs_DM_WC.o

$(OBJECT_DIR)/motifs_DM_WCLB.o:$(MAIN_SRC_DIR)/motifs_DM_WCLB.cu $(MAIN_SRC_DIR)/DuMato.h $(MAIN_SRC_DIR)/Structs.cu $(MAIN_SRC_DIR)/DuMatoCPU.h $(MAIN_SRC_DIR)/DuMatoGPU.cu $(MAIN_SRC_DIR)/Timer.h $(MAIN_SRC_DIR)/Report.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<
	mv motifs_DM_WCLB.o $(OBJECT_DIR)/motifs_DM_WCLB.o

$(OBJECT_DIR)/Graph.o:$(MAIN_SRC_DIR)/Graph.cpp $(MAIN_SRC_DIR)/Graph.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<
	mv Graph.o $(OBJECT_DIR)/Graph.o

$(OBJECT_DIR)/Timer.o:$(MAIN_SRC_DIR)/Timer.cpp $(MAIN_SRC_DIR)/Timer.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<
	mv Timer.o $(OBJECT_DIR)/Timer.o

$(OBJECT_DIR)/QuickMapping.o:$(MAIN_SRC_DIR)/QuickMapping.cpp $(MAIN_SRC_DIR)/QuickMapping.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<
	mv QuickMapping.o $(OBJECT_DIR)/QuickMapping.o 

$(OBJECT_DIR)/EnumerationHelper.o: $(MAIN_SRC_DIR)/EnumerationHelper.cpp $(MAIN_SRC_DIR)/EnumerationHelper.h $(MAIN_SRC_DIR)/Structs.cu
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<
	mv EnumerationHelper.o $(OBJECT_DIR)/EnumerationHelper.o

$(OBJECT_DIR)/Report.o: $(MAIN_SRC_DIR)/Report.cpp $(MAIN_SRC_DIR)/Report.h $(MAIN_SRC_DIR)/Structs.cu $(MAIN_SRC_DIR)/DuMatoCPU.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<
	mv Report.o $(OBJECT_DIR)/Report.o

$(OBJECT_DIR)/DuMatoCPU.o:$(MAIN_SRC_DIR)/DuMatoCPU.cu $(MAIN_SRC_DIR)/DuMatoCPU.h $(MAIN_SRC_DIR)/Structs.cu $(MAIN_SRC_DIR)/Graph.h $(MAIN_SRC_DIR)/EnumerationHelper.h $(MAIN_SRC_DIR)/Timer.h $(MAIN_SRC_DIR)/QuickMapping.h
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) -c $<
	mv DuMatoCPU.o $(OBJECT_DIR)/DuMatoCPU.o

$(EXEC_DIR)/clique_DM_WC:$(OBJECT_DIR)/clique_DM_WC.o $(OBJECT_DIR)/Timer.o $(OBJECT_DIR)/Graph.o $(OBJECT_DIR)/EnumerationHelper.o $(OBJECT_DIR)/DuMatoCPU.o $(OBJECT_DIR)/QuickMapping.o $(OBJECT_DIR)/Report.o
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) $^ -o $@

$(EXEC_DIR)/clique_DM_WCLB:$(OBJECT_DIR)/clique_DM_WCLB.o $(OBJECT_DIR)/Timer.o $(OBJECT_DIR)/Graph.o $(OBJECT_DIR)/EnumerationHelper.o $(OBJECT_DIR)/DuMatoCPU.o $(OBJECT_DIR)/QuickMapping.o $(OBJECT_DIR)/Report.o
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) $^ -o $@

$(EXEC_DIR)/clique:$(EXEC_DIR)/clique_DM_WCLB
	cp $(EXEC_DIR)/clique_DM_WCLB $(EXEC_DIR)/clique

$(EXEC_DIR)/motifs_DM_WC:$(OBJECT_DIR)/motifs_DM_WC.o $(OBJECT_DIR)/Timer.o $(OBJECT_DIR)/Graph.o $(OBJECT_DIR)/EnumerationHelper.o $(OBJECT_DIR)/DuMatoCPU.o $(OBJECT_DIR)/QuickMapping.o $(OBJECT_DIR)/Report.o
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) $^ -o $@

$(EXEC_DIR)/motifs_DM_WCLB:$(OBJECT_DIR)/motifs_DM_WCLB.o $(OBJECT_DIR)/Timer.o $(OBJECT_DIR)/Graph.o $(OBJECT_DIR)/EnumerationHelper.o $(OBJECT_DIR)/DuMatoCPU.o $(OBJECT_DIR)/QuickMapping.o $(OBJECT_DIR)/Report.o
	nvcc $(ALL_CCFLAGS) $(GENCODE_FLAGS) $^ -o $@
	cp $(EXEC_DIR)/motifs_DM_WCLB $(EXEC_DIR)/motifs

$(EXEC_DIR)/motifs:$(EXEC_DIR)/motifs_DM_WCLB
	cp $(EXEC_DIR)/motifs_DM_WCLB $(EXEC_DIR)/motifs

motifs_HAND_WC:
	cd $(MOTIFS_HAND_WC_DIR) && $(MAKE)
	cp $(MOTIFS_HAND_WC_DIR)/motifs_HAND_WC $(EXEC_DIR)/
	cd $(MOTIFS_HAND_WC_DIR) && $(MAKE) clean


clique_HAND_WC:
	cd $(CLIQUE_HAND_WC_DIR) && $(MAKE)
	cp $(CLIQUE_HAND_WC_DIR)/clique_HAND_WC $(EXEC_DIR)/
	cd $(CLIQUE_HAND_WC_DIR) && $(MAKE) clean


clique_DM_DFS:
	cd $(CLIQUE_DM_DFS_DIR) && $(MAKE)
	cp $(CLIQUE_DM_DFS_DIR)/clique_DM_DFS $(EXEC_DIR)/
	cd $(CLIQUE_DM_DFS_DIR) && $(MAKE) clean


motifs_DM_DFS:
	cd $(MOTIFS_DM_DFS_DIR) && $(MAKE) sm=70
	cp $(MOTIFS_DM_DFS_DIR)/motifs_DM_DFS $(EXEC_DIR)/
	cd $(MOTIFS_DM_DFS_DIR) && $(MAKE) clean

DM_WCV:
	cd $(DM_WCV_DIR) && $(MAKE) sm=70
	cp $(DM_WCV_DIR)/clique_DM_WCV8 $(EXEC_DIR)/
	cp $(DM_WCV_DIR)/clique_DM_WCV16 $(EXEC_DIR)/
	cp $(DM_WCV_DIR)/motifs_DM_WCV8 $(EXEC_DIR)/
	cp $(DM_WCV_DIR)/motifs_DM_WCV16 $(EXEC_DIR)/
	cd $(DM_WCV_DIR) && $(MAKE) clean

clean:
	rm $(OBJECT_DIR)/*.o
	rm $(EXEC_DIR)/*
