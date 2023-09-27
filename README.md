
# DuMato: An Efficient GPU-accelerated Graph Pattern Mining System

DuMato is a runtime system with a high-level API that efficiently executes GPM algorithms on GPU using CUDA/C++. 

## Requirements
- CUDA Toolkit >= 10.1 and < 12 (unexpected issues are happening in the latest version of CUDA)
- $PATH variable pointing to the nvcc compiler
- gcc >= 7.5.0.
- GNU Make >= 4.1

## Compilation

In order to compile, access the root directory and type:

>make sm=compute_capability

The flag compute_capability is the compute capability of your NVIDIA GPU. For example, the compute capability of the GPU used in our experiments is 7.0 and we used the following compilation line:

>make sm=70

## Input
DuMato supports undirected graphs without labels and uses the following input format:

src_vertex dst_vertex <br />
src_vertex dst_vertex <br />

DuMato expects vertex ids of a graph G to be in the range [0 .. V(G)-1].

## Directory Tree

DuMato directory tree is organized as follows:

> src/
- Source code of all versions of DuMato described in the PhD thesis (available soon), according to the following table:

|Folder                 |Description
|-----|--------
|src/clique_DM_DFS      |Clique counting using DuMato API and standard DFS approach.|
|src/motifs_DM_DFS      |Motif counting using DuMato API and standard DFS approach.|
|src/clique_HAND_WC     |Clique counting using warp-centric steps but without DuMato API.|
|src/motifs_HAND_WC     |Motif counting using warp-centric steps but without DuMato API.          |
|src/DM_WCV             |Clique/motif counting using DuMato API, DFS-wide, warp-centric workflow with warp virtualization and warp-level load balancing.           |
|src/main               |Optimized clique/motif counting versions. It uses DuMato API, DFS-wide, warp-centric workflow and warp-level load balancing.

> obj/
- Object files of the optimized version of DuMato (src/main). The object files of other versions are located in the source folder of the version.


> exec/ <br />
- Executable files of all versions of DuMato, as follows:

|Executable                 |Description |
|-|-|
|clique_DM_DFS          |Clique counting using DuMato API and standard DFS approach.| 
|clique_HAND_WC         |Clique counting using warp-centric steps but without DuMato API.| 
|clique_DM_WCV8         |Clique counting using DuMato API, DFS-wide, warp-centric workflow using virtual warps with 8 threads, and warp-level load balancing.           | *dataset* *k*
|clique_DM_WCV16         |Clique counting using DuMato API, DFS-wide, warp-centric workflow using virtual warps with 16 threads, and warp-level load balancing.
|clique_DM_WC         |Clique counting using DuMato API, DFS-wide and warp-centric workflow.
|clique_DM_WCLB         |Clique counting using DuMato API, DFS-wide, warp-centric workflow and warp-level load balancing.
|motifs_DM_DFS          |Motif counting using DuMato API and standard DFS approach.|
|motifs_HAND_WC         |Motif counting using warp-centric steps but without DuMato API.          |
|motifs_DM_WCV8         |Motif counting using DuMato API, DFS-wide, warp-centric workflow using virtual warps with 8 threads, and warp-level load balancing.         |
|motifs_DM_WCV16         |Motif counting using DuMato API, DFS-wide, warp-centric workflow using virtual warps with 16 threads, and warp-level load balancing. |
|motifs_DM_WC         |Motif counting using DuMato API, DFS-wide and warp-centric workflow.
|motifs_DM_WCLB         |Motif counting using DuMato API, DFS-wide, warp-centric workflow and warp-level load balancing.

Execute each binary without passing arguments to discover the input parameters.

> datasets/ <br />

The datasets used in the experiments. Due to lack of space in github, the full set of datasets can be obtained in the following link: https://drive.google.com/file/d/1mTknrtvpF0OROG5JTsFNcOaIDD9fLb5M/view?usp=sharing

> dictionaries/ <br />

The dictionaries (csv format) needed for canonical relabeling on GPU. Due to the lack of space in github, the full set of dictionaries can be obtained in the following link: https://drive.google.com/file/d/1ZJJzqiLu6mGHuUTEBV9CLmEEhqCC58dD/view?usp=sharing.

> reproducibility/ <br />

Shell scripts used to reproduce the results of all experiments in the PhD thesis.

> results/ <br />

Folders to store the results produced by the scripts in the *reproducibility* folder.

## References

S. Ferraz, V. Dias, C. H. C. Teixeira, G. Teodoro and W. Meira, "Efficient Strategies for Graph Pattern Mining Algorithms on GPUs," 2022 IEEE 34th International Symposium on Computer Architecture and High Performance Computing (SBAC-PAD), Bordeaux, France, 2022, pp. 110-119, doi: 10.1109/SBAC-PAD55451.2022.00022.
