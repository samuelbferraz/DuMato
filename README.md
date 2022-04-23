# DuMato: An Efficient GPU-accelerated Graph Pattern Mining System

*DuMato* is a runtime system with a high-level API that efficiently executes GPM algorithms on GPU.

## Requirements
- CUDA Toolkit >= 10.1
- gcc 7.5.0
- GNU Make >= 4.1

## Input
*DuMato* supports undirected graphs and uses the following input format:

src_vertex dst_vertex <br />
src_vertex dst_vertex <br />

*DuMato* expects vertex ids of a graph G to be in the range [0 .. V(G)-1].

## Compilation

We provide two examples of application: clique counting (clique_counting.cu) and motif counting (motif_counting.cu).

In order to compile, access the directory and type:

>make sm=compute_capability

The flag *compute_capability* is the compute capability of your NVIDIA GPU. For example, the compute capability of the GPU used in our experiments is 7.0 and we used the following compilation line:

>make sm=70

## Datasets

All datasets used in the experiments are available as a zip file in the *data* folder. Please unzip the datasets prior to the execution.


## Dictionaries

All dictionaries (needed for canonical relabeling on GPU) are provided as a zip in the *dictionaries* folder. Please unzip the the dictionaries prior to the execution.

## Executing applications
Both applications (clique counting and motif counting) require the following arguments for execution:

>./app_name graph_file k number_of_threads block_size number_of_SMs report_interval

Where:
> -app_name: motif_counting or clique_counting. <br />
> -graph_file: url of graph dataset.<br />
> -k: size of enumerated subgraphs.<br />
> -number_of_threads: number of threads to instantiate on GPU.<br />
> -number_of_SMS: number of streaming multiprocessor (SM) in the target GPU. Needed for the runtime report.<br />
> -report_interval: the frequency (in millisecons) the runtime report should appear in the screen during execution.<br />

For example, motif counting can be executed using the following command line:

> ./motif_counting data/dblp.edgelist 5 409600 256 80 90 1000

The command line above would run motif counting to search for motifs with 5 vertices using 409600 threads, blocks with 256 threads, a GPU with 80 SMs, a threshold of 10\% (up to 90\% of threads are allowed to be idle) and a runtime report being exhibit every second (1000 ms).
