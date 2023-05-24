# DuMato: An Efficient GPU-accelerated Graph Pattern Mining System

*DuMato* is a runtime system with a high-level API that efficiently executes GPM algorithms on GPU.

## Requirements
- CUDA Toolkit >= 10.1
- gcc 7.5.0
- GNU Make >= 4.1
- Nauty canonical relabeling tool (nauty.h and nauty.a, provided in the source)

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

Some datasets are available in the *dataset* directory. Du to lack of space in github, the full set of datasets can be obtained in the following link: YYY

## Dictionaries

Some dictionaries (needed for canonical relabeling on GPU) are available in the *dictionaries* directory. Due to the lack of space in github, the full set of dictionaries can be obtained in the following link: YYY.

## Executing applications
Both applications (clique counting and motif counting) require the following arguments for execution:

>./app_name graph_file k number_of_threads block_size number_of_SMs load_balancing_threshold jobs_per_warp

Where:
> -app_name: motif_counting or clique_counting. <br />
> -graph_file: url of graph dataset.<br />
> -k: size of enumerated subgraphs.<br />
> -number_of_threads: number of threads to instantiate on GPU.<br />
> -block_size: block size on GPU.<br/>
> -number_of_SMS: number of streaming multiprocessor (SM) in the target GPU. Needed for the runtime report.<br />
> -load_balancing_threshold: the threshold (percentage) of idle threads allowed in the enumeration. After this threshold, the load-balancing layer is invoked.<br />
> -jobs_per_warp: the size of the job queue per warp.<br />

Our experimental evaluation suggests that 102400 threads, blocks with 256 threads, 30\% of load balancing threshold and 16 jobs per warp is a good choice.

For example, motif counting can be executed using the following command line: <br />

> ./motifs datasets/citeseer.edgelist 5 102400 256 80 30 16

The command line above would run motif counting to search for motifs with 5 vertices using 409600 threads, blocks with 256 threads, a GPU with 80 SMs, a threshold of 10\% (up to 90\% of threads are allowed to be idle), a runtime report being exhibit every second (1000 ms) and canonical relabeling required on GPU.

Clique counting can be executed using the following command line: <br />

> ./clique datasets/citeseer.edgelist 5 102400 256 80 30 16

## References

S. Ferraz, V. Dias, C. H. C. Teixeira, G. Teodoro and W. Meira, "Efficient Strategies for Graph Pattern Mining Algorithms on GPUs," 2022 IEEE 34th International Symposium on Computer Architecture and High Performance Computing (SBAC-PAD), Bordeaux, France, 2022, pp. 110-119, doi: 10.1109/SBAC-PAD55451.2022.00022.
