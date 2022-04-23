# DuMato: An Efficient GPU-accelerated Graph Pattern Mining System

*DuMato* is a runtime system with a high-level API that efficiently executes GPM algorithms on GPU.

## Requirements
- CUDA Toolkit >= 10.1
- gcc 7.5.0
- GNU Make >= 4.1

## Input
*DuMato* supports undirected graphs and uses the following input format:

src_vertex dst_vertex

src_vertex dst_vertex

*DuMato* expects vertex ids of a graph G to be in the range [0 .. V(G)-1].

## Compilation

We provide two examples of application: clique counting and motif counting.

In order to compile, access the directory and type:

>make sm=compute_capability

Where *compute_capability* is the compute capability of your NVIDIA GPU. For example, the compute capability of the GPU used in our experiments is 7.0 and we used the following compilation line:

>make sm=70
