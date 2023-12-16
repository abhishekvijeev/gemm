
<h1 align="center">Matrix Multiplication with CUDA</h1>

This repository is a product of my efforts to systematically understand optimization strategies commonly used to improve the arithmetic intensity of data-parallel algorithms, specifically in the context of dense general matrix multiplication (GEMM) which serves as an ideal starting point due to (a) its ubiquity in contemporary deep neural networks and (b) it being ["the most optimized mathematical operation in the history of computing"](https://arxiv.org/pdf/2311.10770.pdf).

Each data-parallel algorithm, henceforth referred to as a "kernel", has been implemented in CUDA on account of its widespread adoption within the HPC community.

## Build

```
mkdir build
cd build
cmake ../
cmake --build .
```

Details about command line parameters can be obtained with `--help`

Run the program

```
./main
```
