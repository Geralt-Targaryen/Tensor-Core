#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cfloat>
#include <cassert>
#include <time.h>
#include "mma.h"

using namespace nvcuda;
using std::cout;
using std::endl;
using std::vector;
using std::ifstream;
using std::stringstream;
using std::ios;

/*
  this header file declares all the helper functions that are shared among all the modules
  no function overload is used whenever cuda is involved, for the convenience of debugging
*/

const int CUDA_NUM_THREADS = 512;
inline int GET_NUM_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

// grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

// in-place transposition
__global__ void transpose_half_kernel(half *a, half *aT, int M, int N);
__global__ void transpose_kernel(float *a, float *aT, int M, int N);
void transpose_half(half *a, int M, int N);
void transpose(float *a, int M, int N);

// conversion between half-precision matrix and single-precision matrix
__global__ void float2half_matrix_kernel(const float *input, half *output, int N);
__global__ void half2float_matrix_kernel(const half *input, float *output, int N);
half* float2half_matrix(const float *input, int N);
float* half2float_matrix(const half *input, int N);

/*  
  simple addition of two matrices
  the forward pass of fc layer's bias is declared as a stand-alone function, as the bias vector
  needs to be broadcast, as opposed to the forward pass of residual module
*/
__global__ void linear_bias_kernel(float* input1, float* input2, int h, int w);
float * linear_bias(float* input1, float* input2, int h, int w);
__global__ void add_matrix_kernel(float* input1, float* input2, int N);
float * add_matrix(float* input1, float* input2, int N);

// argmax
__global__ void argmax_kernel(float* input, int num, int d, int *result);
int * argmax(float* input, int num, int d);

// accuracy (no cuda)
float accuracy_score(int *label, int *label_, int N);

// auxiliary functions: loading and printing matrices
void load_vector(vector<float> &v, const char *file);
void load_vector_int(vector<int> &v, const char *file);
void load_binary(vector<float> &v, const char *file, int size);
void load_binary(vector<int> &v, const char *file, int size);

void print_matrix(const float* x, int h, int w, int c, bool device=false);
void print_matrix(const int* x, int h, int w, int c, bool device=false);
void print_matrix_half(const half* x, int h, int w, int c, bool device=false);

#endif
