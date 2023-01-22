#ifndef MATRIX_HPP_
#define MATRIX_HPP_

#include "utils.hpp"

const int TILE_WIDTH = 16;
const int WARP_SIZE = 32;

// matrix multiplication using tensor cores (through wmma)
extern __global__ void wmma_kernel(half* a, half* b, float* c, int M, int N, int K);
extern float* gemm(half* a, half* b, int M, int N, int K);

// matrix multiplication using cuda cores (for debugging purposes only)
__global__ void matrix_mul_kernel(float *Ad, float *Bd, float *Cd, int M, int N, int K);
__global__ void matrix_mul_kernel_half(half *Ad, half *Bd, half *Cd, int M, int N, int K);

#endif
