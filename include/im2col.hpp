#ifndef IM2COL_HPP_
#define IM2COL_HPP_
#include "utils.hpp"

// this header declares the function that unrolls an image for matrix-multiplication based convolution
// partially borrowed from caffe

__global__ void im2col_kernel(const int n,
                              half* data_im,
                              const int channels,
                              const int height,
                              const int width,
                              const int kernel_h,
                              const int kernel_w,
                              const int pad_h,
                              const int pad_w,
                              const int stride_h,
                              const int stride_w,
                              const int out_h,
                              const int out_w,
                              half* data_col);

void im2col(half* data_im,
            const int channels,
            const int height,
            const int width,
            const int kernel_h,
            const int kernel_w,
            const int pad_h,
            const int pad_w,
            const int stride_h,
            const int stride_w,
            half* data_col
            );

#endif
