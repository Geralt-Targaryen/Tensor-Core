#include "include/im2col.hpp"

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
                              half* data_col)
{
    // unroll an image for convolution, and store it in column-major order
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<n){
      int c_index = tid / (out_w * out_h);
      int kid = tid % (out_w * out_h);
      int h_index = kid / out_w * stride_h - pad_h;
      int w_index = kid % out_w * stride_w - pad_w;
      const half* data_im_ptr = data_im + c_index * height * width + h_index * width + w_index;
      half* data_col_ptr = data_col + (kernel_h * kernel_w * channels * kid) + (kernel_h * kernel_w) * c_index;

      for (int i = 0; i < kernel_h; ++i) {
        for (int j = 0; j < kernel_w; ++j) {
          int h_im = h_index + i;
          int w_im = w_index + j;
          *data_col_ptr =
              (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) ?
              data_im_ptr[i * width + j] : __float2half(0);
          ++data_col_ptr;
        }
      }
    }
}


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
            ){
      // both data_im and data_col are assumed to be on the device
      // launch channels * height_col * width_col kernels, each responsible for copying a single-channel grid.
      // height_col, width_col: shape of the output (each column in matrix $col$ corresponds to one element in the output)
      int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
      int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
      int num_kernels = channels * height_col * width_col;

      im2col_kernel<<<(num_kernels+255)/256, 256>>>(
        num_kernels, data_im, channels, height, width, kernel_h, kernel_w, pad_h,
        pad_w, stride_h, stride_w, height_col, width_col, data_col);
}

