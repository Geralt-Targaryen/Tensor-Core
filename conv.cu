#include "include/conv.hpp"

conv2d::conv2d(
            const int kernel_h,
            const int kernel_w,
            const int stride_h,
            const int stride_w,
            const int pad_h,
            const int pad_w,
            const int conv_in_channels,
            const int conv_out_channels
         ):
    kernel_shape_({kernel_h, kernel_w}),
    stride_({stride_h, stride_w}),
    pad_({pad_h, pad_w}),
    conv_in_channels_(conv_in_channels),
    conv_out_channels_(conv_out_channels),
    weights_(),
    time_im2col(0), time_gemm(0)
    {
        // only single-precision weights' memory is allocated at initialization
        // half-precision weights' memory space is allocated when loading the weights
        cudaMalloc((void**)(&weights_float), sizeof(float) * kernel_h * kernel_w * conv_out_channels * conv_in_channels);
    }

conv2d::~conv2d(){
    cudaFree(weights_d);
    cudaFree(weights_float);
}


void conv2d::load_weights(float* weights){
    // weights_float is already allocated
    // weight_d is is allocated in the conversion function
    int num_param = kernel_shape_[0] * kernel_shape_[1] * conv_out_channels_ * conv_in_channels_;
    cudaMemcpy(weights_float, weights, sizeof(float) * num_param, cudaMemcpyHostToDevice);
    weights_d = float2half_matrix(weights_float, num_param);
    }


float* conv2d::forward(float* input_, const int h, const int w, const int num, const bool free_input) {
    // assumes that input is already on device
    
    // compute output shape
    // output_size: the output size of each sample in a batch
    // col_buffer: (kernel_shape_[0] * kernel_shape_[1] * conv_in_channels_, height_out * width_out)
    output_h = (h + 2 * pad_[0] - kernel_shape_[0]) / stride_[0] + 1;
    output_w = (w + 2 * pad_[1] - kernel_shape_[1]) / stride_[1] + 1;
    col_h = kernel_shape_[0] * kernel_shape_[1] * conv_in_channels_;
    output_size = output_h * output_w * conv_out_channels_;
    input_size = h * w * conv_in_channels_;
    
    // convert the input to half precision, and allocate memory for col_buffer and output
    half *input = float2half_matrix(input_, h * w * conv_in_channels_ * num);
    float *output, *tmp;
    cudaMalloc((void**)(&output), sizeof(float) * output_size * num);
    cudaMalloc((void**)(&col_buffer_), sizeof(half) * output_h * output_w * col_h);

    // for each sample, call im2col and perform matrix multiplication
    time_t tic, toc;
    time_im2col = 0;
    time_gemm = 0;
    for(int n=0; n<num; ++n){
        tic = clock();
        im2col(input + input_size * n, conv_in_channels_, h, w,
                kernel_shape_[0], kernel_shape_[1], pad_[0], pad_[1], 
                stride_[0], stride_[1], col_buffer_);
        // col_buffer is filled in column-major order
        toc = clock();
        time_im2col += double(toc-tic)/1000000;
        tic = clock();
        tmp = gemm(weights_d, col_buffer_, conv_out_channels_, col_h, output_h * output_w);
        toc = clock();
        time_gemm += double(toc-tic)/1000000;
        cudaMemcpy(output + output_size * n, tmp, sizeof(float) * output_size, cudaMemcpyDeviceToDevice);
        cudaFree(tmp);
    }
    
    // remember to free the tmporary variables
    cudaFree(col_buffer_);
    cudaFree(input);
    if(free_input)cudaFree(input_);
    return output;
}


float* conv2d::forward_float(float* input_, const int h, const int w, const int num,  const bool free_input) {
    // just for debugging
    // basically the same as the other forward function, but uses cuda core for matrix multiplication
    
    output_h = (h + 2 * pad_[0] - kernel_shape_[0]) / stride_[0] + 1;
    output_w = (w + 2 * pad_[1] - kernel_shape_[1]) / stride_[1] + 1;
    col_h = kernel_shape_[0] * kernel_shape_[1] * conv_in_channels_;
    output_size = output_h * output_w * conv_out_channels_;
    input_size = h * w * conv_in_channels_;
    
    half *input = float2half_matrix(input_, h * w * conv_in_channels_ * num);
    float *output;
    cudaMalloc((void**)(&output), sizeof(float) * output_size * num);
    cudaMalloc((void**)(&col_buffer_), sizeof(half) * output_h * output_w * col_h);

    for(int n=0; n<num; ++n){
        im2col(input + input_size * n, conv_in_channels_, h, w,
                kernel_shape_[0], kernel_shape_[1], pad_[0], pad_[1],
                stride_[0], stride_[1], col_buffer_);
        transpose_half(col_buffer_, output_h * output_w, col_h);
        float *col_buffer_float = half2float_matrix(col_buffer_, col_h * output_h * output_w);

        dim3 dimGrid(ceil(float(conv_out_channels_)/TILE_WIDTH), ceil(float(output_h * output_w)/TILE_WIDTH));
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
        matrix_mul_kernel<<<dimGrid, dimBlock>>>
            (weights_float, col_buffer_float, output + output_size * n, conv_out_channels_, col_h, output_h * output_w);
        cudaFree(col_buffer_float);
    }

    cudaFree(col_buffer_);
    cudaFree(input);
    if(free_input)cudaFree(input_);
    return output;
}
