#ifndef CONVOLUTION_LAYER_HPP_
#define CONVOLUTION_LAYER_HPP_

#include "utils.hpp"
#include "matrix.hpp"
#include "im2col.hpp"
#include "sim.hpp"

/*
    the convolution layer, specifically tailored for ResNet18
    i.e. no bias, no dilation, no grouping 
    (honestly I don't really understand the concept of grouping)

    note that for the purpose of debugging and comparison,
    we store two copies of kernel weights, one in single precision 
    and one in half precision, and defines two forward functions
*/

class conv2d{
private:
    vector<half> weights_;
    float * weights_float;
    half* weights_d;
    half* col_buffer_;
    
    // layer hyper-parameters, determined at initilization
    int conv_out_channels_;
    int conv_in_channels_;
    vector<int> kernel_shape_;
    vector<int> stride_;
    vector<int> pad_;
    
    // data-specific parameters
    int input_size;
    int output_size;
    int output_h;
    int output_w;
    int col_h;

public:
    conv2d(const int kernel_h,
            const int kernel_w,
            const int stride_h,
            const int stride_w,
            const int pad_h,
            const int pad_w,
            const int conv_in_channels,
            const int conv_out_channels
           );
    ~conv2d();
    
    void load_weights(float* weights);
    float* forward(float* input, const int h, const int w, const int num, const bool free_input=true);
    float* forward_float(float* input, const int h, const int w, const int num, const bool free_input=true);
    int get_output_h(){return output_h;}
    int get_output_w(){return output_w;}
    int get_output_c(){return conv_out_channels_;}
    int get_output_size(){return output_size;}

    double time_im2col;
    double time_gemm;
    
};


#endif



