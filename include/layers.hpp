#ifndef LAYERS_HPP_
#define LAYERS_HPP_

#include "utils.hpp"
#include "matrix.hpp"
#include <cfloat>
#include <algorithm>

/*
    this header declares the other layers (besides conv2d)
*/


class relu{
public:
    float * forward(float* input, int N);
};


/*
    as in conv2d, two copies of weights are stored, one in single precision (for debuggin)
    and one in half precision, and two forward functions are defined
*/
class linear{
public:
    // weights: (d_in, d_out), bias: (1, d_out)
    half * weights_;
    float * weights_f;
    float * bias_;
    
public:
    int in_features_;
    int out_features_;
    
    linear(int in_features, int out_features);
    ~linear();
    void load_weights(float* weights, float* bias);
    float * forward(float* input, int num);
    float * forward_float(float* input, int num);
};


// max pool and avg pool, partly borrowed from caffe
class maxpool2d {
private:
    int kernel_h_;
    int kernel_w_;
    int stride_h_;
    int stride_w_;
    int pad_h_;
    int pad_w_;
    
    int channels_;
    int output_h;
    int output_w;
    
public:
    maxpool2d(int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w);
    float * forward(float * input, const int h, const int w, const int c, const int num);
    
    int get_output_w(){return output_w;}
    int get_output_h(){return output_h;}
};


class adaptiveavgpool2d {
private:
    int output_h_;
    int output_w_;
    
public:
    adaptiveavgpool2d(int out_h, int out_w);
    float * forward(float * input, const int h, const int w, const int c, const int num);
};


class batchnorm2d {
private:
    float *mean_, *variance_, *weight_, *bias_;
    int channels_;
    float eps_;
public:
    batchnorm2d(int channels, float eps);
    void load_weights(float* mean, float* variance, float* weights, float* bias);
    float * forward(float *input, const int h, const int w, const int num);
};

#endif
