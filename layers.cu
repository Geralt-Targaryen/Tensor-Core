#include "include/layers.hpp"


__global__ void relu_kernel(float* input, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N && input[i]<0)input[i] = 0;
}


float * relu::forward(float* input, int N){
    relu_kernel<<<(N+255)/256,256>>>(input, N);
    return input;
}


linear::linear(int in_features, int out_features):
in_features_(in_features), out_features_(out_features)
{
    // only single-precision weights' memory is allocated at initialization
    // half-precision weights' memory space is allocated when loading the weights
    cudaMalloc((void**)(&weights_f), sizeof(float) * in_features * out_features);
    cudaMalloc((void**)(&bias_), sizeof(float) * out_features);
}

linear::~linear(){
    cudaFree(weights_f);
    cudaFree(weights_);
    cudaFree(bias_);
}


void linear::load_weights(float* weights, float* bias){
    /*
        weights_float is already allocated
        weight_d is is allocated in the conversion function

        weights are stored col-first, but in PyTorch, weight matrix's shape is (dim_out, dim_in),
        while ours is (dim_in, dim_out), so just read in the weights as usual for half-precision

        however, the single-precision weights need to be transposed back to row-major order
    */
    cudaMemcpy(weights_f, weights, sizeof(float) * in_features_ * out_features_, cudaMemcpyHostToDevice);
    cudaMemcpy(bias_, bias, sizeof(float) * out_features_, cudaMemcpyHostToDevice);
    weights_ = float2half_matrix(weights_f, in_features_ * out_features_);
    transpose(weights_f, out_features_, in_features_);
}


float * linear::forward(float* input_, int num){
    // input: (num, in_features_)
    // output: (num, out_features_)
    
    half *input = float2half_matrix(input_, in_features_ * num);
    float* output;
    
    output = gemm(input, weights_, num, in_features_, out_features_);
    output = linear_bias(output, bias_, num, out_features_);
    
    cudaFree(input);
    cudaFree(input_);
    return output;
}


float * linear::forward_float(float* input, int num){
    // the single-precision version, for debuggin purpose only
    
    float* output;
    cudaMalloc((void**)(&output), sizeof(float) * out_features_ * num);
    dim3 dimGrid(ceil(float(num)/TILE_WIDTH), ceil(float(out_features_)/TILE_WIDTH));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    matrix_mul_kernel<<<dimGrid, dimBlock>>>
        (input, weights_f, output, num, in_features_, out_features_);
    output = linear_bias(output, bias_, num, out_features_);
    
    cudaFree(input);
    return output;
}


maxpool2d::maxpool2d(int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w):
kernel_h_(kernel_h), kernel_w_(kernel_w), stride_h_(stride_h), stride_w_(stride_w), pad_h_(pad_h), pad_w_(pad_w){}


__global__ void MaxPoolForward(
    const int nthreads,
    float* input,
    const int num,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    float* output
    )
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    float maxval = -FLT_MAX;
    int maxidx = -1;
    const float* const bottom_slice =
        input + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    output[index] = maxval;
  }
}



float * maxpool2d::forward(float * input, const int h, const int w, const int c, const int num){
    channels_ = c;
    output_w = static_cast<int>(floor(static_cast<float>(w + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
    output_h = static_cast<int>(floor(static_cast<float>(h + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
    int num_kernels = output_w * output_h * c * num;
    
    float* output;
    cudaMalloc((void**)(&output), sizeof(float) * num_kernels);
    
    MaxPoolForward<<<GET_NUM_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
        num_kernels, input, num, channels_, h, w, output_h, output_w, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, output);
    
    cudaFree(input);
    return output;
}


adaptiveavgpool2d::adaptiveavgpool2d(int out_h, int out_w):
output_h_(out_h), output_w_(out_w) {}


__global__ void AvePoolForward(
    const int nthreads,
    float* input,
    const int num,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    float* top_data)
{
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    float aveval = 0;
    const float* const bottom_slice =
        input + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    top_data[index] = aveval / pool_size;
  }
}


float * adaptiveavgpool2d::forward(float * input, const int h, const int w, const int c, const int num){
    int stride_h_ = h / output_h_;
    int stride_w_ = w / output_w_;
    int kernel_h_ = h - (output_h_ - 1) * stride_h_;
    int kernel_w_ = w - (output_w_ - 1) * stride_w_;
    int num_kernels = output_w_ * output_h_ * c * num;
    
    float* output;
    cudaMalloc((void**)(&output), sizeof(float) * num_kernels);
    
    AvePoolForward<<<GET_NUM_BLOCKS(num_kernels), CUDA_NUM_THREADS>>>(
        num_kernels, input, num, c, h, w, output_h_, output_w_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, 0, 0, output);
    
    cudaFree(input);
    return output;
}


batchnorm2d::batchnorm2d(int channels, float eps):channels_(channels), eps_(eps)
{
    cudaMalloc((void**)(&mean_), sizeof(float) * channels_);
    cudaMalloc((void**)(&variance_), sizeof(float) * channels_);
    cudaMalloc((void**)(&weight_), sizeof(float) * channels_);
    cudaMalloc((void**)(&bias_), sizeof(float) * channels_);
}


void batchnorm2d::load_weights(float* mean, float* variance, float* weights, float* bias){
    cudaMemcpy(weight_, weights, sizeof(float) * channels_, cudaMemcpyHostToDevice);
    cudaMemcpy(mean_, mean, sizeof(float) * channels_, cudaMemcpyHostToDevice);
    cudaMemcpy(variance_, variance, sizeof(float) * channels_, cudaMemcpyHostToDevice);
    cudaMemcpy(bias_, bias, sizeof(float) * channels_, cudaMemcpyHostToDevice);
}


__global__ void batchnorm_kernel(float* input,
                                 float* mean,
                                 float* variance,
                                 float* weight,
                                 float* bias,
                                 int c,
                                 int h,
                                 int w,
                                 int num,
                                 float eps
                                 )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<h*c*num){
        // tid: the row index in the entire input matrix (of a total of num*c*h rows)
        // index is the index of channel
        int index = (tid % (h * c)) / h;
        float *ptr = input + tid * w;
        float m = mean[index], var = variance[index], wei = weight[index], b = bias[index];
        for(int i=0; i<w; ++i,++ptr)
            *ptr = (*ptr - m) / __powf(var + eps, 0.5) * wei + b;
    }
}


float * batchnorm2d::forward(float * input, const int h, const int w, const int num){
    // each kernel deals with one row, so we launch h*c*num kernels
    int num_kernels = h * channels_ * num;
    batchnorm_kernel<<<(num_kernels+255)/256, 256>>>(input, mean_, variance_, weight_, bias_, channels_, h, w, num, eps_);
    return input;
}
