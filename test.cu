#include <iostream>
#include <vector>
#include "include/utils.hpp"
#include "include/resnet.hpp"
using namespace std;


int main(int argc, char *argv[]){

    int h=224, w=224, c=3, N=5000, bs = argc>1? atoi(argv[1]):8;
    bool forward_with_float=false;
    int h_=1, w_=5000, c_=1;
    int input_size = h * w * c, output_size = h_ * w_ * c_;
    double time_argmax=0;
    time_t tic, toc, tic1, toc1;
    
    resnet18 model(forward_with_float);
    model.load_weights("input/param.bin");

    // load images
    vector<float> input(input_size * N);
    vector<int> label(N), label_(N);
    cout<<"Loading data..."<<endl;
    load_binary(input, "input/data.bin", input_size * N);
    load_binary(label, "input/label.bin", N);
    printf("Loaded data of shape (%d, %d, %d, %d)\n", (int)input.size()/(h*w*c), c, h, w);

    float *d_input, *d_output;
    cudaMalloc((void**)(&d_input), sizeof(float) * input_size * bs);

    cout<<"Starting inference with batch size "<<bs<<"..."<<endl;
    tic = time(NULL);
    for(int i=0; i<N; i+=bs){
        int bs_ = min(bs, N-i);
        cudaMemcpy(d_input, input.data() + i * input_size, sizeof(float) * input_size * bs_, cudaMemcpyHostToDevice);

        d_output = model.forward(d_input, h, w, bs_);
        tic1 = clock();
        int *prediction = argmax(d_output, bs_, 1000);
        toc1 = clock();
        time_argmax += double(toc1-tic1)/1000000;
        cudaMemcpy(label_.data() + i, prediction, sizeof(int) * bs_, cudaMemcpyDeviceToHost);
        cudaFree(prediction);
        cudaFree(d_output);
        
    }
    toc = time(NULL);
    model.compute_time();

    cudaFree(d_input);

    cout<<"time - conv2d: "<<model.time_conv2d<<"s\n";
    cout<<"\ttime - im2col: "<<model.time_im2col<<"s\n";
    cout<<"\ttime - gemm: "<<model.time_gemm<<"s\n";
    cout<<"time - batchnorm2d: "<<model.time_batchnorm2d<<"s\n";
    cout<<"time - relu: "<<model.time_relu<<"s\n";
    cout<<"time - maxpool2d: "<<model.time_maxpool2d<<"s\n";
    cout<<"time - adaptiveavgpool2d: "<<model.time_adaptiveavgpool2d<<"s\n";
    cout<<"time - linear: "<<model.time_linear<<"s"<<endl;
    cout<<"time - argmax: "<<time_argmax<<"s\n"<<endl;
    
    cout<<"Inference time: "<<difftime(toc, tic)<<"s"<<endl;
    cout<<"Test acc: "<<accuracy_score(label.data(), label_.data(), N)<<endl;
}
