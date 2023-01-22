#include "matrix.hpp"
#include "im2col.hpp"
#include "conv.hpp"
#include "layers.hpp"
#include "utils.hpp"

/*
    this header defines the basic block of resnet and the resnet18 network
    for resnet18, the only interface left to caller is the debugging option
    of forwarding with single-precision computation only, and the rest 
    hyper-parameters are all hard-coded
*/

// basic block: the constructor, weight loading, and forward functions just
// call each layers' corresponding functions
class Block {
private:
    conv2d conv1;
    batchnorm2d bn1;
    relu relu1;
    conv2d conv2;
    batchnorm2d bn2;
    conv2d conv_down;
    batchnorm2d bn_down;
    
    bool down_;
    int num_param;
    int out_c_;
    int in_c_;
    bool forward_with_float_;
    
public:
    double time_conv2d;
    double time_batchnorm2d;
    double time_relu;
    double time_im2col;
    double time_gemm;

    Block(int in_c, int out_c, int s, bool down, bool forward_with_float=true):
    conv1(3,3,s,s,1,1, in_c, out_c),
    bn1(out_c, 1e-5),
    relu1(),
    conv2(3,3,1,1,1,1, out_c, out_c),
    bn2(out_c, 1e-5),
    conv_down(1,1,2,2,0,0, in_c, out_c),
    bn_down(out_c, 1e-5),
    out_c_(out_c),
    in_c_(in_c),
    down_(down),
    forward_with_float_(forward_with_float),
    time_conv2d(0), time_batchnorm2d(0), time_relu(0),
    time_im2col(0), time_gemm(0)
    {
        num_param = 9*in_c*out_c + 4*out_c + 9*out_c*out_c + 4*out_c + (down ? in_c*out_c + 4*out_c : 0);
    }
    
    void load_weights(float *weight){
        float * tmp;
        conv1.load_weights(weight);
        tmp = weight + 9*in_c_*out_c_;
        bn1.load_weights(tmp, tmp+out_c_, tmp+2*out_c_, tmp+3*out_c_);
        conv2.load_weights(tmp + out_c_*4);
        tmp = tmp + out_c_*4 + 9*out_c_*out_c_;
        bn2.load_weights(tmp, tmp+out_c_, tmp+2*out_c_, tmp+3*out_c_);
        if(down_){
            conv_down.load_weights(tmp + out_c_*4);
            tmp = tmp + out_c_*4 + in_c_*out_c_;
            bn_down.load_weights(tmp, tmp+out_c_, tmp+2*out_c_, tmp+3*out_c_);
        }
    }
    
    float * forward(float * input, int h, int w, int num){
        float *output, *identity=input;
        time_t tic, toc;
        if(forward_with_float_){
            // using single-precision operations for the forward call of conv and fc
            output = conv1.forward_float(input, h, w, num, false);
            output = bn1.forward(output, conv1.get_output_h(), conv1.get_output_w(), num);
            output = relu1.forward(output, conv1.get_output_h()*conv1.get_output_w()*num*out_c_);
            output = conv2.forward_float(output, conv1.get_output_h(), conv1.get_output_w(), num);
            output = bn2.forward(output, conv2.get_output_h(), conv2.get_output_w(), num);
            if(down_){
                identity = conv_down.forward_float(identity, h, w, num);
                identity = bn_down.forward(identity, conv_down.get_output_h(), conv_down.get_output_w(), num);
            }
        }
        else{
            // using half-precision
            tic = clock();
            output = conv1.forward(input, h, w, num, false);
            toc = clock();
            time_conv2d += double(toc-tic)/1000000;
            time_im2col += conv1.time_im2col;
            time_gemm += conv1.time_gemm;
            tic = clock();
            output = bn1.forward(output, conv1.get_output_h(), conv1.get_output_w(), num);
            toc = clock();
            time_batchnorm2d += double(toc-tic)/1000000;
            tic = clock();
            output = relu1.forward(output, conv1.get_output_h()*conv1.get_output_w()*num*out_c_);
            toc = clock();
            time_relu += double(toc-tic)/1000000;
            tic = clock();
            output = conv2.forward(output, conv1.get_output_h(), conv1.get_output_w(), num);
            toc = clock();
            time_conv2d += double(toc-tic)/1000000;
            time_im2col += conv2.time_im2col;
            time_gemm += conv2.time_gemm;
            tic = clock();
            output = bn2.forward(output, conv2.get_output_h(), conv2.get_output_w(), num);
            toc = clock();
            time_batchnorm2d += double(toc-tic)/1000000;
            if(down_){
                tic = clock();
                identity = conv_down.forward(identity, h, w, num);
                toc = clock();
                time_conv2d += double(toc-tic)/1000000;
                time_im2col += conv_down.time_im2col;
                time_gemm += conv_down.time_gemm;
                tic = clock();
                identity = bn_down.forward(identity, conv_down.get_output_h(), conv_down.get_output_w(), num);
                toc = clock();
                time_batchnorm2d += double(toc-tic)/1000000;
            }
        }
        
        // residual connection
        output = add_matrix(output, identity, conv2.get_output_h()*conv2.get_output_w()*num*out_c_);
        tic = clock();
        output = relu1.forward(output, conv2.get_output_h()*conv2.get_output_w()*num*out_c_);
        toc = clock();
        time_relu += double(toc-tic)/1000000;
        if(!down_)cudaFree(identity);
        return output;
    }
    
    int get_output_h(){return conv2.get_output_h();}
    int get_output_w(){return conv2.get_output_w();}
    int get_output_c(){return out_c_;}
    int get_num_param(){return num_param;}
};


// all that's left is just putting together the blocks
class resnet18 {
public:
    conv2d conv1;
    batchnorm2d bn1;
    relu relu1;
    maxpool2d maxpool;
    
    Block block1_1;
    Block block1_2;
    
    Block block2_1;
    Block block2_2;
    
    Block block3_1;
    Block block3_2;
    
    Block block4_1;
    Block block4_2;
    
    adaptiveavgpool2d avgpool;
    linear fc;
    
    int num_param;
    bool forward_with_float_;
    
public:
    double time_conv2d;
    double time_batchnorm2d;
    double time_relu;
    double time_maxpool2d;
    double time_adaptiveavgpool2d;
    double time_linear;
    double time_im2col;
    double time_gemm;

    resnet18(bool forward_with_float=true):
    conv1(7,7,2,2,3,3,3,64),
    bn1(64, 1e-5),
    relu1(),
    maxpool(3,3,2,2,1,1),
    
    block1_1(64,64,1,false, forward_with_float),
    block1_2(64,64,1,false, forward_with_float),
    
    block2_1(64,128,2,true, forward_with_float),
    block2_2(128,128,1,false, forward_with_float),
    
    block3_1(128,256,2,true, forward_with_float),
    block3_2(256,256,1,false, forward_with_float),
    
    block4_1(256,512,2,true, forward_with_float),
    block4_2(512,512,1,false, forward_with_float),
    
    avgpool(1,1),
    fc(512,1000),

    forward_with_float_(forward_with_float),
    time_conv2d(0), time_relu(0), time_batchnorm2d(0), time_adaptiveavgpool2d(0),
    time_maxpool2d(0), time_linear(0), time_gemm(0), time_im2col(0)
    {}
    
    void load_weights(const char *file){
        printf("Loading weights of ResNet18...\n");
        vector<float> params(11699112);
        load_binary(params, file, 11699112);
        float *weight=params.data(), *tmp;
        conv1.load_weights(weight);
        tmp = weight + 49*3*64;
        bn1.load_weights(tmp, tmp+64, tmp+2*64, tmp+3*64);
        tmp = tmp + 64*4;
        
        block1_1.load_weights(tmp);
        tmp += block1_1.get_num_param();
        block1_2.load_weights(tmp);
        tmp += block1_2.get_num_param();
        
        block2_1.load_weights(tmp);
        tmp += block2_1.get_num_param();
        block2_2.load_weights(tmp);
        tmp += block2_2.get_num_param();
        
        block3_1.load_weights(tmp);
        tmp += block3_1.get_num_param();
        block3_2.load_weights(tmp);
        tmp += block3_2.get_num_param();
        
        block4_1.load_weights(tmp);
        tmp += block4_1.get_num_param();
        block4_2.load_weights(tmp);
        tmp += block4_2.get_num_param();
        
        fc.load_weights(tmp, tmp+512*1000);
        
        num_param = tmp+512*1000+1000-weight;
        // sanity check
        assert(11699112==num_param);
        printf("Loaded %d parameters.\n", num_param);
    }
    
    float * forward(float * input, int h, int w, int num){
        float *output;
        time_t tic, toc;

        tic = clock();
        if(forward_with_float_)output = conv1.forward_float(input, h, w, num, false);
        else output = conv1.forward(input, h, w, num, false);
        toc = clock();
        time_conv2d += double(toc-tic)/1000000;
        time_im2col += conv1.time_im2col;
        time_gemm += conv1.time_gemm;
        tic = clock();
        output = bn1.forward(output, conv1.get_output_h(), conv1.get_output_w(), num);
        toc = clock();
        time_batchnorm2d += double(toc-tic)/1000000;
        tic = clock();
        output = relu1.forward(output, conv1.get_output_h()*conv1.get_output_w()*num*conv1.get_output_c());
        toc = clock();
        time_relu += double(toc-tic)/1000000;
        tic = clock();
        output = maxpool.forward(output, conv1.get_output_h(), conv1.get_output_w(), conv1.get_output_c(), num);
        toc = clock();
        time_maxpool2d += double(toc-tic)/1000000;
        
        output = block1_1.forward(output, maxpool.get_output_h(), maxpool.get_output_w(), num);
        output = block1_2.forward(output, block1_1.get_output_h(), block1_1.get_output_w(), num);
        
        output = block2_1.forward(output, block1_2.get_output_h(), block1_2.get_output_w(), num);
        output = block2_2.forward(output, block2_1.get_output_h(), block2_1.get_output_w(), num);
        
        output = block3_1.forward(output, block2_2.get_output_h(), block2_2.get_output_w(), num);
        output = block3_2.forward(output, block3_1.get_output_h(), block3_1.get_output_w(), num);
        
        output = block4_1.forward(output, block3_2.get_output_h(), block3_2.get_output_w(), num);
        output = block4_2.forward(output, block4_1.get_output_h(), block4_1.get_output_w(), num);
        
        tic = clock();
        output = avgpool.forward(output, block4_2.get_output_h(), block4_2.get_output_w(), block4_2.get_output_c(), num);
        toc = clock();
        time_adaptiveavgpool2d += double(toc-tic)/1000000;
        tic = clock();
        if(forward_with_float_)output = fc.forward_float(output, num);
        else output = fc.forward(output, num);
        toc = clock();
        time_linear += double(toc-tic)/1000000;
        tic = clock();
        
        return output;
    }
    
    int get_num_param(){return num_param;}
    void compute_time(){
        time_conv2d += (block1_1.time_conv2d + block1_2.time_conv2d + block2_1.time_conv2d + block2_2.time_conv2d +
                        block3_1.time_conv2d + block3_2.time_conv2d + block4_1.time_conv2d + block4_2.time_conv2d);
        time_batchnorm2d += (block1_1.time_batchnorm2d + block1_2.time_batchnorm2d + block2_1.time_batchnorm2d + block2_2.time_batchnorm2d + 
                             block3_1.time_batchnorm2d + block3_2.time_batchnorm2d + block4_1.time_batchnorm2d + block4_2.time_batchnorm2d);
        time_relu += (block1_1.time_relu + block1_2.time_relu + block2_1.time_relu + block2_2.time_relu + 
                      block3_1.time_relu + block3_2.time_relu + block4_1.time_relu + block4_2.time_relu);
        time_im2col += (block1_1.time_im2col + block1_2.time_im2col + block2_1.time_im2col + block2_2.time_im2col +
                        block3_1.time_im2col + block3_2.time_im2col + block4_1.time_im2col + block4_2.time_im2col);
        time_gemm += (block1_1.time_gemm + block1_2.time_gemm + block2_1.time_gemm + block2_2.time_gemm +
                      block3_1.time_gemm + block3_2.time_gemm + block4_1.time_gemm + block4_2.time_gemm);
    }
};
