#include "include/utils.hpp"

__global__ void transpose_half_kernel(half *a, half *aT, int M, int N){
    // store the transpose of matrix a (M, N) to aT of (N, M)
    // i.e. aT[j,i] = a[i,j]
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i<M&&j<N){
        aT[j*M+i] = a[i*N+j];
    }
}

void transpose_half(half *a, int M, int N){
    // the transposition is performed in-place
    if(M==1||N==1)return;
    half *aT;
    cudaMalloc((void**)(&aT), sizeof(half) * M * N);
    dim3 blockDim(16,16), gridDim((M+15)/16, (N+15)/16);
    transpose_half_kernel<<<gridDim, blockDim>>>(a, aT, M, N);
    cudaMemcpy(a, aT, sizeof(half) * M * N, cudaMemcpyDeviceToDevice);
    cudaFree(aT);
}

__global__ void transpose_kernel(float *a, float *aT, int M, int N){
    // store the transpose of matrix a (M, N) to aT of (N, M)
    // i.e. aT[j,i] = a[i,j]
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i<M&&j<N){
        aT[j*M+i] = a[i*N+j];
    }
}

void transpose(float *a, int M, int N){
    // the transposition is performed in-place
    if(M==1||N==1)return;
    float *aT;
    cudaMalloc((void**)(&aT), sizeof(float) * M * N);
    dim3 blockDim(16,16), gridDim((M+15)/16, (N+15)/16);
    transpose_kernel<<<gridDim, blockDim>>>(a, aT, M, N);
    cudaMemcpy(a, aT, sizeof(float) * M * N, cudaMemcpyDeviceToDevice);
    cudaFree(aT);
}

// conversion between float matrix and half matrix
__global__ void float2half_matrix_kernel(const float *input, half *output, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N)output[i] = __float2half(input[i]);
}

__global__ void half2float_matrix_kernel(const half *input, float *output, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N)output[i] = __half2float(input[i]);
}

half* float2half_matrix(const float *input, int N){
    half *output;
    cudaMalloc((void**)(&output), sizeof(half) * N);
    float2half_matrix_kernel<<<(N+255)/256,256>>>(input, output, N);
    return output;
}

float* half2float_matrix(const half *input, int N){
    float *output;
    cudaMalloc((void**)(&output), sizeof(float) * N);
    half2float_matrix_kernel<<<(N+255)/256,256>>>(input, output, N);
    return output;
}


__global__ void linear_bias_kernel(float* input1, float* input2, int h, int w){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bias_index = tid % w;
    if(tid<h)
        for(int i=0;i<w;++i)
            input1[i] = input1[i] + input2[bias_index];
}

float * linear_bias(float* input1, float* input2, int h, int w){
    // input2, i.e. the bias vector, needs to be broadcast, but it's done virtually
    // launch one kernel for each row (i.e. one feature dimension across all samples)
    linear_bias_kernel<<<(h+255)/256,256>>>(input1, input2, h, w);
    return input1;
}

// this is for the residual connection
__global__ void add_matrix_kernel(float* input1, float* input2, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i<N)input1[i] = input1[i] + input2[i];
}

float * add_matrix(float* input1, float* input2, int N){
    add_matrix_kernel<<<(N+255)/256,256>>>(input1, input2, N);
    return input1;
}

// argmax
__global__ void argmax_kernel(float* input, int num, int d, int *result){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid<num){
        float max_val = -FLT_MAX, *ptr = input + tid * d;
        int max_index = -1;
        for(int i=0;i<d;++i,++ptr){
            if(*ptr>max_val){
                max_val = *ptr;
                max_index = i;
            }
        }
        result[tid] = max_index;
    }
}

int * argmax(float* input, int num, int d){
    int *result;
    cudaMalloc((void**)(&result), sizeof(int) * num);
    argmax_kernel<<<(num+255)/256,256>>>(input, num, d, result);
    return result;
}

// accuracy score, a bit silly using cuda here
float accuracy_score(int *label, int *label_, int N){
    int acc=0;
    for(int i=0;i<N;++i){
        if(label[i]==label_[i])++acc;
    }
    return (float)acc/N;
}

// the rest are all auxiliary functions
void load_vector(vector<float> &v, const char *file){
    ifstream input_file(file);
    stringstream input_stream;
    input_stream << input_file.rdbuf();
    input_file.close();
    float x;
    while(input_stream >> x){
        v.push_back(x);
    }
}

void load_vector_int(vector<int> &v, const char *file){
    ifstream input_file(file);
    stringstream input_stream;
    input_stream << input_file.rdbuf();
    input_file.close();
    int x;
    while(input_stream >> x){
        v.push_back(x);
    }
}

void load_binary(vector<float> &v, const char *file, int size){
    ifstream f(file, ios::in|ios::binary);
    f.read((char*)v.data(), sizeof(float) * size);
    f.close();
}
void load_binary(vector<int> &v, const char *file, int size){
    ifstream f(file, ios::in|ios::binary);
    f.read((char*)v.data(), sizeof(int) * size);
    f.close();
}

void print_matrix_half(const half* x, int h, int w, int c, bool device){
    if(!device)
        for(int k=0; k<c;++k){
            for(int i=0; i<h; ++i){
                for(int j=0;j<w;++j)cout<<__half2float(x[k*h*w+i*w+j])<<"\t";
                cout<<endl;
            }
            cout<<endl;
        }
    else{
        vector<half> x_(h*w*c);
        cudaMemcpy(x_.data(), x, sizeof(half)*h*w*c, cudaMemcpyDeviceToHost);
        for(int k=0; k<c;++k){
            for(int i=0; i<h; ++i){
                for(int j=0;j<w;++j)cout<<__half2float(x_[k*h*w+i*w+j])<<"\t";
                cout<<endl;
            }
        }
    }
    cout<<endl;
}

void print_matrix(const float* x, int h, int w, int c, bool device){
    if(!device)
        for(int k=0; k<c;++k){
            for(int i=0; i<h; ++i){
                for(int j=0;j<w;++j)cout<<x[k*h*w+i*w+j]<<"\t";
                cout<<endl;
            }
            cout<<endl;
        }
    else{
        vector<float> x_(h*w*c);
        cudaMemcpy(x_.data(), x, sizeof(float)*h*w*c, cudaMemcpyDeviceToHost);
        for(int k=0; k<c;++k){
            for(int i=0; i<h; ++i){
                for(int j=0;j<w;++j)cout<<x_[k*h*w+i*w+j]<<"\t";
                cout<<endl;
            }
        }
    }
    cout<<endl;
}

void print_matrix(const int* x, int h, int w, int c, bool device){
    if(!device)
        for(int k=0; k<c;++k){
            for(int i=0; i<h; ++i){
                for(int j=0;j<w;++j)cout<<x[k*h*w+i*w+j]<<"\t";
                cout<<endl;
            }
            cout<<endl;
        }
    else{
        vector<int> x_(h*w*c);
        cudaMemcpy(x_.data(), x, sizeof(int)*h*w*c, cudaMemcpyDeviceToHost);
        for(int k=0; k<c;++k){
            for(int i=0; i<h; ++i){
                for(int j=0;j<w;++j)cout<<x_[k*h*w+i*w+j]<<"\t";
                cout<<endl;
            }
        }
    }
    cout<<endl;
}