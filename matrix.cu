#include "include/matrix.hpp"


__global__ void wmma_kernel(half* a, half* b, float* c, int M, int N, int K) {
    /*
        M, N, K are made sure to be multiples of 16
        b is stored in column-major order
        each warp fills one tile in the target matrix
    */
    int row = blockIdx.x*16;
    int col = blockIdx.y*16;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    
    // iterate through the tiles in the source matrices, accumulate the results in c_frag
    for(int n=0; n<N; n+=16){
        wmma::load_matrix_sync(a_frag, a + row * N + n, N);
        wmma::load_matrix_sync(b_frag, b + col * N + n, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // write to the target
    wmma::store_matrix_sync(c + row * K + col, c_frag, K, wmma::mem_row_major);
}


// pad the input matrix to multiples of 16
__global__ void fill_matrix(half *a, half *a_, int M, int N, int M_, int N_){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i<M){
        if(j<N) a_[i * N_ + j] = a[i * N + j];
        else if(j<N_) a_[i * N_ + j] = __float2half(0);
    }
    else if(i<M_&&j<N) a_[i * N_ + j] = __float2half(0);
}

// un-pad the output matrix
__global__ void extract_matrix(float *a, float *a_, int M, int N, int M_, int N_){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if(i<M&&j<N) a[i*N+j] = a_[i*N_+j];
}


// general matrix multiplication, to be used by network layers (conv and fc)
float* gemm(half* a, half* b, int M, int N, int K){
    // assumes a, b are both on device
    // prepare the memory
    int M_ = M%16==0? M : (M/16+1)*16;
    int N_ = N%16==0? N : (N/16+1)*16;
    int K_ = K%16==0? K : (K/16+1)*16;
    
    half *a_, *b_;
    float *c_, *c;
    cudaMalloc((void**)(&a_), sizeof(half) * M_*N_);
    cudaMalloc((void**)(&b_), sizeof(half) * N_*K_);
    cudaMalloc((void**)(&c_), sizeof(float) * M_*K_);
    cudaMalloc((void**)(&c), sizeof(float) * M*K);
    
    // pad a, b to be multiples of 16
    dim3 blockDim(16,16), gridDim(ceil(float(M)/16), ceil(float(N)/16));
    fill_matrix<<<gridDim, blockDim>>>(a, a_, M, N, M_, N_);
    gridDim.y = ceil(float(N)/16);
    gridDim.x = ceil(float(K)/16);
    fill_matrix<<<gridDim, blockDim>>>(b, b_, K, N, K_, N_);
    
    // multiplication
    gridDim.x = M_/16;
    gridDim.y = K_/16;
    wmma_kernel<<<gridDim, blockDim>>>(a_, b_, c_, M_, N_, K_);
    
    // un-pad c
    gridDim.x = ceil(float(M)/16);
    gridDim.y = ceil(float(K)/16);
    extract_matrix<<<gridDim, blockDim>>>(c, c_, M, K, M_, K_);
    
    cudaFree(a_);
    cudaFree(b_);
    cudaFree(c_);
    return c;
}


// the following two functions are matrix multiplication using cuda cores (instead of tensor cores)
// used for debugging only
__global__ void matrix_mul_kernel(float *Ad, float *Bd, float *Cd, int M, int N, int K){
    __shared__ float Ads[TILE_WIDTH*TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH*TILE_WIDTH];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.x*TILE_WIDTH + tx;
    int col = blockIdx.y*TILE_WIDTH + ty;
    
    int n=0;
    float sum=0;
    // process all the complete blocks
    for(;n<N/TILE_WIDTH;++n){
        // load a tile from A and B into shared memory
        Ads[tx*TILE_WIDTH+ty] = Ad[row*N+(n*TILE_WIDTH+ty)];
        Bds[tx*TILE_WIDTH+ty] = Bd[(n*TILE_WIDTH+tx)*K+col];
        __syncthreads();
        
        // comopute a tile of C
        for(int n_=0;n_<TILE_WIDTH;++n_){
            sum += Ads[tx*TILE_WIDTH+n_]*Bds[n_*TILE_WIDTH+ty];
        }
        __syncthreads();
    }
    
    // deal with the remaining columns of A and rows of B, in case the width of A and the height of B isn't a multiple of tile width
    if(n*TILE_WIDTH+ty<N)
        Ads[tx*TILE_WIDTH+ty] = Ad[row*N+(n*TILE_WIDTH+ty)];
    if(n*TILE_WIDTH+tx<N)
        Bds[tx*TILE_WIDTH+ty] = Bd[(n*TILE_WIDTH+tx)*K+col];
    __syncthreads();
    
    for(int n_=0;n_<N-n*TILE_WIDTH;++n_){
        sum += Ads[tx*TILE_WIDTH+n_]*Bds[n_*TILE_WIDTH+ty];
    }
    __syncthreads();
    
    // write the result to C
    if(row<M&&col<K)
        Cd[row*K+col]=sum;
}

__global__ void matrix_mul_kernel_half(half *Ad, half *Bd, half *Cd, int M, int N, int K){
    __shared__ half Ads[TILE_WIDTH*TILE_WIDTH];
    __shared__ half Bds[TILE_WIDTH*TILE_WIDTH];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.x*TILE_WIDTH + tx;
    int col = blockIdx.y*TILE_WIDTH + ty;
    
    int n=0;
    half sum=0;
    // process all the complete blocks
    for(;n<N/TILE_WIDTH;++n){
        // load a tile from A and B into shared memory
        Ads[tx*TILE_WIDTH+ty] = Ad[row*N+(n*TILE_WIDTH+ty)];
        Bds[tx*TILE_WIDTH+ty] = Bd[(n*TILE_WIDTH+tx)*K+col];
        __syncthreads();
        
        // comopute a tile of C
        for(int n_=0;n_<TILE_WIDTH;++n_){
            sum += Ads[tx*TILE_WIDTH+n_]*Bds[n_*TILE_WIDTH+ty];
        }
        __syncthreads();
    }
    
    // deal with the remaining columns of A and rows of B, in case the width of A and the height of B isn't a multiple of tile width
    if(n*TILE_WIDTH+ty<N)
        Ads[tx*TILE_WIDTH+ty] = Ad[row*N+(n*TILE_WIDTH+ty)];
    if(n*TILE_WIDTH+tx<N)
        Bds[tx*TILE_WIDTH+ty] = Bd[(n*TILE_WIDTH+tx)*K+col];
    __syncthreads();
    
    for(int n_=0;n_<N-n*TILE_WIDTH;++n_){
        sum += Ads[tx*TILE_WIDTH+n_]*Bds[n_*TILE_WIDTH+ty];
    }
    __syncthreads();
    
    // write the result to C
    if(row<M&&col<K)
        Cd[row*K+col]=sum;
}

