#include "include/sim.hpp"
#include "include/utils.hpp"

inline constexpr unsigned int signal(unsigned int x, unsigned int y){
		return ((x&0x7FFF)>0x7C00) ? (x|0x200) : (y|0x200);
}
constexpr unsigned int rounded(unsigned int value, int g, int s){
		return (value+(g&(s|value)));
}
unsigned int fixed2half(std::uint_fast32_t m, int exp = 14, unsigned int sign = 0, int s = 0){
			if(exp < 0)
				return rounded(sign+(m>>(20-10-exp)), (m>>(20-11-exp))&1, s|((m&((static_cast<std::uint_fast32_t>(1)<<(20-11-exp))-1))!=0));
			return rounded(sign+(exp<<10)+(m>>(20-10)), (m>>(20-11))&1, s|((m&((static_cast<std::uint_fast32_t>(1)<<(20-11))-1))!=0));
}

__half_sim __float2half_sim(const float &a) {
  // Convert floating-point numbers
  // to half-precision floating-point numbers
  std::uint_least32_t fbits;
  std::memcpy(&fbits, &a, sizeof(float));
  unsigned int sign = (fbits>>16) & 0x8000;
  fbits &= 0x7FFFFFFF;
  if(fbits >= 0x7F800000)
    return sign | 0x7C00 | ((fbits>0x7F800000) ? (0x200|((fbits>>13)&0x3FF)) : 0);
  if(fbits >= 0x47800000)
    return (sign|0x7C00);
  if(fbits >= 0x38800000)
    return rounded(sign|(((fbits>>23)-112)<<10)|((fbits>>13)&0x3FF), (fbits>>12)&1, (fbits&0xFFF)!=0);
  if(fbits >= 0x33000000)
  {
    int i = 125 - (fbits>>23);
    fbits = (fbits&0x7FFFFF) | 0x800000;
    return rounded(sign|(fbits>>(i+1)), (fbits>>i)&1, (fbits&((static_cast<std::uint_fast32_t>(1)<<i)-1))!=0);
  }
  if(fbits != 0)
    return sign;
  return sign;
}

float __half2float_sim(const __half_sim &a) {
  // Convert half-precision floating-point numbers
  // to  floating-point numbers
  unsigned int value = a.__x;
  std::uint_least32_t fbits = static_cast<std::uint_least32_t>(value&0x8000) << 16;
  int abs = value & 0x7FFF;
  if(abs)
  {
    fbits |= 0x38000000 << static_cast<unsigned>(abs>=0x7C00);
    for(; abs<0x400; abs<<=1,fbits-=0x800000) ;
    fbits += static_cast<std::uint_least32_t>(abs) << 13;
  }
  float out;
  std::memcpy(&out, &fbits, sizeof(float));
  return out;
}

float operator*(const __half_sim &x, const __half_sim &y) {
  // Overloaded half-precision
  // floating-point number multiplication
  int absx = x.__x & 0x7FFF, absy = y.__x & 0x7FFF, exp = -16;
  unsigned int sign = (x.__x^y.__x) & 0x8000;
  if(absx >= 0x7C00 || absy >= 0x7C00)
    return __half2float_sim(__half_sim((absx>0x7C00 || absy>0x7C00) ? signal(x.__x, y.__x) :
                  ((absx==0x7C00 && !absy)||(absy==0x7C00 && !absx)) ? 0x7FFF : (sign|0x7C00)));
  if(!absx || !absy)
    return __half2float_sim(__half_sim(sign));
  for(; absx<0x400; absx<<=1,--exp) ;
  for(; absy<0x400; absy<<=1,--exp) ;
  std::uint_fast32_t m = static_cast<std::uint_fast32_t>((absx&0x3FF)|0x400) * static_cast<std::uint_fast32_t>((absy&0x3FF)|0x400);
  int i = m >> 21, s = m & i;
  exp += (absx>>10) + (absy>>10) + i;
  if(exp > 29)
    return __half2float_sim(__half_sim(sign|0x7C00));
  else if(exp < -11)
    return __half2float_sim(__half_sim(sign));
  return __half2float_sim(__half_sim(fixed2half(m>>i, exp, sign, s)));
}

GPU::GPU() {
  // Initialize GPU resources reasonably, including regfile size and global
  // memory size, assuming sm=1 and warp_num=1

  // resources to be initialized: unsigned *memory_, unsigned *regfile_, bool *pregfile_
  // assuming only one sub-core in the sm, 1024 regfiles, 32 pregfiles, 1GB = (1024 * 1024 * 1024 / 4 unsigned) global memory
  memory_size = 1024*1024*256;
  regfile_ = new unsigned[1024];
  pregfile_ = new bool[32];
  memory_ = new unsigned[memory_size];
  memory_top = 0;
}


void simMalloc(void **ptr, size_t size, GPU &volta) {
  // sim cudaMalloc
  // Request GPU memory
  if(volta.memory_top + size / sizeof(unsigned) < volta.memory_size){
    *ptr = volta.memory_ + volta.memory_top;
    volta.memory_top += size / sizeof(unsigned);
  }
}

void simMemcpy(void *dst, void *src, size_t count, enum cudaMemcpyKind_sim kind,
               GPU &volta) {
  // sim cudaMemcpy
  // memcpy host memory to class GPU memory or
  // memcpy class GPU memory to host memory
  std::memcpy(dst, src, count);
}



void load_matrix_sync_kernel(__half_sim *dst, __half_sim *src, int stride, int tid){
  // one warp in load_matrix_sync_sim
  if(tid<WARP_SIZE_/2) dst[tid] = src[tid];
  else dst[tid] = src[stride + tid - WARP_SIZE_/2];
}


void load_matrix_sync_sim(__half_sim *dst, __half_sim *src, int stride){
  // dst: 16x16
  // launch one warp at a time
  for(int i=0; i<256/WARP_SIZE_; ++i){
    for(int j=0; j<WARP_SIZE_; ++j)
      load_matrix_sync_kernel(dst+i*WARP_SIZE_, src+i*2*stride, stride, j);
  }
}

void store_matrix_sync_kernel(float *dst, float *src, int stride, int tid){
  // one warp in store_matrix_sync_sim
  if(tid<WARP_SIZE_/2) dst[tid] = src[tid];
  else dst[stride + tid - WARP_SIZE_/2] = src[tid];
}

void store_matrix_sync_sim(float *dst, float *src, int stride){
  // src: 16x16
  // launch one warp at a time
  for(int i=0; i<256/WARP_SIZE_; ++i){
    for(int j=0; j<WARP_SIZE_; ++j)
      store_matrix_sync_kernel(dst+i*2*stride, src+i*WARP_SIZE_, stride, j);
  }
}

void fill_fragment_sim(float *dst, float target){
  for(int i=0;i<16;++i)
    for(int j=0;j<16;++j)dst[i*16+j] = target;
}

void mma_sync_sim_kernel(float *dst, __half_sim *a, __half_sim *b, float *acc, int tid) {
  int row = tid / 16;
  int col = tid % 16;
  float result = 0;
  for(int i=0;i<16;++i)result += a[row*16+i] * b[col*16+i];
  dst[row*16+col] = result + acc[row*16+col];
}

void mma_sync_sim(float *dst, __half_sim *a, __half_sim *b, float *acc) {
  // performs dst = a.dot(b) + acc
  // all 4 arguments are of shape (16, 16)
  // launch one kernel at a time
  for(int tid=0;tid<256;++tid){
    mma_sync_sim_kernel(dst, a, b, acc, tid);
  }
}

void wmma_kernel(__half_sim *a, __half_sim *b, float *c, int M, int N, int K, int m, int k) {
  // device kernel function for computing one target tile
  __half_sim *frag_a = new __half_sim[16*16];
  __half_sim *frag_b = new __half_sim[16*16];
  float *frag_c = new float[16*16];
  fill_fragment_sim(frag_c, 0.0);

  for(int n=0; n<N; n+=16){
    load_matrix_sync_sim(frag_a, a + m * N + n, N);
    load_matrix_sync_sim(frag_b, b + k * N + n, N);
    mma_sync_sim(frag_c, frag_a, frag_b, frag_c);
  }
  
  store_matrix_sync_sim(c + m * K + k, frag_c, K);
}

void wmma_sim_wrapper(__half_sim *a, __half_sim *b, float *c, int M, int N, int K){
  // M, N, K are all multiples of 16
  // a stored in row-major order, b is transposed (col-major)
  
  // iterate through each target tile (simulating multiple warps)
  assert(M%16==0 && N%16==0 && K%16==0);
  for(int i=0; i<M; i+=16){
    for(int j=0; j<K; j+=16){
      wmma_kernel(a, b, c, M, N, K, i, j);
    }
  }
  /*//傻瓜式矩阵乘法, for debugging
  for(int i=0;i<M;++i){
    for(int k=0;k<K;++k){
      c[i*K+k] = 0;
      for(int j=0;j<N;++j)c[i*K+k] += a[i*N+j] * b[k*N+j];
    }
  }
  print_matrix(c, M, K, 1, false);
  assert(0);
  */
}

void pad(half *a, __half_sim *a_, int M, int N, int M_, int N_){
  // pad the input matrix to multiples of 16,
  // and also convert to __half
  for(int i=0;i<M_;++i)
    for(int j=0;j<N_;++j){
      if(i<M&&j<N)
        a_[i * N_ + j] = __float2half_sim(__half2float(a[i * N + j]));
      else 
        a_[i * N_ + j] = __half_sim(0);
    }
      
}

// un-pad the output matrix
void unpad(float *a, float *a_, int M, int N, int M_, int N_){
    for(int i=0;i<M_;++i)
    for(int j=0;j<N_;++j){
      if(i<M&&j<N) a[i*N+j] = a_[i*N_+j];
    }
}

float* gemm_sim(half *a_d, half *b_d, int M, int N, int K) {
  // host function gemm c = a.dot(b)
  // a_d, b_d are assumed to be on the real cuda device
  // we first move them to cpu and convert them to float for simulation

  half *a = new half[M*N], *b = new half[N*K];
  cudaMemcpy(a, a_d, sizeof(half)*M*N, cudaMemcpyDeviceToHost);
  cudaMemcpy(b, b_d, sizeof(half)*K*N, cudaMemcpyDeviceToHost);

  // pad a, b to be multiples of 16 (and convert them to simulated __half)
  int M_ = M%16==0? M : (M/16+1)*16;
  int N_ = N%16==0? N : (N/16+1)*16;
  int K_ = K%16==0? K : (K/16+1)*16;

  __half_sim *a_ = new __half_sim[M_*N_], *b_ = new __half_sim[N_*K_];
  float *c_ = new float[M_*K_], *c = new float[M*K], *c_d;
  pad(a, a_, M, N, M_, N_);
  pad(b, b_, K, N, K_, N_);

  // multiplication
  wmma_sim_wrapper(a_, b_, c_, M_, N_, K_);

  // un-pad c, move to the real device
  unpad(c, c_, M, K, M_, K_);
  cudaMalloc((void**)(&c_d), sizeof(float) * M*K);
  cudaMemcpy(c_d, c, sizeof(float)*M*K, cudaMemcpyHostToDevice);

  // garbage collection
  delete [] a;
  delete [] b;
  delete [] c;
  delete [] a_;
  delete [] b_;
  delete [] c_;

  return c_d;
}


void print_matrix_half__(const __half_sim* x, int h, int w, int c){
      for(int k=0; k<c;++k){
          for(int i=0; i<h; ++i){
              for(int j=0;j<w;++j)cout<<__half2float_sim(x[k*h*w+i*w+j])<<"\t";
              cout<<endl;
          }
          cout<<endl;
      }
    cout<<endl;
}
