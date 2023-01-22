#ifndef SIM_HPP_
#define SIM_HPP_
#include <stdint.h>
#include <string.h>
#include <cstring>
#include "utils.hpp"

const unsigned WARP_SIZE_ = 32;
enum cudaMemcpyKind_sim {
  MemcpyHostToDevice = 0, /**< Host   -> Device */
  MemcpyDeviceToHost = 1  /**< Device -> Host */
};

enum s_reg_t { SRZ = 0, SR_LAINID, SR_TID_X, SR_TID_Y, SR_CTAID_X, SR_CTAID_Y };

struct dim3_sim {
  unsigned int x, y, z;
  constexpr dim3_sim(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1)
      : x(vx), y(vy), z(vz) {}
};

struct __half_sim {
 public:
  unsigned short __x;
  __half_sim(unsigned short x):__x(x){}
  __half_sim():__x(0){}
  void operator=(const __half_sim &rhs){__x = rhs.__x;}
};

struct memory_block{
  int start;
  int end;
  memory_block(int s, int e):start(s), end(e){}
};

extern __half_sim __float2half_sim(const float &a);

extern float __half2float_sim(const __half_sim &a);

extern float operator*(const __half_sim &lh, const __half_sim &rh);

extern void print_matrix_half__(const __half_sim* x, int h, int w, int c);


class GPU {
 public:
  GPU();
  void SIM_LDG_INSTR();
  void SIM_STG_INSTR();
  void SIM_HMMA_INSTR_STEP0();
  void SIM_HMMA_INSTR_STEP1();
  void SIM_HMMA_INSTR_STEP2();
  void SIM_HMMA_INSTR_STEP3();
  void SIM_S2R_INSTR();
  void SIM_IMAD_INSTR();
  void SIM_LOP3_INSTR(unsigned Rd, unsigned Ra, unsigned Sb, unsigned Sc,
                      unsigned imm);
  void SIM_SHF_INSTR();
  void SIM_CS2R_INSTR();
  void SIM_LEA_INSTR(bool HI, bool X, unsigned Rd, unsigned Ra, unsigned Sb,
                     unsigned Sc, unsigned imm, unsigned Pd0 = 7, unsigned Ps0 = 7);
  void SIM_EXIT_INSTR();
  unsigned *memory_;
  int memory_size;
  int memory_top;

 private:
  // unsigned warpNum_;
  unsigned *regfile_;
  bool *pregfile_;
};

extern void simMalloc(void **ptr, size_t size, GPU &volta);

extern void simMemcpy(void *dst, void *src, size_t count,
                      enum cudaMemcpyKind_sim kind, GPU &volta);

extern void wmma_kernel(__half_sim *a, __half_sim *b, float *c, float *d, dim3_sim &gridDim,
                        dim3_sim &blockDim, GPU &volta);

extern float* gemm_sim(half *a_d, half *b_d, int M, int N, int K);
#endif
