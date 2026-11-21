// sm_port.cu -- single-SM read/write/duplex port width, L2-resident, L1 bypassed.
// 64 KB window = 512 lines of the local partition (hash b12^b13^b15 == 0).

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

static const int LINES = 512;          // 64 KiB
static const int ITERS = 2000;
static const int UNROLL = 8;

__device__ __forceinline__ float4 ldcg4(const float4* p) {
  float4 v;
  asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void stcg4(float4* p, float4 v) {
  asm volatile("st.global.cg.v4.f32 [%0], {%1,%2,%3,%4};"
               :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}

__global__ void port(const float4* wr, float4* ww, int mode, unsigned long long* out) {
  if (blockIdx.x != 0) return;
  const int tid = threadIdx.x;
  float4 acc = make_float4(0, 0, 0, 0);
  float4 val = make_float4(1, 2, 3, 4);

  // touch everything once
  for (int i = tid; i < LINES * 8; i += blockDim.x) acc.x += ldcg4(wr + i).x;

  unsigned long long t0 = clock64();
  for (int it = 0; it < ITERS; ++it) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int i = (tid + (it * UNROLL + u) * blockDim.x) & (LINES * 8 - 1);
      if (mode == 0) {
        float4 v = ldcg4(wr + i); acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
      } else if (mode == 1) {
        stcg4(ww + i, val);
      } else {
        float4 v = ldcg4(wr + i); acc.x += v.x;
        stcg4(ww + i, val);
      }
    }
  }
  unsigned long long t1 = clock64();

  if (threadIdx.x == 0) out[0] = t1 - t0;
  if (acc.x == -12345.f) ((float4*)wr)[0] = acc;
}

int main(int argc, char** argv) {
  const int mode = (argc > 1) ? atoi(argv[1]) : 0;
  int mhz = 0; CK(cudaDeviceGetAttribute(&mhz, cudaDevAttrClockRate, 0));
  double ghz = mhz / 1e6;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));

  size_t page = 2ull << 20;
  void* raw = nullptr; CK(cudaMalloc(&raw, page * 2));
  uintptr_t base = ((uintptr_t)raw + page - 1) & ~((uintptr_t)page - 1);
  CK(cudaMemset((void*)base, 0x3c, page));

  // pick 512 lines of the local partition
  std::vector<float4*> pick;
  for (int line = 0; line < (int)(page / 128) && (int)pick.size() < LINES; ++line) {
    uint64_t a = (uint64_t)base + line * 128;
    int pr = (int)(((a >> 12) & 1) ^ ((a >> 13) & 1) ^ ((a >> 15) & 1));
    if (pr == 0) pick.push_back((float4*)a);
  }
  if ((int)pick.size() < LINES) { printf("only %zu lines\n", pick.size()); return 1; }

  // build contiguous 64 KiB windows (read + write) so the kernel indexes linearly
  float4* d_r = nullptr; float4* d_w = nullptr;
  CK(cudaMalloc(&d_r, LINES * 128)); CK(cudaMalloc(&d_w, LINES * 128));
  std::vector<char> tmp(LINES * 128, 0x3c);
  CK(cudaMemcpy(d_r, tmp.data(), LINES * 128, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_w, tmp.data(), LINES * 128, cudaMemcpyHostToDevice));

  unsigned long long* out = nullptr; CK(cudaMalloc(&out, 8));
  port<<<1, 512>>>(d_r, d_w, mode, out);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

  unsigned long long cyc = 0;
  port<<<1, 512>>>(d_r, d_w, mode, out);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(&cyc, out, 8, cudaMemcpyDeviceToHost));

  double ops = (double)ITERS * UNROLL * 512;
  double bytes = ops * 16.0 * ((mode == 2) ? 2.0 : 1.0);
  const char* nm = (mode == 0) ? "read " : (mode == 1) ? "write" : "1:1  ";
  printf("mode %s  cycles=%llu  bytes=%.1f MB  %.2f B/cycle  (%.1f GB/s)  clock=%.2f GHz\n",
         nm, cyc, bytes / 1048576.0, bytes / (double)cyc,
         bytes / (double)cyc * ghz, ghz);

  CK(cudaFree(raw)); CK(cudaFree(d_r)); CK(cudaFree(d_w)); CK(cudaFree(out));
  return 0;
}
