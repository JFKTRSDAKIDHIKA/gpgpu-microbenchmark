// xbar_bw.cu -- Xbar bisection bandwidth across the two L2 partitions.
// SM groups come from each SM's home partition (measured with atomics).
// Address groups come from the recovered hash: partition = b12^b13^b15^b21^b22.
//
// mode 0 All-Local  : each SM group reads its own partition
// mode 1 Uni-Cross  : only the "home 1" group reads partition 0
// mode 2 Bi-Cross   : each group reads the other partition
// mode 3 Asymmetric : both groups read partition 0

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

static const int home_table[78] = {1,1,1,1,1,1,0,0,1,1,0,0,0,0,1,1,1,1,1,1,0,0,1,1,0,0,0,0,1,1,1,1,1,1,0,0,1,1,0,0,0,0,1,1,1,1,1,1,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,1,1,0,0,0,0};
static const int NBLK = 78;
static const int THREADS = 512;
static const int ITERS = 200;
static const int NLINES = 16384;          // 2 MiB per partition
static const int PAGES = 32;              // 64 MiB buffer

__device__ __forceinline__ float4 ldcg4(const float4* p) {
  float4 v;
  asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
  return v;
}

__global__ void bw(const float4* const* p0, const float4* const* p1, int mode,
                   const int* home, float* sink) {
  unsigned int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  const int h = home[smid];

  const float4* const* arr = nullptr;
  if (mode == 0) arr = (h == 0) ? p0 : p1;            // all-local
  else if (mode == 1) arr = (h == 1) ? p0 : nullptr;  // uni-cross (home1 -> P0)
  else if (mode == 2) arr = (h == 0) ? p1 : p0;       // bi-cross
  else arr = p0;                                      // asymmetric

  if (!arr) return;

  float acc = 0.f;
  for (int it = 0; it < ITERS; ++it) {
    for (int i = threadIdx.x; i < NLINES; i += blockDim.x) {
      const float4* line = arr[i];
      #pragma unroll
      for (int k = 0; k < 8; ++k) {
        float4 v = ldcg4(line + k);
        acc += v.x + v.y + v.z + v.w;
      }
    }
  }
  if (acc == -12345.0f) *sink = acc;
}

int main(int argc, char** argv) {
  const int mode = (argc > 1) ? atoi(argv[1]) : 0;
  int mhz = 0; CK(cudaDeviceGetAttribute(&mhz, cudaDevAttrClockRate, 0));
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));

  size_t span = (size_t)PAGES << 20;
  void* raw = nullptr; CK(cudaMalloc(&raw, span + (2 << 20)));
  uintptr_t base = ((uintptr_t)raw + (2 << 20) - 1) & ~((uintptr_t)(2 << 20) - 1);

  // collect lines of each partition by the recovered hash
  std::vector<float4*> h0, h1;
  for (int line = 0; line < (int)(span / 128) && ((int)h0.size() < NLINES || (int)h1.size() < NLINES); ++line) {
    uint64_t a = (uint64_t)base + (uint64_t)line * 128;
    int pr = (int)(((a >> 12) & 1) ^ ((a >> 13) & 1) ^ ((a >> 15) & 1) ^ ((a >> 21) & 1) ^ ((a >> 22) & 1));
    if (pr == 0 && (int)h0.size() < NLINES) h0.push_back((float4*)a);
    if (pr == 1 && (int)h1.size() < NLINES) h1.push_back((float4*)a);
  }
  if ((int)h0.size() < NLINES || (int)h1.size() < NLINES) {
    printf("not enough lines found (%zu / %zu)\n", h0.size(), h1.size()); return 1;
  }
  CK(cudaMemset((void*)base, 0x3c, span));

  float4 *d0 = nullptr, *d1 = nullptr;
  CK(cudaMalloc(&d0, NLINES * 8)); CK(cudaMalloc(&d1, NLINES * 8));
  std::vector<float4*> a0(h0.begin(), h0.begin() + NLINES), a1(h1.begin(), h1.begin() + NLINES);
  CK(cudaMemcpy(d0, a0.data(), NLINES * 8, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d1, a1.data(), NLINES * 8, cudaMemcpyHostToDevice));

  int* dhome = nullptr; CK(cudaMalloc(&dhome, 128 * 4));
  CK(cudaMemcpy(dhome, home_table, 78 * 4, cudaMemcpyHostToDevice));
  float* sink = nullptr; CK(cudaMalloc(&sink, 4));

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  bw<<<NBLK, THREADS>>>((const float4* const*)d0, (const float4* const*)d1, mode, dhome, sink);
  CK(cudaDeviceSynchronize());   // warm
  CK(cudaEventRecord(e0));
  bw<<<NBLK, THREADS>>>((const float4* const*)d0, (const float4* const*)d1, mode, dhome, sink);
  CK(cudaGetLastError()); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
  float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));

  int active = 0;
  for (int i = 0; i < 78; ++i) {
    if (mode == 0) active++;
    else if (mode == 1) active += (home_table[i] == 1);
    else if (mode == 2) active++;
    else active++;
  }
  double bytes = (double)active * NLINES * 128.0 * ITERS;
  printf("mode %d: active blocks %d, %.2f ms, %.1f GB/s aggregate (%.1f GB/s per SM)\n",
         mode, active, ms, bytes / (ms * 1e6), bytes / (ms * 1e6) / active);
  CK(cudaFree(raw)); CK(cudaFree(d0)); CK(cudaFree(d1)); CK(cudaFree(dhome)); CK(cudaFree(sink));
  return 0;
}
