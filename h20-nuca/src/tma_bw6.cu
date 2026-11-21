// tma_bw.cu -- per-SM bandwidth using cp.async.bulk (1D TMA), no LSU issue slots.
// 4-stage pipeline of 32 KB copies into shared memory; one thread per SM issues them.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

static const int NBLK    = 78;
static const int STAGES  = 6;
static const int BYTES   = 32768;          // 32 KiB per copy
static const int ITERS   = 2000;

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void mbar_init(uint64_t* b, uint32_t c) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(sa(b)), "r"(c));
}
__device__ __forceinline__ void mbar_expect(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sa(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t phase) {
  uint32_t done = 0;
  while (!done) {
    asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.b32 %0, 1, 0, p;\n\t}"
                 : "=r"(done) : "r"(sa(b)), "r"(phase) : "memory");
  }
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(sa(dst)), "l"(src), "r"(bytes), "r"(sa(bar)) : "memory");
}

__global__ void tma_bw(const char* buf, size_t stride, int iters, size_t region, float* sink) {
  extern __shared__ __align__(1024) char smem[];
  __shared__ __align__(8) uint64_t bar[STAGES];

  unsigned int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));

  if (threadIdx.x == 0) {
    for (int s = 0; s < STAGES; ++s) mbar_init(&bar[s], 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (threadIdx.x != 0) return;

  const char* base = buf + (size_t)blockIdx.x * stride;
  uint32_t phase[STAGES];
  for (int s = 0; s < STAGES; ++s) phase[s] = 0;

  // prefill
  for (int i = 0; i < STAGES && i < iters; ++i) {
    mbar_expect(&bar[i], BYTES);
    bulk_g2s(smem + (size_t)i * BYTES, base + ((size_t)i * BYTES) % region, BYTES, &bar[i]);
  }
  for (int i = STAGES; i < iters; ++i) {
    int s = i % STAGES;
    mbar_wait(&bar[s], phase[s]); phase[s] ^= 1;
    mbar_expect(&bar[s], BYTES);
    bulk_g2s(smem + (size_t)s * BYTES, base + ((size_t)i * BYTES) % region, BYTES, &bar[s]);
  }
  for (int s = 0; s < STAGES; ++s) {
    int i = iters - STAGES + s;
    if (i >= 0) { mbar_wait(&bar[s], phase[s]); phase[s] ^= 1; }
  }
  if (smid == 0xdeadbeef) *sink = 1.f;
}

int main(int argc, char** argv) {
  const int iters = (argc > 1) ? atoi(argv[1]) : ITERS;
  const size_t region = (argc > 2) ? (size_t)atoi(argv[2]) * 1024 : (size_t)iters * BYTES;
  int mhz = 0; CK(cudaDeviceGetAttribute(&mhz, cudaDevAttrClockRate, 0));
  double ghz = mhz / 1e6;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));

  size_t stride = region + 4096;
  size_t total = stride * NBLK;
  void* raw = nullptr; CK(cudaMalloc(&raw, total));
  CK(cudaMemset(raw, 0x3c, total));
  printf("device=%s sm=%d clock=%.3f GHz  per-SM region %.1f MiB  total %.1f MiB\n",
         p.name, p.multiProcessorCount, ghz, stride / 1048576.0, total / 1048576.0);

  int smem = STAGES * BYTES + 1024;
  CK(cudaFuncSetAttribute((const void*)tma_bw, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  float* sink = nullptr; CK(cudaMalloc(&sink, 4));

  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  tma_bw<<<NBLK, 32, smem>>>((const char*)raw, stride, iters, region, sink);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

  CK(cudaEventRecord(e0));
  tma_bw<<<NBLK, 32, smem>>>((const char*)raw, stride, iters, region, sink);
  CK(cudaGetLastError()); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
  float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));

  double bytes = (double)NBLK * iters * BYTES;
  printf("iters=%d region=%zu KiB  footprint=%zu MiB  time=%.3f ms  aggregate %.1f GB/s  per SM %.2f GB/s\n",
         iters, region / 1024, (size_t)NBLK * region / 1048576, ms,
         bytes / (ms * 1e6), bytes / (ms * 1e6) / NBLK);
  CK(cudaFree(raw)); CK(cudaFree(sink));
  return 0;
}
