// chase.cu -- pointer-chase latency probe (immune to clock-hoisting).
// Each load's result is the next address, so the chain is serialized by hardware
// data dependency; one clock64() pair brackets the whole loop and we divide.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <random>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

__device__ __forceinline__ int load_cg(const int* a) {
  int v;
  asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(a));
  return v;
}

// one thread chases `steps` links inside a buffer of `lines` 128-byte lines
__global__ void chase(const int* buf, int start, int steps, int lines, unsigned long long* out) {
  if (blockIdx.x || threadIdx.x) return;
  int idx = start;
  unsigned long long t0 = clock64();
  for (int i = 0; i < steps; ++i) {
    idx = load_cg(&buf[(size_t)idx * 32]);     // next index stored at line start
    if (idx >= lines) idx = 0;
  }
  unsigned long long t1 = clock64();
  out[0] = t1 - t0;
  out[1] = (unsigned long long)idx;
}

int main() {
  int mhz = 0; CK(cudaDeviceGetAttribute(&mhz, cudaDevAttrClockRate, 0));
  double ghz = mhz / 1e6;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("device=%s sm=%d L2=%d MiB clock=%.3f GHz\n", p.name, p.multiProcessorCount, p.l2CacheSize >> 20, ghz);

  const int STEPS_SMALL = 200000, STEPS_BIG = 200000;

  for (int variant = 0; variant < 3; ++variant) {
    const int lines = (variant == 0) ? 8192
                    : (variant == 1) ? (60 * 1024 * 1024 / 128)
                                     : (1024 * 1024 * 1024 / 128);          // 1 MiB / 60 MiB / 1 GiB
    const size_t bytes = (size_t)lines * 128;
    std::vector<int> h((size_t)lines * 32, 0);

    std::mt19937 rng(12345);
    std::vector<int> perm(lines);
    for (int i = 0; i < lines; ++i) perm[i] = i;
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int i = 0; i < lines; ++i) h[(size_t)i * 32] = perm[i];

    int* d = nullptr; CK(cudaMalloc(&d, bytes));
    CK(cudaMemcpy(d, h.data(), bytes, cudaMemcpyHostToDevice));
    unsigned long long* out = nullptr; CK(cudaMalloc(&out, 16));

    const int steps = (variant == 0) ? STEPS_SMALL : STEPS_BIG;
    // warm up so the small case is L2-resident
    CK(cudaMemset(out, 0, 16));
    chase<<<1, 1>>>(d, 0, steps / 10, lines, out);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    CK(cudaMemset(out, 0, 16));
    chase<<<1, 1>>>(d, 0, steps, lines, out);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    unsigned long long hh[2]; CK(cudaMemcpy(hh, out, 16, cudaMemcpyDeviceToHost));

    double cyc = (double)hh[0] / steps;
    double ns = cyc / ghz;
    printf("%s  lines=%d (%.1f MiB) steps=%d  total=%llu cyc  -> %.1f cyc/step = %.0f ns  (%.1f MB/s effective)\n",
           (variant == 0) ? "[L2-resident]" : (variant == 1 ? "[60 MiB     ]" : "[1 GiB      ]"), lines, bytes / 1048576.0,
           steps, hh[0], cyc, ns, 128.0 / ns * 1000.0);
    CK(cudaFree(d)); CK(cudaFree(out));
  }
  return 0;
}
