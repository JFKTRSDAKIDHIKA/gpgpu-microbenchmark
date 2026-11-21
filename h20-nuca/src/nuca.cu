// nuca.cu -- per-line L2 latency with address recorded, to look for NUCA splits.
// 8 MiB working set (fully L2 resident), random permutation chase, predicated clock.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <random>
#include <cstdint>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

__device__ __forceinline__ int load_cg(const int* a) {
  int v; asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(a)); return v;
}

__device__ __forceinline__ unsigned long long timed_load(const int* ptr, int* out_val) {
  unsigned long long t0, t1; int val;
  asm volatile(
      "mov.u64 %0, %%clock64;\n\t"
      "ld.global.cg.s32 %1, [%3];\n\t"
      "{\n\t"
      "  .reg .pred p;\n\t"
      "  setp.ne.s32 p, %1, -123456789;\n\t"
      "  @p mov.u64 %2, %%clock64;\n\t"
      "}\n\t"
      : "=l"(t0), "=r"(val), "=l"(t1) : "l"(ptr) : "memory");
  *out_val = val;
  return t1 - t0;
}

__global__ void chase(const int* buf, int start, int steps, int lines,
                      unsigned long long* lat, int* idxs) {
  if (blockIdx.x || threadIdx.x) return;
  int idx = start;
  for (int i = 0; i < steps; ++i) {
    int next;
    unsigned long long c = timed_load(&buf[(size_t)idx * 32], &next);
    lat[i] = c;
    idxs[i] = idx;
    idx = (next >= 0 && next < lines) ? next : 0;
  }
}

int main(int argc, char** argv) {
  const int mib = (argc > 1) ? atoi(argv[1]) : 8;
  const int steps = (argc > 2) ? atoi(argv[2]) : 200000;
  const int lines = mib * 1024 * 1024 / 128;

  int mhz = 0; CK(cudaDeviceGetAttribute(&mhz, cudaDevAttrClockRate, 0));
  double ghz = mhz / 1e6;
  printf("lines=%d (%.1f MiB) steps=%d clock=%.3f GHz\n", lines, lines * 128.0 / 1048576.0, steps, ghz);

  std::vector<int> h((size_t)lines * 32, 0);
  std::mt19937 rng(7);
  std::vector<int> perm(lines);
  for (int i = 0; i < lines; ++i) perm[i] = i;
  std::shuffle(perm.begin(), perm.end(), rng);
  for (int i = 0; i < lines; ++i) h[(size_t)i * 32] = perm[i];

  int* d = nullptr; CK(cudaMalloc(&d, (size_t)lines * 128));
  CK(cudaMemcpy(d, h.data(), (size_t)lines * 128, cudaMemcpyHostToDevice));
  unsigned long long* dlat = nullptr; CK(cudaMalloc(&dlat, (size_t)steps * 8));
  int* didx = nullptr; CK(cudaMalloc(&didx, (size_t)steps * 4));

  chase<<<1, 1>>>(d, 0, steps / 10, lines, dlat, didx);   // warm
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  chase<<<1, 1>>>(d, 0, steps, lines, dlat, didx);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

  std::vector<unsigned long long> lat(steps);
  std::vector<int> idx(steps);
  CK(cudaMemcpy(lat.data(), dlat, (size_t)steps * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(idx.data(), didx, (size_t)steps * 4, cudaMemcpyDeviceToHost));

  // dump csv: idx,cycles
  FILE* f = fopen("nuca.csv", "w");
  for (int i = 0; i < steps; ++i) fprintf(f, "%d,%llu\n", idx[i], lat[i]);
  fclose(f);

  std::vector<unsigned long long> s = lat;
  std::sort(s.begin(), s.end());
  auto pct = [&](double p) { return s[(size_t)(p * (steps - 1))]; };
  double mean = 0; for (auto x : s) mean += x; mean /= steps;
  printf("min=%llu p10=%llu p50=%llu p90=%llu p99=%llu max=%llu mean=%.1f\n",
         s.front(), pct(0.10), pct(0.50), pct(0.90), pct(0.99), s.back(), mean);

  // latency vs address bucket (64 buckets across the buffer)
  int NB = 64;
  std::vector<double> sum(NB, 0); std::vector<int> cnt(NB, 0);
  for (int i = 0; i < steps; ++i) {
    int b = (int)((long long)idx[i] * NB / lines);
    sum[b] += lat[i]; cnt[b]++;
  }
  printf("bucket  mean_cyc  n\n");
  for (int b = 0; b < NB; ++b)
    if (cnt[b]) printf("%2d %8.1f %6d\n", b, sum[b] / cnt[b], cnt[b]);

  CK(cudaFree(d)); CK(cudaFree(dlat)); CK(cudaFree(didx));
  return 0;
}
