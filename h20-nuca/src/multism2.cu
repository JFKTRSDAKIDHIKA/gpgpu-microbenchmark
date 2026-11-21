// multism2.cu -- same-address self-chase from all 78 SMs.
// For every (SM, line) cell: 64 consecutive timed loads to that SAME line, record
// mean and stdev. 78 blocks (1 thread each) so each SM measures the same 512 lines.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

static const int NLINES = 512;
static const int REPS   = 64;
static const int NBLK   = 78;

__device__ __forceinline__ unsigned long long timed_load_cg(const int* ptr) {
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
  return t1 - t0;
}

__global__ void probe(const char* buf, double* out_mean, double* out_sd, int* smids) {
  if (threadIdx.x != 0) return;
  unsigned int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  const int b = blockIdx.x;
  smids[b] = (int)smid;

  int start = (b * 7) % NLINES;
  for (int a = 0; a < NLINES; ++a) {
    int idx = (start + a) % NLINES;
    const int* p = (const int*)(buf + (size_t)idx * 128);
    unsigned long long sum = 0, sq = 0;
    for (int r = 0; r < REPS; ++r) {
      unsigned long long c = timed_load_cg(p);
      sum += c; sq += c * c;
    }
    double mean = (double)sum / REPS;
    double var = (double)sq / REPS - mean * mean;
    size_t o = (size_t)b * NLINES + idx;
    out_mean[o] = mean;
    out_sd[o] = sqrt(var > 0 ? var : 0);
  }
}

int main() {
  int mhz = 0; CK(cudaDeviceGetAttribute(&mhz, cudaDevAttrClockRate, 0));
  double ghz = mhz / 1e6;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("device=%s sm=%d L2=%d MiB clock=%.3f GHz lines=%d reps=%d blocks=%d\n",
         p.name, p.multiProcessorCount, p.l2CacheSize >> 20, ghz, NLINES, REPS, NBLK);

  size_t bytes = (size_t)NLINES * 128;
  char* buf = nullptr; CK(cudaMalloc(&buf, bytes));
  CK(cudaMemset(buf, 0x5a, bytes));
  double *om = nullptr, *os = nullptr;
  int* smids = nullptr;
  CK(cudaMalloc(&om, (size_t)NBLK * NLINES * 8));
  CK(cudaMalloc(&os, (size_t)NBLK * NLINES * 8));
  CK(cudaMalloc(&smids, NBLK * 4));

  probe<<<NBLK, 1>>>(buf, om, os, smids);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

  std::vector<double> m((size_t)NBLK * NLINES), s((size_t)NBLK * NLINES);
  std::vector<int> sid(NBLK);
  CK(cudaMemcpy(m.data(), om, m.size() * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(s.data(), os, s.size() * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(sid.data(), smids, NBLK * 4, cudaMemcpyDeviceToHost));

  FILE* f = fopen("multism2.csv", "w");
  for (int b = 0; b < NBLK; ++b)
    for (int a = 0; a < NLINES; ++a)
      fprintf(f, "%d,%d,%.3f,%.3f\n", sid[b], a, m[(size_t)b * NLINES + a], s[(size_t)b * NLINES + a]);
  fclose(f);

  double sum = 0; int n = 0;
  for (auto v : m) { sum += v; n++; }
  printf("cells=%d mean=%.1f cyc\n", n, sum / n);
  printf("wrote multism2.csv\n");
  CK(cudaFree(buf)); CK(cudaFree(om)); CK(cudaFree(os)); CK(cudaFree(smids));
  return 0;
}
