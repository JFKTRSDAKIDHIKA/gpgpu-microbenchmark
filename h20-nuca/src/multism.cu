// multism.cu -- same 512 cache lines, measured from all 78 SMs.
// Each block (1 thread) is expected to land on a distinct SM; we read %smid to know.
// For every line we take the min of REPS timed loads (min filters transient noise).
// Output: per-SM mean latency, per-line spread across SMs, CSV dump.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>
#include <map>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

static const int NLINES = 512;      // 512 x 128 B = 64 KiB, fully L2 resident
static const int REPS   = 32;
static const int NBLK   = 78;       // one block per SM

__device__ __forceinline__ unsigned long long timed_load_cg(const int* ptr) {
  unsigned long long t0, t1;
  int val;
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

__global__ void probe(const char* buf, unsigned long long* out, int* smids) {
  if (threadIdx.x != 0) return;
  unsigned int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  const int b = blockIdx.x;
  smids[b] = (int)smid;

  // warm this SM's L1/L2 path and stagger the start point per block
  int start = (b * 7) % NLINES;
  for (int a = 0; a < NLINES; ++a) {
    int idx = (start + a) % NLINES;
    const int* p = (const int*)(buf + (size_t)idx * 128);
    size_t base = ((size_t)b * NLINES + idx) * REPS;
    for (int r = 0; r < REPS; ++r) {
      out[base + r] = timed_load_cg(p);   // store right after each measurement
    }
  }
}

int main() {
  int mhz = 0; CK(cudaDeviceGetAttribute(&mhz, cudaDevAttrClockRate, 0));
  double ghz = mhz / 1e6;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("device=%s sm=%d L2=%d MiB clock=%.3f GHz  lines=%d reps=%d blocks=%d\n",
         p.name, p.multiProcessorCount, p.l2CacheSize >> 20, ghz, NLINES, REPS, NBLK);

  size_t bytes = (size_t)NLINES * 128;
  char* buf = nullptr; CK(cudaMalloc(&buf, bytes));
  CK(cudaMemset(buf, 0x5a, bytes));          // != magic, predicate stays true
  unsigned long long* out = nullptr; CK(cudaMalloc(&out, (size_t)NBLK * NLINES * REPS * 8));
  int* smids = nullptr; CK(cudaMalloc(&smids, NBLK * 4));
  CK(cudaMemset(out, 0, (size_t)NBLK * NLINES * REPS * 8));

  probe<<<NBLK, 1>>>(buf, out, smids);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

  std::vector<unsigned long long> h((size_t)NBLK * NLINES * REPS);
  std::vector<int> sid(NBLK);
  CK(cudaMemcpy(h.data(), out, (size_t)NBLK * NLINES * REPS * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(sid.data(), smids, NBLK * 4, cudaMemcpyDeviceToHost));

  std::map<int, int> uniq;
  for (int i = 0; i < NBLK; ++i) uniq[sid[i]]++;
  printf("distinct SMs hit: %zu (blocks=%d)\n", uniq.size(), NBLK);
  if (uniq.size() != (size_t)NBLK) printf("WARN: some SM got more than one block\n");

  // per-SM mean over lines
  std::vector<std::pair<double, int>> smmean;
  for (int b = 0; b < NBLK; ++b) {
    double s = 0;
    for (int a = 0; a < NLINES; ++a) { unsigned long long m = ~0ull; for (int r = 0; r < REPS; ++r) { size_t o = ((size_t)b * NLINES + a) * REPS + r; if (h[o] < m) m = h[o]; } s += (double)m; }
    smmean.push_back({s / NLINES, sid[b]});
  }
  std::sort(smmean.begin(), smmean.end());
  double gsum = 0;
  for (auto& x : smmean) gsum += x.first;
  double gmean = gsum / NBLK;
  printf("\nper-SM mean latency (sorted):\n  min %.1f (sm %d)  ...  max %.1f (sm %d)  spread %.1f cyc = %.1f ns\n",
         smmean.front().first, smmean.front().second,
         smmean.back().first, smmean.back().second,
         smmean.back().first - smmean.front().first,
         (smmean.back().first - smmean.front().first) / ghz);
  printf("  global mean %.1f cyc = %.1f ns\n", gmean, gmean / ghz);
  printf("  first 8: ");
  for (int i = 0; i < 8; ++i) printf("%.0f ", smmean[i].first);
  printf("\n  last  8: ");
  for (int i = NBLK - 8; i < NBLK; ++i) printf("%.0f ", smmean[i].first);
  printf("\n");

  // per-line spread across SMs
  std::vector<int> spread(NLINES);
  for (int a = 0; a < NLINES; ++a) {
    unsigned long long lo = ~0ull, hi = 0;
    for (int b = 0; b < NBLK; ++b) {
      unsigned long long v = ~0ull; for (int r = 0; r < REPS; ++r) { size_t o = ((size_t)b * NLINES + a) * REPS + r; if (h[o] < v) v = h[o]; }
      lo = std::min(lo, v); hi = std::max(hi, v);
    }
    spread[a] = (int)(hi - lo);
  }
  std::vector<int> ss = spread;
  std::sort(ss.begin(), ss.end());
  auto pct = [&](double q) { return ss[(size_t)(q * (NLINES - 1))]; };
  printf("\nper-line spread across 78 SMs (max-min):  p10=%d  p50=%d  p90=%d  p99=%d  max=%d cyc\n",
         pct(0.10), pct(0.50), pct(0.90), pct(0.99), ss.back());

  // csv: smid,line,cycles
  FILE* f = fopen("multism.csv", "w");
  for (int b = 0; b < NBLK; ++b)
    for (int a = 0; a < NLINES; ++a)
      { unsigned long long m = ~0ull; for (int r = 0; r < REPS; ++r) { size_t o2 = ((size_t)b * NLINES + a) * REPS + r; if (h[o2] < m) m = h[o2]; } fprintf(f, "%d,%d,%llu\n", sid[b], a, m); }
  fclose(f);

  CK(cudaFree(buf)); CK(cudaFree(out)); CK(cudaFree(smids));
  return 0;
}
