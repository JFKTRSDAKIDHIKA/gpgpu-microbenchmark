// l2lat.cu -- per-line load latency probe, chunked over 60 MB, .cg load + .cs store.
// Reports distribution (min/p50/p90/p99/max) and a histogram, plus a cold-data
// control run and a clock64() overhead calibration.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

static const int LINE_BYTES  = 128;
static const int TOTAL_LINES = 60 * 1024 * 1024 / LINE_BYTES;   // 491520
static const int CHUNK_LINES = 8192;                            // 1 MiB
static const int ITERS_CAL   = 200000;

__device__ __forceinline__ int load_l2_bypass_l1(const int* addr) {
  int val;
  asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(val) : "l"(addr));
  return val;
}

__device__ __forceinline__ void store_streaming(unsigned long long* addr, unsigned long long val) {
  asm volatile("st.global.cs.u64 [%0], %1;" :: "l"(addr), "l"(val));
}

__global__ void clock_overhead(unsigned long long* out) {
  if (threadIdx.x != 0) return;
  unsigned long long acc = 0;
  for (int i = 0; i < ITERS_CAL; ++i) {
    unsigned long long t0 = clock64();
    unsigned long long t1 = clock64();
    acc += t1 - t0;
  }
  out[0] = acc;
}

__global__ void probe(int* buf, unsigned long long* lat, int do_warmup) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;   // single thread
  int sink = 0;

  for (int base = 0; base < TOTAL_LINES; base += CHUNK_LINES) {
    int count = (TOTAL_LINES - base < CHUNK_LINES) ? (TOTAL_LINES - base) : CHUNK_LINES;

    if (do_warmup) {
      for (int i = 0; i < count; ++i) sink += load_l2_bypass_l1(&buf[(base + i) * 32]);
    }
    asm volatile("" :: "r"(sink));

    for (int i = count - 1; i >= 0; --i) {
      int idx = base + i;
      const int* ptr = &buf[idx * 32];
      unsigned long long t0 = clock64();
      unsigned long long t1;
      int val;
      // load + dependent add + clock in ONE asm block: the compiler cannot reorder
      // them, and the add stalls until the load returns (in-order issue).
      asm volatile("ld.global.cg.s32 %0, [%2];\n\t"
                   "add.s32 %0, %0, 1;\n\t"
                   "mov.u64 %1, %%clock64;\n\t"
                   : "=&r"(val), "=l"(t1) : "l"(ptr) : "memory");
      store_streaming(&lat[idx], t1 - t0);
    }
  }
  if (sink == 0x12345678) buf[0] = sink;
}

static void stats(const char* tag, std::vector<unsigned long long>& v, double ghz, unsigned long long cal) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  auto pct = [&](double p) { return v[(size_t)(p * (n - 1))]; };
  double mean = 0;
  for (auto x : v) mean += (double)x;
  mean /= n;
  unsigned long long mn = v.front(), mx = v.back();
  printf("%s  n=%zu  min=%llu  p50=%llu  p90=%llu  p99=%llu  max=%llu  mean=%.1f  (cycles, clock64 overhead=%llu)\n",
         tag, n, mn, pct(0.50), pct(0.90), pct(0.99), mx, mean, cal);
  printf("        -> ns:  p50=%.0f  p90=%.0f  p99=%.0f  max=%.0f   (after subtracting cal)\n",
         (pct(0.50) - (double)cal) / ghz, (pct(0.90) - (double)cal) / ghz,
         (pct(0.99) - (double)cal) / ghz, (mx - (double)cal) / ghz);

  // histogram: 16-cycle buckets up to 1024
  int h[64] = {0};
  for (auto x : v) { int b = (int)(x / 16); if (b < 64) h[b]++; }
  printf("        histogram (16-cycle buckets):\n");
  for (int b = 0; b < 64; ++b) {
    if (!h[b]) continue;
    double lo = b * 16, hi = lo + 15;
    printf("        %5.0f-%5.0f : %8d  %5.1f%%\n", lo, hi, h[b], 100.0 * h[b] / n);
  }
}

int main() {
  int ghz_mhz = 0;
  CK(cudaDeviceGetAttribute(&ghz_mhz, cudaDevAttrClockRate, 0));
  double ghz = ghz_mhz / 1e6;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("device=%s sm=%d L2=%d MiB clock=%.3f GHz  lines=%d (%.0f MiB)  chunk=%d lines (1 MiB)\n",
         p.name, p.multiProcessorCount, p.l2CacheSize >> 20, ghz, TOTAL_LINES,
         TOTAL_LINES * (double)LINE_BYTES / 1048576.0, CHUNK_LINES);

  size_t buf_bytes = (size_t)TOTAL_LINES * LINE_BYTES;
  int* buf = nullptr; CK(cudaMalloc(&buf, buf_bytes));
  CK(cudaMemset(buf, 0x5a, buf_bytes));
  unsigned long long* lat = nullptr; CK(cudaMalloc(&lat, (size_t)TOTAL_LINES * 8));
  unsigned long long* cal = nullptr; CK(cudaMalloc(&cal, 8));

  CK(cudaMemset(cal, 0, 8));
  clock_overhead<<<1, 1>>>(cal);
  CK(cudaDeviceSynchronize());
  unsigned long long h_cal = 0; CK(cudaMemcpy(&h_cal, cal, 8, cudaMemcpyDeviceToHost));
  unsigned long long cal_per = h_cal / ITERS_CAL;
  printf("clock64 pair overhead = %llu cycles (%.1f ns)\n", cal_per, cal_per / ghz);

  std::vector<unsigned long long> v(TOTAL_LINES);

  // ---- warm (chunk preloaded into L2) ----
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  probe<<<1, 1>>>(buf, lat, 1);
  CK(cudaGetLastError()); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
  float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
  CK(cudaMemcpy(v.data(), lat, (size_t)TOTAL_LINES * 8, cudaMemcpyDeviceToHost));
  printf("\n[warm probe] kernel %.1f ms, %.1f lines/us\n", ms, TOTAL_LINES / (ms * 1000.0));
  stats("[warm]", v, ghz, cal_per);

  // ---- flush L2 with a 240 MB sweep, then cold probe (no warmup) ----
  {
    size_t fb = 240ull << 20;
    char* f = nullptr; CK(cudaMalloc(&f, fb));
    CK(cudaMemset(f, 0x11, fb));
    CK(cudaDeviceSynchronize());
    CK(cudaFree(f));
  }
  CK(cudaEventRecord(e0));
  probe<<<1, 1>>>(buf, lat, 0);
  CK(cudaGetLastError()); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
  CK(cudaEventElapsedTime(&ms, e0, e1));
  CK(cudaMemcpy(v.data(), lat, (size_t)TOTAL_LINES * 8, cudaMemcpyDeviceToHost));
  printf("\n[cold probe] kernel %.1f ms\n", ms);
  stats("[cold]", v, ghz, cal_per);

  CK(cudaFree(buf)); CK(cudaFree(lat)); CK(cudaFree(cal));
  return 0;
}
