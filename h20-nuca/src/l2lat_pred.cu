// l2lat_pred.cu -- same chunked structure as the original probe, but the clock read
// is predicated on the loaded value so ptxas cannot hoist it.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

static const int LINE_BYTES  = 128;
static const int TOTAL_LINES = 60 * 1024 * 1024 / LINE_BYTES;   // 491520
static const int CHUNK_LINES = 8192;                            // 1 MiB

__device__ __forceinline__ int load_l2_bypass_l1(const int* addr) {
  int val;
  asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(val) : "l"(addr));
  return val;
}

__device__ __forceinline__ void store_streaming(unsigned long long* addr, unsigned long long val) {
  asm volatile("st.global.cs.u64 [%0], %1;" :: "l"(addr), "l"(val));
}

__device__ __forceinline__ unsigned long long measure_one_load_o3(const int* ptr) {
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
      : "=l"(t0), "=r"(val), "=l"(t1)
      : "l"(ptr)
      : "memory");
  return t1 - t0;
}

__global__ void probe_pred(int* buf, unsigned long long* lat, int do_warmup) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int sink = 0;

  for (int base = 0; base < TOTAL_LINES; base += CHUNK_LINES) {
    int count = (TOTAL_LINES - base < CHUNK_LINES) ? (TOTAL_LINES - base) : CHUNK_LINES;

    if (do_warmup)
      for (int i = 0; i < count; ++i) sink += load_l2_bypass_l1(&buf[(base + i) * 32]);
    asm volatile("" :: "r"(sink));

    for (int i = count - 1; i >= 0; --i) {
      int idx = base + i;
      store_streaming(&lat[idx], measure_one_load_o3(&buf[idx * 32]));
    }
  }
  if (sink == 0x12345678) buf[0] = sink;
}

static void stats(const char* tag, std::vector<unsigned long long>& v, double ghz) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  auto pct = [&](double p) { return v[(size_t)(p * (n - 1))]; };
  double mean = 0; for (auto x : v) mean += (double)x; mean /= n;
  printf("%s n=%zu  min=%llu  p50=%llu  p90=%llu  p99=%llu  max=%llu  mean=%.1f cyc\n",
         tag, n, v.front(), pct(0.5), pct(0.9), pct(0.99), v.back(), mean);
  printf("        ns: p50=%.0f p90=%.0f p99=%.0f  (clock=%.2f GHz)\n",
         pct(0.5) / ghz, pct(0.9) / ghz, pct(0.99) / ghz, ghz);
  int h[48] = {0};
  for (auto x : v) { int b = (int)(x / 32); if (b < 48) h[b]++; }
  for (int b = 0; b < 48; ++b)
    if (h[b]) printf("        %4d-%4d cyc : %8d  %5.1f%%\n", b * 32, b * 32 + 31, h[b], 100.0 * h[b] / n);
}

int main() {
  int mhz = 0; CK(cudaDeviceGetAttribute(&mhz, cudaDevAttrClockRate, 0));
  double ghz = mhz / 1e6;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("device=%s sm=%d L2=%d MiB clock=%.3f GHz lines=%d\n",
         p.name, p.multiProcessorCount, p.l2CacheSize >> 20, ghz, TOTAL_LINES);

  size_t bytes = (size_t)TOTAL_LINES * LINE_BYTES;
  int* buf = nullptr; CK(cudaMalloc(&buf, bytes));
  CK(cudaMemset(buf, 0x5a, bytes));          // value != -123456789, so p stays true
  unsigned long long* lat = nullptr; CK(cudaMalloc(&lat, (size_t)TOTAL_LINES * 8));

  std::vector<unsigned long long> v(TOTAL_LINES);
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  float ms = 0;

  CK(cudaEventRecord(e0));
  probe_pred<<<1, 1>>>(buf, lat, 1);
  CK(cudaGetLastError()); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
  CK(cudaEventElapsedTime(&ms, e0, e1));
  CK(cudaMemcpy(v.data(), lat, (size_t)TOTAL_LINES * 8, cudaMemcpyDeviceToHost));
  printf("\n[warm] kernel %.1f ms\n", ms);
  stats("[warm]", v, ghz);

  {   // flush L2 with 240 MB, then measure without warming
    size_t fb = 240ull << 20; char* f = nullptr;
    CK(cudaMalloc(&f, fb)); CK(cudaMemset(f, 0x11, fb)); CK(cudaDeviceSynchronize()); CK(cudaFree(f));
  }
  CK(cudaEventRecord(e0));
  probe_pred<<<1, 1>>>(buf, lat, 0);
  CK(cudaGetLastError()); CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
  CK(cudaEventElapsedTime(&ms, e0, e1));
  CK(cudaMemcpy(v.data(), lat, (size_t)TOTAL_LINES * 8, cudaMemcpyDeviceToHost));
  printf("\n[cold] kernel %.1f ms\n", ms);
  stats("[cold]", v, ghz);

  CK(cudaFree(buf)); CK(cudaFree(lat));
  return 0;
}
