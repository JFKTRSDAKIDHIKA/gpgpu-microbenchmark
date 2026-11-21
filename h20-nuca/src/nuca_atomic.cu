// nuca_atomic.cu -- same-address atomic chain (atomicAdd) from all 78 SMs.
// Atomics cannot be served by a read-only local replica: every op must be ordered
// at the address's home L2 slice, so the SM<->partition distance should show up
// much more strongly than with plain loads.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

static const int NLINES = 512;
static const int REPS   = 64;
static const int NBLK   = 78;

__global__ void probe(char* buf, double* out_mean, int* smids) {
  if (threadIdx.x != 0) return;
  unsigned int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  const int b = blockIdx.x;
  smids[b] = (int)smid;

  int start = (b * 7) % NLINES;
  for (int a = 0; a < NLINES; ++a) {
    int idx = (start + a) % NLINES;
    int* p = (int*)(buf + (size_t)idx * 128);

    // CAS(0,0) keeps the word at 0 and returns 0, but the returned value is
    // still a real dependency: the next address is base + (old & 0x7F) == base.
    int old = atomicCAS(p, 0, 0);
    const char* base = buf + (size_t)idx * 128;
    p = (int*)(base + (old & 0x7F));

    unsigned long long t0 = clock64();
    for (int r = 0; r < REPS; ++r) {
      old = atomicCAS(p, 0, 0);
      p = (int*)(base + (old & 0x7F));
    }
    unsigned long long t1 = clock64();

    double v = (double)(t1 - t0) / REPS;
    if (old == 0x7fffffff) v = -1;
    out_mean[(size_t)b * NLINES + idx] = v;
  }
}

int main() {
  int mhz = 0; CK(cudaDeviceGetAttribute(&mhz, cudaDevAttrClockRate, 0));
  double ghz = mhz / 1e6;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("device=%s sm=%d L2=%d MiB clock=%.3f GHz lines=%d reps=%d blocks=%d\n",
         p.name, p.multiProcessorCount, p.l2CacheSize >> 20, ghz, NLINES, REPS, NBLK);

  void* raw = nullptr; CK(cudaMalloc(&raw, 4ull << 20));
  uintptr_t addr = ((uintptr_t)raw + (2ull << 20) - 1) & ~((uintptr_t)(2ull << 20) - 1);
  char* buf = (char*)addr;
  size_t bytes = (size_t)NLINES * 128;
  std::vector<int> h(bytes / 4, 0);
  for (int i = 0; i < NLINES; ++i) h[(size_t)i * 32] = 0;
  CK(cudaMemcpy(buf, h.data(), bytes, cudaMemcpyHostToDevice));

  double* om = nullptr; int* smids = nullptr;
  CK(cudaMalloc(&om, (size_t)NBLK * NLINES * 8));
  CK(cudaMalloc(&smids, NBLK * 4));

  probe<<<NBLK, 1>>>(buf, om, smids);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

  std::vector<double> m((size_t)NBLK * NLINES);
  std::vector<int> sid(NBLK);
  CK(cudaMemcpy(m.data(), om, m.size() * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(sid.data(), smids, NBLK * 4, cudaMemcpyDeviceToHost));

  FILE* f = fopen("nuca_atomic.csv", "w");
  for (int b = 0; b < NBLK; ++b)
    for (int a = 0; a < NLINES; ++a)
      fprintf(f, "%d,%d,%.3f\n", sid[b], a, m[(size_t)b * NLINES + a]);
  fclose(f);

  double s = 0; for (auto v : m) s += v;
  printf("cells=%d mean=%.1f cyc (%.1f ns)\n", (int)m.size(), s / m.size(), s / m.size() / ghz);
  printf("wrote nuca_atomic.csv\n");
  CK(cudaFree(raw)); CK(cudaFree(om)); CK(cudaFree(smids));
  return 0;
}
