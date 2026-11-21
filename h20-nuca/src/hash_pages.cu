// hash_pages.cu -- same atomic CAS chain, but sampling 4 x 2 MiB pages.
// Purpose: check whether partition = b12^b13^b15 still holds across 2 MiB
// boundaries, or whether a higher physical-address bit flips it.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

static const int PAGES  = 4;
static const int PERPG  = 512;
static const size_t PGSZ = 2ull << 20;
static const int NLINES = PAGES * PERPG;
static const int REPS   = 64;
static const int NBLK   = 78;

__global__ void probe(char* base, double* out, int* smids) {
  if (threadIdx.x != 0) return;
  unsigned int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  const int b = blockIdx.x;
  smids[b] = (int)smid;

  int start = (b * 7) % NLINES;
  for (int a = 0; a < NLINES; ++a) {
    int t = (start + a) % NLINES;
    int page = t / PERPG, i = t % PERPG;
    char* line = base + page * PGSZ + (size_t)i * 128;
    int* p = (int*)line;

    int old = atomicCAS(p, 0, 0);
    p = (int*)(line + (old & 0x7F));
    unsigned long long t0 = clock64();
    for (int r = 0; r < REPS; ++r) {
      old = atomicCAS(p, 0, 0);
      p = (int*)(line + (old & 0x7F));
    }
    unsigned long long t1 = clock64();

    double v = (double)(t1 - t0) / REPS;
    if (old == 0x7fffffff) v = -1;
    out[(size_t)b * NLINES + t] = v;
  }
}

int main() {
  int mhz = 0; CK(cudaDeviceGetAttribute(&mhz, cudaDevAttrClockRate, 0));
  double ghz = mhz / 1e6;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));

  size_t need = PAGES * PGSZ;
  void* raw = nullptr; CK(cudaMalloc(&raw, need + PGSZ));
  uintptr_t addr = ((uintptr_t)raw + PGSZ - 1) & ~((uintptr_t)PGSZ - 1);
  char* base = (char*)addr;
  printf("device=%s sm=%d L2=%d MiB clock=%.3f GHz pages=%d lines/page=%d\n",
         p.name, p.multiProcessorCount, p.l2CacheSize >> 20, ghz, PAGES, PERPG);
  printf("base=%p 2MiB-aligned=%s span=%.1f MiB\n", base,
         ((uintptr_t)base & (PGSZ - 1)) == 0 ? "yes" : "no", need / 1048576.0);

  CK(cudaMemset(base, 0, need));

  double* om = nullptr; int* smids = nullptr;
  CK(cudaMalloc(&om, (size_t)NBLK * NLINES * 8));
  CK(cudaMalloc(&smids, NBLK * 4));
  probe<<<NBLK, 1>>>(base, om, smids);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

  std::vector<double> m((size_t)NBLK * NLINES);
  std::vector<int> sid(NBLK);
  CK(cudaMemcpy(m.data(), om, m.size() * 8, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(sid.data(), smids, NBLK * 4, cudaMemcpyDeviceToHost));

  FILE* f = fopen("hash_pages.csv", "w");
  for (int b = 0; b < NBLK; ++b)
    for (int t = 0; t < NLINES; ++t)
      fprintf(f, "%d,%d,%.3f\n", sid[b], t, m[(size_t)b * NLINES + t]);
  fclose(f);

  double s = 0; for (auto v : m) s += v;
  printf("cells=%d mean=%.1f cyc (%.1f ns)\n", (int)m.size(), s / m.size(), s / m.size() / ghz);
  printf("wrote hash_pages.csv\n");
  CK(cudaFree(raw)); CK(cudaFree(om)); CK(cudaFree(smids));
  return 0;
}
