// multism3.cu -- true same-address dependent-load chain from all 78 SMs.
// Each line's first int is 0x5a5a5a00 (low 7 bits zero), so the next address
// base + (val & 0x7F) is always the SAME line, but the address genuinely depends
// on the loaded value -> the 64-load chain is serialized by hardware.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

static const int NLINES = 512;
static const int REPS   = 64;
static const int NBLK   = 78;

__device__ __forceinline__ int load_cg(const int* a) {
  int v; asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(a) : "memory"); return v;
}

__global__ void probe(const char* buf, double* out_mean, int* smids) {
  if (threadIdx.x != 0) return;
  unsigned int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  const int b = blockIdx.x;
  smids[b] = (int)smid;

  int start = (b * 7) % NLINES;
  for (int a = 0; a < NLINES; ++a) {
    int idx = (start + a) % NLINES;
    const char* base = buf + (size_t)idx * 128;
    const int* p = (const int*)base;
    // pointer-chase the SAME line: 0x5a5a5a00 & 0x7f == 0
    int v = *(volatile const int*)p;
    unsigned long long t0 = clock64();
    for (int r = 0; r < REPS; ++r) {
      v = *(volatile const int*)p;
      p = (const int*)(base + (v & 0x7F));
    }
    unsigned long long t1 = clock64();
    out_mean[(size_t)b * NLINES + idx] = (double)(t1 - t0) / REPS + (v & 0);
  }
}

int main() {
  int mhz = 0; CK(cudaDeviceGetAttribute(&mhz, cudaDevAttrClockRate, 0));
  double ghz = mhz / 1e6;
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("device=%s sm=%d L2=%d MiB clock=%.3f GHz lines=%d reps=%d blocks=%d\n",
         p.name, p.multiProcessorCount, p.l2CacheSize >> 20, ghz, NLINES, REPS, NBLK);

  size_t bytes = (size_t)NLINES * 128;
  std::vector<int> h((size_t)NLINES * 32, 0x11);
  for (int i = 0; i < NLINES; ++i) h[(size_t)i * 32] = 0x5a5a5a00;   // low 7 bits = 0
  char* buf = nullptr; CK(cudaMalloc(&buf, bytes));
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

  FILE* f = fopen("multism3.csv", "w");
  for (int b = 0; b < NBLK; ++b)
    for (int a = 0; a < NLINES; ++a)
      fprintf(f, "%d,%d,%.3f\n", sid[b], a, m[(size_t)b * NLINES + a]);
  fclose(f);

  double s = 0;
  for (auto v : m) s += v;
  printf("cells=%d mean=%.1f cyc (%.1f ns)\n", (int)m.size(), s / m.size(), s / m.size() / ghz);
  printf("wrote multism3.csv\n");
  CK(cudaFree(buf)); CK(cudaFree(om)); CK(cudaFree(smids));
  return 0;
}
