// multism4.cu -- 78 x 1024 latency matrix in a 2 MiB-aligned region.
// Each cell: 64 dependent same-line loads (address = base + (val & 0x7F)),
// value 0x5a5a5a00 keeps the address fixed while creating a real dependency.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

static const int NLINES = 1024;       // 128 KiB of lines, inside a 2 MiB aligned window
static const int REPS   = 64;
static const int NBLK   = 78;

__global__ void probe(const char* buf, double* out_mean, int* smids) {
  if (threadIdx.x != 0) return;
  unsigned int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  const int b = blockIdx.x;
  smids[b] = (int)smid;

  int start = (b * 7) % NLINES;               // stagger start, same for all lines per block
  for (int a = 0; a < NLINES; ++a) {
    int idx = (start + a) % NLINES;
    const char* base = buf + (size_t)idx * 128;
    const int* p = (const int*)base;
    int v = *(volatile const int*)p;
    unsigned long long t0 = clock64();
    for (int r = 0; r < REPS; ++r) {
      v = *(volatile const int*)p;
      p = (const int*)(base + (v & 0x7F));     // stays on the same line
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

  // 2 MiB aligned window: allocate 4 MiB, round the pointer up
  void* raw = nullptr; CK(cudaMalloc(&raw, 4ull << 20));
  uintptr_t addr = ((uintptr_t)raw + (2ull << 20) - 1) & ~((uintptr_t)(2ull << 20) - 1);
  char* buf = (char*)addr;
  printf("raw=%p aligned=%p  (2 MiB aligned: %s)\n", raw, buf,
         ((uintptr_t)buf & ((2ull << 20) - 1)) == 0 ? "yes" : "no");

  size_t bytes = (size_t)NLINES * 128;
  std::vector<int> h(bytes / 4, 0x11111111);
  for (int i = 0; i < NLINES; ++i) h[(size_t)i * 32] = 0x5a5a5a00;   // low 7 bits = 0
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

  FILE* f = fopen("multism4.csv", "w");
  for (int b = 0; b < NBLK; ++b)
    for (int a = 0; a < NLINES; ++a)
      fprintf(f, "%d,%d,%.3f\n", sid[b], a, m[(size_t)b * NLINES + a]);
  fclose(f);

  double s = 0; for (auto v : m) s += v;
  printf("cells=%d mean=%.1f cyc (%.1f ns)\n", (int)m.size(), s / m.size(), s / m.size() / ghz);
  printf("wrote multism4.csv\n");
  CK(cudaFree(raw)); CK(cudaFree(om)); CK(cudaFree(smids));
  return 0;
}
