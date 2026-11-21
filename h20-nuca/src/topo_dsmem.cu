// topo_dsmem.cu -- 8x8 DSMEM latency matrix inside one GPC.
// Same-TPC pairs are closest, same-CPC next, cross-CPC furthest.
// One cluster of 8 blocks (guaranteed same GPC), remote shared-memory atomics.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

static const int C = 8;
static const int SMEM = 120 * 1024;   // occupancy 1 -> one block per SM
static const int REPS = 64;

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}

__device__ __forceinline__ int atom_remote_add(uint32_t addr) {
  int old;
  asm volatile("atom.shared::cluster.add.u32 %0, [%1], 1;" : "=r"(old) : "r"(addr) : "memory");
  return old;
}

__global__ void probe(int* out, int* smids) {
  extern __shared__ __align__(16) int smem[];
  __shared__ int cnt;

  unsigned int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  unsigned int rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));

  if (threadIdx.x == 0) {
    cnt = 0;
    smids[rank] = (int)smid;
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive.aligned;\n\tbarrier.cluster.wait.aligned;" ::: "memory");

  if (threadIdx.x != 0) return;
  uint32_t base = sa(&cnt);
  out[rank * 9 + 8] = (int)smid;

  for (int t = 0; t < C; ++t) {
    if (t == (int)rank) { out[rank * 9 + t] = -1; continue; }
    uint32_t ra = mapa(base, (uint32_t)t);
    int acc = atom_remote_add(ra);           // warm
    unsigned long long t0 = clock64();
    for (int r = 0; r < REPS; ++r) acc += atom_remote_add(ra);
    unsigned long long t1 = clock64();
    double v = (double)(t1 - t0) / REPS;
    if (acc == 0x7fffffff) v = -1;
    out[rank * 9 + t] = (int)lrint(v);
  }
}

int main() {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("device=%s sm=%d\n", p.name, p.multiProcessorCount);

  int* out = nullptr; int* smids = nullptr;
  CK(cudaMalloc(&out, C * 9 * 4)); CK(cudaMalloc(&smids, C * 4));
  CK(cudaFuncSetAttribute((const void*)probe, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(C); cfg.blockDim = dim3(64); cfg.dynamicSmemBytes = SMEM;
  cudaLaunchAttribute a[1];
  a[0].id = cudaLaunchAttributeClusterDimension;
  a[0].val.clusterDim.x = C; a[0].val.clusterDim.y = 1; a[0].val.clusterDim.z = 1;
  cfg.attrs = a; cfg.numAttrs = 1;
  CK(cudaLaunchKernelEx(&cfg, probe, out, smids));
  CK(cudaDeviceSynchronize());

  int m[C * 9];
  CK(cudaMemcpy(m, out, C * 9 * 4, cudaMemcpyDeviceToHost));
  int s[C];
  CK(cudaMemcpy(s, smids, C * 4, cudaMemcpyDeviceToHost));

  printf("\nSM ids in this cluster:"); for (int i = 0; i < C; ++i) printf(" %d", s[i]);
  printf("\n\nDSMEM remote-atomic latency matrix (cyc):\n     ");
  for (int j = 0; j < C; ++j) printf("  r%d ", j);
  printf("\n");
  for (int i = 0; i < C; ++i) {
    printf("r%d  ", i);
    for (int j = 0; j < C; ++j) printf("%5d ", m[i * 9 + j]);
    printf("\n");
  }

  // rank of each SM
  printf("\nrank -> smid:");
  for (int i = 0; i < C; ++i) printf("  r%d=%d", i, s[i]);
  printf("\n");
  CK(cudaFree(out)); CK(cudaFree(smids));
  return 0;
}
