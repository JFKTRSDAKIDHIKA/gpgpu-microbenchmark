// occ_check.cu -- true occupancy for the pack_spread2 launch shape, at 100 KB vs 120 KB,
// and whether 78 blocks really map 1:1 onto 78 SMs.

#include <cstdio>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

__global__ void k(int* out) {
  extern __shared__ char smem[];
  unsigned int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  if (threadIdx.x == 0) out[blockIdx.x] = (int)smid;
}

int main() {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("device=%s  SMs=%d  maxThreads/SM=%d  smem/SM=%zu KB\n",
         p.name, p.multiProcessorCount, p.maxThreadsPerMultiProcessor,
         p.sharedMemPerMultiprocessor / 1024);

  for (int kb : {48, 100, 120}) {
    size_t sm = (size_t)kb * 1024;
    CK(cudaFuncSetAttribute((const void*)k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
    int occ = -1;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k, 1024, sm));
    printf("smem=%3d KB, 1024 thr -> occupancy %d block/SM  %s\n", kb, occ,
           occ == 1 ? "(1 block/SM: 78 blocks must map 1:1)" : "(co-residency possible)");
  }

  // empirical placement at 100 KB
  size_t sm = 100 * 1024;
  CK(cudaFuncSetAttribute((const void*)k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm));
  int* d = nullptr; CK(cudaMalloc(&d, 78 * 4));
  for (int it = 0; it < 5; ++it) {
    CK(cudaMemset(d, 0xff, 78 * 4));
    k<<<78, 1024, sm>>>(d);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    int h[78]; CK(cudaMemcpy(h, d, 78 * 4, cudaMemcpyDeviceToHost));
    int seen[256] = {0}, distinct = 0;
    for (int i = 0; i < 78; ++i) if (h[i] >= 0 && !seen[h[i]]++) distinct++;
    printf("run %d @100KB: %d distinct SMs for 78 blocks\n", it, distinct);
  }
  CK(cudaFree(d));
  return 0;
}
