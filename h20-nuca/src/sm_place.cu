// sm_place.cu -- does the pack_spread2 launch config really put 1 block per SM?
// Same launch shape (78 blocks, 512 threads, 100 KB smem); every block records its
// %smid so the host can count distinct SMs and detect co-residency.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

static const int SMEM = 100 * 1024;
static const int THREADS = 1024;

__global__ void who(int* out) {
  extern __shared__ char smem[];
  unsigned int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  if (threadIdx.x == 0) out[blockIdx.x] = (int)smid;
}

int main() {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("device=%s sm=%d  smem/SM=%zu KB  launch: 78 blocks x %d threads, %d KB smem\n",
         p.name, p.multiProcessorCount, p.sharedMemPerMultiprocessor / 1024, THREADS, SMEM / 1024);
  int occ = 0;
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, who, THREADS, SMEM));
  printf("occupancy: %d block(s) per SM can co-reside at this smem size\n", occ);

  int* d = nullptr; CK(cudaMalloc(&d, 78 * 4));
  CK(cudaFuncSetAttribute((const void*)who, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));

  for (int it = 0; it < 5; ++it) {
    CK(cudaMemset(d, 0xff, 78 * 4));
    who<<<78, THREADS, SMEM>>>(d);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    std::vector<int> h(78);
    CK(cudaMemcpy(h.data(), d, 78 * 4, cudaMemcpyDeviceToHost));
    std::vector<int> s = h;
    std::sort(s.begin(), s.end());
    s.erase(std::unique(s.begin(), s.end()), s.end());
    printf("run %d: %zu distinct SMs for 78 blocks %s\n", it, s.size(),
           s.size() == 78 ? "(1 block/SM OK)" : "(CO-RESIDENT!)");
  }
  CK(cudaFree(d));
  return 0;
}
