// verify_pick.cu -- does every SM in the chosen set really get one working block?
// Same launch shape as ps_occ1 (78 blocks, 1024 thr, 120 KB smem -> occupancy 1).

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

static const int SMEM = 120 * 1024;

__global__ void v(const int* active, int* smid_out, int* ran_out) {
  extern __shared__ char smem[];
  unsigned int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  if (threadIdx.x == 0) smid_out[blockIdx.x] = (int)smid;
  if (!active[smid]) return;
  if (threadIdx.x == 0) ran_out[blockIdx.x] = 1;
}

static const std::vector<std::vector<int>> GPC = {
  {0,1,14,15,28,29,42,43},
  {2,3,16,17,30,31,44,45},
  {4,5,18,19,32,33,46,47},
  {6,7,20,21,34,35,48,49},
  {8,9,22,23,36,37,50,51,56},
  {10,11,24,25,38,39,52,53,58,59},
  {12,13,26,27,40,41,54,55,60,61,62,63},
};

int main() {
  for (int K : {1, 2, 4, 8, 16}) {
    std::vector<int> want;
    for (auto& g : GPC) for (int s : g) if ((int)want.size() < K) want.push_back(s);
    std::vector<int> act(128, 0);
    for (int s : want) act[s] = 1;

    int* dact = nullptr; int* dsm = nullptr; int* dran = nullptr;
    CK(cudaMalloc(&dact, 128 * 4)); CK(cudaMalloc(&dsm, 78 * 4)); CK(cudaMalloc(&dran, 78 * 4));
    CK(cudaMemcpy(dact, act.data(), 128 * 4, cudaMemcpyHostToDevice));
    CK(cudaFuncSetAttribute((const void*)v, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));

    CK(cudaMemset(dran, 0, 78 * 4));
    v<<<78, 1024, SMEM>>>(dact, dsm, dran);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());

    int sm[78], ran[78];
    CK(cudaMemcpy(sm, dsm, 78 * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(ran, dran, 78 * 4, cudaMemcpyDeviceToHost));

    std::vector<int> got;
    int total_ran = 0;
    for (int i = 0; i < 78; ++i) if (ran[i]) { got.push_back(sm[i]); total_ran++; }
    std::sort(got.begin(), got.end());

    std::vector<int> w = want; std::sort(w.begin(), w.end());
    bool same = (got == w);
    printf("K=%2d  want %2zu SMs -> got %2d working blocks  match=%s\n",
           K, want.size(), total_ran, same ? "YES" : "NO");
    if (!same) {
      printf("   want:"); for (int s : w) printf(" %d", s);
      printf("\n   got :"); for (int s : got) printf(" %d", s);
      printf("\n");
    }
    CK(cudaFree(dact)); CK(cudaFree(dsm)); CK(cudaFree(dran));
  }
  return 0;
}
