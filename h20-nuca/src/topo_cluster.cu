// topo_cluster.cu -- recover the GPC partition of the 78 SMs using Thread Block Clusters.
// A cluster is guaranteed to fit inside one GPC, so the set of %smid values that
// appear together in one cluster is a GPC subset. Repeat many launches and take
// connected components -> the GPC groups.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

static const int CLUSTER = 8;
static const int NCLU    = 1;          // one cluster per launch: no merge ambiguity
static const int NBLK    = CLUSTER * NCLU;
static const int LAUNCH  = 4000;
static const int SMEM    = 120 * 1024; // 1 block per SM

__global__ void probe(int* out) {
  unsigned int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  out[blockIdx.x] = (int)smid;
}

struct DSU {
  std::vector<int> p;
  explicit DSU(int n) : p(n) { for (int i = 0; i < n; ++i) p[i] = i; }
  int f(int x) { return p[x] == x ? x : p[x] = f(p[x]); }
  void u(int a, int b) { a = f(a); b = f(b); if (a != b) p[a] = b; }
};

int main() {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("device=%s sm=%d cluster=%d\n", p.name, p.multiProcessorCount, CLUSTER);

  CK(cudaFuncSetAttribute((const void*)probe, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  CK(cudaFuncSetAttribute((const void*)probe, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));

  int* d = nullptr; CK(cudaMalloc(&d, NBLK * 4));
  std::vector<int> h(NBLK);

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NBLK);
  cfg.blockDim = dim3(1);
  cfg.dynamicSmemBytes = SMEM;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = CLUSTER;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;

  DSU dsu(128);
  std::map<std::vector<int>, int> seen;
  int ok = 0;

  for (int it = 0; it < LAUNCH; ++it) {
    CK(cudaMemset(d, 0xff, NBLK * 4));
    cudaError_t e = cudaLaunchKernelEx(&cfg, probe, d);
    if (e != cudaSuccess) { printf("launch failed at %d: %s\n", it, cudaGetErrorString(e)); break; }
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(h.data(), d, NBLK * 4, cudaMemcpyDeviceToHost));

    for (int c = 0; c < NCLU; ++c) {
      std::vector<int> s;
      for (int k = 0; k < CLUSTER; ++k) s.push_back(h[c * CLUSTER + k]);
      bool bad = false;
      for (int x : s) if (x < 0 || x > 127) bad = true;
      if (bad) continue;
      std::sort(s.begin(), s.end());
      s.erase(std::unique(s.begin(), s.end()), s.end());
      if ((int)s.size() < CLUSTER) continue;      // duplicate SM -> skip
      seen[s]++;
      for (size_t i = 1; i < s.size(); ++i) dsu.u(s[0], s[i]);
      ok++;
    }
  }
  printf("valid clusters collected: %d over %d launches, distinct sets: %zu\n",
         ok, LAUNCH, seen.size());

  std::map<int, std::vector<int>> groups;
  for (int sm = 0; sm < 128; ++sm) {
    // only include SMs that actually appeared
  }
  std::vector<int> appeared;
  for (auto& kv : seen) for (int x : kv.first) appeared.push_back(x);
  std::sort(appeared.begin(), appeared.end());
  appeared.erase(std::unique(appeared.begin(), appeared.end()), appeared.end());
  for (int sm : appeared) groups[dsu.f(sm)].push_back(sm);

  printf("\nGPC groups (connected components of cluster co-occurrence): %zu\n", groups.size());
  int gi = 0;
  for (auto& kv : groups) {
    printf("  group %d (%2zu SMs): ", gi++, kv.second.size());
    for (int x : kv.second) printf("%d ", x);
    printf("\n");
  }

  // how often each pair co-occurred (sanity)
  FILE* f = fopen("cluster_sets.csv", "w");
  for (auto& kv : seen) {
    for (size_t i = 0; i < kv.first.size(); ++i)
      fprintf(f, "%s%d", i ? "," : "", kv.first[i]);
    fprintf(f, ",%d\n", kv.second);
  }
  fclose(f);
  printf("\nwrote cluster_sets.csv\n");
  CK(cudaFree(d));
  return 0;
}
