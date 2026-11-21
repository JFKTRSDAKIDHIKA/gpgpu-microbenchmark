// pack_spread2.cu -- K-SM scaling with per-SM 64 KiB windows (no hot spot) and
// branch-free read / write / 1:1 duplex kernels.
//   args: <K> <packed 0|1> <mode 0=read 1=write 2=mixed>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

static const int ITERS = 4000, UNROLL = 8, SMEM = 100 * 1024;
static const int WLINES = 512;                  // 64 KiB per SM

__device__ __forceinline__ float4 ldcg4(const float4* p) {
  float4 v;
  asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void stcg4(float4* p, float4 v) {
  asm volatile("st.global.cg.v4.f32 [%0], {%1,%2,%3,%4};"
               :: "l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}

template <int MODE>
__global__ void work(float4* buf, const int* active, unsigned long long* out) {
  extern __shared__ char smem[];
  unsigned int smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  if (!active[smid]) return;

  const int tid = threadIdx.x;
  float4* w = buf + (size_t)(smid & 127) * WLINES * 8;
  float4 acc = make_float4(0, 0, 0, 0);
  float4 val = make_float4(1, 2, 3, 4);

  __syncthreads();
  unsigned long long t0 = clock64();
  for (int it = 0; it < ITERS; ++it) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int i = (tid + (it * UNROLL + u) * blockDim.x) & (WLINES * 8 - 1);
      if (MODE == 0) {
        float4 v = ldcg4(w + i); acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
      } else if (MODE == 1) {
        stcg4(w + i, val);
      } else {
        float4 v = ldcg4(w + i); acc.x += v.x;
        stcg4(w + i, val);
      }
    }
  }
  unsigned long long t1 = clock64();
  if (tid == 0) out[blockIdx.x] = t1 - t0;
  if (acc.x == -12345.f) w[0] = acc;
}

// GPC groups recovered from the cluster experiment
static const std::vector<std::vector<int>> GPC = {
  {0,1,14,15,28,29,42,43},
  {2,3,16,17,30,31,44,45},
  {4,5,18,19,32,33,46,47},
  {6,7,20,21,34,35,48,49},
  {8,9,22,23,36,37,50,51,56},
  {10,11,24,25,38,39,52,53,58,59},
  {12,13,26,27,40,41,54,55,60,61,62,63},
};

static std::vector<int> pick(int K, bool packed) {
  std::vector<int> out;
  if (packed) {
    for (auto& g : GPC) for (int s : g) if ((int)out.size() < K) out.push_back(s);
  } else {
    size_t idx[7] = {0};
    while ((int)out.size() < K) {
      bool any = false;
      for (int g = 0; g < 7 && (int)out.size() < K; ++g)
        if (idx[g] < GPC[g].size()) { out.push_back(GPC[g][idx[g]++]); any = true; }
      if (!any) break;
    }
  }
  return out;
}

int main(int argc, char** argv) {
  const int K = (argc > 1) ? atoi(argv[1]) : 1;
  const int packed = (argc > 2) ? atoi(argv[2]) : 0;
  const int mode = (argc > 3) ? atoi(argv[3]) : 0;

  int mhz = 0; CK(cudaDeviceGetAttribute(&mhz, cudaDevAttrClockRate, 0));
  double ghz = mhz / 1e6;

  std::vector<int> set = pick(K, packed != 0);
  std::vector<int> act(128, 0);
  for (int s : set) act[s] = 1;

  int* dact = nullptr; CK(cudaMalloc(&dact, 128 * 4));
  CK(cudaMemcpy(dact, act.data(), 128 * 4, cudaMemcpyHostToDevice));
  float4* buf = nullptr; CK(cudaMalloc(&buf, (size_t)128 * WLINES * 8 * 16));
  CK(cudaMemset(buf, 0x3c, (size_t)128 * WLINES * 8 * 16));
  unsigned long long* out = nullptr; CK(cudaMalloc(&out, 78 * 8));

  const void* fn = (mode == 0) ? (const void*)work<0> : (mode == 1) ? (const void*)work<1> : (const void*)work<2>;
  CK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(78); cfg.blockDim = dim3(512); cfg.dynamicSmemBytes = SMEM;
  cudaLaunchAttribute a[1];
  a[0].id = cudaLaunchAttributeClusterDimension;
  a[0].val.clusterDim.x = 1;   // selection is by smid, no cluster needed
  a[0].val.clusterDim.y = 1; a[0].val.clusterDim.z = 1;
  cfg.attrs = a; cfg.numAttrs = 1;

  cudaError_t e;
  if (mode == 0) e = cudaLaunchKernelEx(&cfg, work<0>, buf, dact, out);
  else if (mode == 1) e = cudaLaunchKernelEx(&cfg, work<1>, buf, dact, out);
  else e = cudaLaunchKernelEx(&cfg, work<2>, buf, dact, out);
  if (e != cudaSuccess) { printf("launch failed: %s\n", cudaGetErrorString(e)); return 1; }
  CK(cudaDeviceSynchronize());

  if (mode == 0) e = cudaLaunchKernelEx(&cfg, work<0>, buf, dact, out);
  else if (mode == 1) e = cudaLaunchKernelEx(&cfg, work<1>, buf, dact, out);
  else e = cudaLaunchKernelEx(&cfg, work<2>, buf, dact, out);
  CK(cudaDeviceSynchronize());

  std::vector<unsigned long long> h(78);
  CK(cudaMemcpy(h.data(), out, 78 * 8, cudaMemcpyDeviceToHost));
  unsigned long long mx = 0;
  for (unsigned long long v : h) mx = std::max(mx, v);

  double ops = (double)ITERS * UNROLL * 512 * (int)set.size();
  double bytes = ops * 16.0 * ((mode == 2) ? 2.0 : 1.0);
  const char* mn = (mode == 0) ? "read " : (mode == 1) ? "write" : "1:1  ";
  printf("K=%2zu %-6s %s cycles=%llu  %.2f B/cyc aggregate  %.2f per SM  (%.0f GB/s)\n",
         set.size(), packed ? "packed" : "spread", mn, mx,
         bytes / mx, bytes / mx / set.size(), bytes / mx * ghz);
  CK(cudaFree(buf)); CK(cudaFree(out)); CK(cudaFree(dact));
  return 0;
}
