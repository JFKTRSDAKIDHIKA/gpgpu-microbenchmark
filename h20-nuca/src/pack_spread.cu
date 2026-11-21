// pack_spread.cu -- aggregate read bandwidth vs number of active SMs.
// packed : K blocks launched as ONE cluster -> guaranteed same GPC
// spread : K blocks in a plain grid -> scheduler spreads them over GPCs

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

static const int LINES = 512;
static const int ITERS = 4000;
static const int UNROLL = 8;
static const int SMEM = 100 * 1024;      // keep 1 block per SM

__device__ __forceinline__ float4 ldcg4(const float4* p) {
  float4 v;
  asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p));
  return v;
}

__global__ void rd(const float4* w, unsigned long long* out) {
  extern __shared__ char smem[];
  const int tid = threadIdx.x;
  float4 acc = make_float4(0, 0, 0, 0);
  __syncthreads();
  unsigned long long t0 = clock64();
  for (int it = 0; it < ITERS; ++it) {
    #pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
      int i = (tid + (it * UNROLL + u) * blockDim.x) & (LINES * 8 - 1);
      float4 v = ldcg4(w + i);
      acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
  }
  unsigned long long t1 = clock64();
  if (tid == 0) out[blockIdx.x] = t1 - t0;
  if (acc.x == -12345.f) ((float4*)w)[0] = acc;
}

int main(int argc, char** argv) {
  const int K = (argc > 1) ? atoi(argv[1]) : 1;
  const int packed = (argc > 2) ? atoi(argv[2]) : 0;
  int mhz = 0; CK(cudaDeviceGetAttribute(&mhz, cudaDevAttrClockRate, 0));
  double ghz = mhz / 1e6;

  CK(cudaFuncSetAttribute((const void*)rd, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  float4* w = nullptr; CK(cudaMalloc(&w, LINES * 128));
  CK(cudaMemset(w, 0x3c, LINES * 128));
  unsigned long long* out = nullptr; CK(cudaMalloc(&out, K * 8));

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(K); cfg.blockDim = dim3(512); cfg.dynamicSmemBytes = SMEM;
  cudaLaunchAttribute a[1];
  a[0].id = cudaLaunchAttributeClusterDimension;
  a[0].val.clusterDim.x = packed ? K : 1;
  a[0].val.clusterDim.y = 1;
  a[0].val.clusterDim.z = 1;
  cfg.attrs = a; cfg.numAttrs = 1;

  cudaError_t e = cudaLaunchKernelEx(&cfg, rd, w, out);
  if (e != cudaSuccess) { printf("K=%d %s launch failed: %s\n", K, packed ? "packed" : "spread",
                                 cudaGetErrorString(e)); return 1; }
  CK(cudaDeviceSynchronize());
  if (cudaLaunchKernelEx(&cfg, rd, w, out) != cudaSuccess) { printf("relaunch failed\n"); return 1; }
  CK(cudaDeviceSynchronize());

  unsigned long long* h = (unsigned long long*)malloc(K * 8);
  CK(cudaMemcpy(h, out, K * 8, cudaMemcpyDeviceToHost));
  unsigned long long mx = 0;
  for (int i = 0; i < K; ++i) if (h[i] > mx) mx = h[i];

  double ops = (double)ITERS * UNROLL * 512 * K;
  double bytes = ops * 16.0;
  printf("K=%2d %-6s cycles=%llu  %.2f B/cycle aggregate  %.2f B/cycle per SM  (%.0f GB/s)\n",
         K, packed ? "packed" : "spread", mx, bytes / mx, bytes / mx / K,
         bytes / mx * ghz);
  CK(cudaFree(w)); CK(cudaFree(out));
  return 0;
}
