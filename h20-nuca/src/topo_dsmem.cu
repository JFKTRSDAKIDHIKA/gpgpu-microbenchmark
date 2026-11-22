// topo_dsmem.cu -- 8x8 DSMEM latency matrix inside one GPC, by pointer chasing.
// For every ordered pair (src,dst) exactly one one-way probe is in flight at a time,
// serialized by cluster barriers. The next hop address depends on the previous
// returned value (strict RAW), so the chain cannot be pipelined or folded away.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA %d %s\n", __LINE__, cudaGetErrorString(e_)); exit(1);} } while (0)

static const int C = 8;
static const int SMEM = 120 * 1024;   // occupancy 1 -> one block per SM
static const int REPS = 256;          // 拉长一点摊薄循环和计时开销

__device__ __forceinline__ uint32_t sa(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ uint32_t mapa(uint32_t addr, uint32_t rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
  return r;
}

__device__ __forceinline__ uint64_t rdclk() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
  return t;
}

__device__ __forceinline__ uint32_t ld_remote_u32(uint32_t addr) {
  uint32_t val;
  asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(val) : "r"(addr) : "memory");
  return val;
}

// 多传一个 host 参数 step（传 4），防止 ptxas 常量折叠优化掉依赖链
__global__ void probe(int* out, int* smids, int step) {
  extern __shared__ __align__(16) int smem[];
  __shared__ __align__(64) uint32_t ring[16];

  unsigned int smid, rank;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));

  if (threadIdx.x == 0) {
    // 构造 16 元素的字节偏移环：0 -> 4 -> 8 -> ... -> 60 -> 0
    for (int i = 0; i < 16; ++i) {
      ring[i] = ((i + 1) & 15) * (uint32_t)step;
    }
    smids[rank] = (int)smid;
    out[rank * 9 + 8] = (int)smid;
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive.aligned;\n\tbarrier.cluster.wait.aligned;" ::: "memory");

  uint32_t base = sa(ring);

  // 关键：把 (src, dst) 逐对串行化，保证同一时刻整个 GPC 内只有 1 条单向探针在跑
  // 注意：绝不能在循环前写 if (threadIdx.x != 0) return; 否则 cluster barrier 会死锁
  for (int src = 0; src < C; ++src) {
    for (int dst = 0; dst < C; ++dst) {
      asm volatile("barrier.cluster.arrive.aligned;\n\tbarrier.cluster.wait.aligned;" ::: "memory");

      if ((int)rank == src && threadIdx.x == 0) {
        if (dst == src) {
          out[src * 9 + dst] = -1;
        } else {
          uint32_t ra = mapa(base, (uint32_t)dst);
          uint32_t off = 0;

          // warm up
          for (int w = 0; w < 16; ++w) off = ld_remote_u32(ra + off);

          uint64_t t0 = rdclk();
          #pragma unroll 16
          for (int r = 0; r < REPS; ++r) {
            off = ld_remote_u32(ra + off); // 严格 RAW 依赖：下一跳地址依赖上一跳返回值
          }
          uint64_t t1 = rdclk();

          double v = (double)(t1 - t0) / REPS;
          if (off == 0x7fffffffu) v = -1; // 防死代码消除
          out[src * 9 + dst] = (int)lrint(v);
        }
      }
    }
  }

  // 结束前统一同步，防止有 block 提前退出导致远端 smem 失效
  asm volatile("barrier.cluster.arrive.aligned;\n\tbarrier.cluster.wait.aligned;" ::: "memory");
}

int main() {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("device=%s sm=%d  REPS=%d\n", p.name, p.multiProcessorCount, REPS);

  int* out = nullptr; int* smids = nullptr;
  CK(cudaMalloc(&out, C * 9 * 4)); CK(cudaMalloc(&smids, C * 4));
  CK(cudaFuncSetAttribute((const void*)probe, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));

  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(C); cfg.blockDim = dim3(64); cfg.dynamicSmemBytes = SMEM;
  cudaLaunchAttribute a[1];
  a[0].id = cudaLaunchAttributeClusterDimension;
  a[0].val.clusterDim.x = C; a[0].val.clusterDim.y = 1; a[0].val.clusterDim.z = 1;
  cfg.attrs = a; cfg.numAttrs = 1;

  cudaError_t e = cudaLaunchKernelEx(&cfg, probe, out, smids, 4);
  if (e != cudaSuccess) { printf("launch failed: %s\n", cudaGetErrorString(e)); return 1; }
  CK(cudaDeviceSynchronize());

  int m[C * 9];
  CK(cudaMemcpy(m, out, C * 9 * 4, cudaMemcpyDeviceToHost));
  int s[C];
  CK(cudaMemcpy(s, smids, C * 4, cudaMemcpyDeviceToHost));

  printf("\nSM ids in this cluster:"); for (int i = 0; i < C; ++i) printf(" %d", s[i]);
  printf("\n\nDSMEM pointer-chase latency matrix (cyc/hop):\n     ");
  for (int j = 0; j < C; ++j) printf("  r%d ", j);
  printf("\n");
  for (int i = 0; i < C; ++i) {
    printf("r%d  ", i);
    for (int j = 0; j < C; ++j) printf("%5d ", m[i * 9 + j]);
    printf("\n");
  }
  printf("\nrank -> smid:");
  for (int i = 0; i < C; ++i) printf("  r%d=%d", i, s[i]);
  printf("\n");
  CK(cudaFree(out)); CK(cudaFree(smids));
  return 0;
}
