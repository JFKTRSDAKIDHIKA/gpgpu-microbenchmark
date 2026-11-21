# H20 L2 NUCA 逆向 + Xbar 带宽实测

在 NVIDIA H20（GH100，78 SM，60 MiB L2）上把 **L2 的分区结构、分区哈希、访问粒度、容量曲线** 和 **L2/Xbar 带宽上限** 全部测了一遍。
所有数字都是实测，代码在本目录，原始输出在 `results/`，图在 `figures/`。

## TL;DR

| 项目 | 结果 |
|---|---|
| L2 分区数 | **2**（Home / Remote） |
| Home 分区原子延迟 | **301 cyc = 152 ns** |
| Remote 分区原子延迟 | **501 cyc = 253 ns** |
| 跨区落差 | **200 cyc = 101 ns（+67%）** |
| 双分区结构的方差占比（PC1） | **97.5%** |
| 分区哈希（20 位地址内） | **`b12 ⊕ b13 ⊕ b15 ⊕ b21 ⊕ b22`**，2048 点 **100%** |
| 哈希粒度 | **4 KB**（bit 12）；256 B / 512 B / 1 KB 块内标签恒定 |
| L2 命中延迟（load，逐行） | **280 cyc = 141 ns** |
| L2 命中延迟（指针追逐） | 295 cyc = 149 ns |
| DRAM 延迟 | **658–710 cyc = 332–359 ns** |
| L2 容量拐点 | 48–64 MB（`cudaDevAttrL2CacheSize` = 60 MiB） |
| 每 SM 读带宽（LSU `ld.global.cg.v4`） | **31.2 GB/s**（= 16 B/cycle × 1.98 GHz，发射率封顶） |
| 每 SM 读带宽（TMA 4-stage，L2 驻留） | **104.2 GB/s**（聚合 **8.13 TB/s**） |
| TMA 超出 L2 后 | 47–48 GB/s/SM（3.68–3.75 TB/s，HBM 受限） |

## 环境

- GPU：NVIDIA H20（GH100，`sm_90a`），78 SM，L2 60 MiB，1.98 GHz
- 驱动 610.43.02；CUDA 12.8 / 13.3；NCCL 2.30
- 编译：`nvcc -O3 -gencode arch=compute_90a,code=sm_90a`（TMA 必须 90a）
- 调度：Slurm，`-p ai --gres=gpu:N`

## 方法演进（关键，别踩这些坑）

1. **逐行 `clock64` 包 load** 测不出延迟：`clock64` 不消费 load 结果，ptxas 把时钟读取提到 load 之前，
   测出来恒为 **2 周期**。即使用谓词 `@p mov` + 依赖，ptxas 也会把它改写成 `CS2R + SEL`；
   真正救场的是 `ISETP` 依赖 load、warp 顺序发射（见 `src/l2lat_pred.cu`，SASS 里 `CS2R/LDG/ISETP/CS2R` 有序）。
2. **同地址自追逐**（`base + (val & 0x7F)`，填充值低 7 位为 0）才能让 64 次访问真正串行；
   普通独立测量会被流水线重叠（测出 6.5 cyc）。
3. **只读 load 会被本地副本“抹平”**：分区落差从 200 cyc 掉到 30 cyc，标签变成 0.375/0.625 的非平衡量。
4. **`atomicAdd` 会被硬件合并**（43 cyc）。必须用 **`atomicCAS(p,0,0)` + 地址依赖返回值**，
   才能强制每次 RMW 回到 Home 切片（401 cyc/次）。
5. 内联 asm 的 load 若结果未使用会被整段删除（SASS 里 0 条 LDG），改用 `volatile` 读。
6. 缓冲区必须 **2 MiB 对齐**，否则 bit 12–20 的哈希推断不成立。

## 实验一：双分区 NUCA（原子法）

`src/nuca_atomic.cu`：78 block（每 SM 一个）× 512 行，每格 64 次 `atomicCAS(0,0)` 链求均值。

| | SM 组 −（42） | SM 组 +（36） |
|---|---:|---:|
| 地址组 + | 301 | **501** |
| 地址组 − | **501** | 301 |

- Home 301 cyc / 152 ns，Remote 501 cyc / 253 ns，**Δ = 200 cyc = 101 ns**
- 每地址跨 SM 标准差 **104.5 cyc**（load 法只有 16 cyc）
- PC1 占 97.5% 方差；标签平衡 **256/512**
- 图：`figures/nuca_atomic_result.png`

对照（load 法，`src/multism3.cu` / `multism4.cu`）：落差只有 **30 cyc**，PC1 84%，标签 384/1024。

## 实验二：分区哈希（GF(2) 高斯消元）

对 512 个地址（一个 2 MiB 页内）的原子标签做 GF(2) 拟合，得到**唯一精确解**：

```
partition = b12 ⊕ b13 ⊕ b15
```

跨 4 个 2 MiB 页重复（`src/hash_pages.cu`，2048 点）：

| 页 | 与 `b12⊕b13⊕b15` 吻合 |
|---|---:|
| 0 | 100% |
| 1 | 100% 取反 |
| 2 | 100% 取反 |
| 3 | 100% |

翻转模式 `0,1,1,0` = `b21 ⊕ b22`，于是全地址空间（≤8 MiB）内的公式为：

```
partition = b12 ⊕ b13 ⊕ b15 ⊕ b21 ⊕ b22     （100%，2048/2048）
```

图：`figures/nuca_crossing.png`、`figures/nuca_topology.png`。

## 实验三：交错粒度

| 分组 | 组内标签纯度 |
|---|---:|
| 256 B | **100%** |
| 512 B | 100% |
| 1 KB | 100% |
| 2 KB | 0% |

→ bit 7（128 B）不参与分区选择，**最低有效位是 bit 12 = 4 KB**。

## 实验四：L2 延迟-容量曲线（gpgpu-microbenchmark test6）

仓库自带的 `hopper-h20/test6`（指针追逐，512 B → 1 GiB）：

| 工作集 | cyc/load | ns |
|---|---:|---:|
| ≤128 KB（L1） | 32.0 | 16.2 |
| 0.5–16 MB（L2） | **275.3** | **139.0** |
| 32 MB | 420.8 | 212.5 |
| 64 MB | 648.3 | 327.4 |
| ≥128 MB（HBM） | 680–685 | 343–346 |

图：`figures/h20_test6_latency.png`。注意这是**平均值**，无法分辨分区（因此看不�� NUCA）。

## 实验五：带宽 —— LSU 墙 vs TMA

`src/xbar_bw.cu`（LSU，128-bit 向量加载）四种流量编排：

| 模式 | 活跃 SM | 聚合 | 每 SM |
|---|---:|---:|---:|
| All-Local | 78 | 2432 GB/s | 31.2 GB/s |
| Uni-Cross | 42 | 1310 GB/s | 31.2 GB/s |
| Bi-Cross | 78 | 2433 GB/s | 31.2 GB/s |
| Asymmetric | 78 | 2435 GB/s | 31.2 GB/s |

**每 SM 恒定 31.2 GB/s**（= 16 B/cycle），说明被 LSU 发射率封顶，与本地/跨区无关；
Bi-Cross 恰好是 Uni-Cross 的 2 倍 → 跨区通路全双工、非共享。

`src/tma_bw.cu`（`cp.async.bulk`，1 线程/SM，4 级 × 32 KB）：

| 工作集 | 每 SM | 聚合 |
|---|---:|---:|
| **39 MB（L2 驻留）** | **104.2 GB/s** | **8127 GB/s** |
| 78 MB | 48.1 GB/s | 3750 GB/s |
| 624 MB | 47.2 GB/s | 3679 GB/s |
| 39 MB（6 级流水） | 96.9 GB/s | 7554 GB/s |

→ LSU 的 31.2 GB/s 是**发射率墙**，不是硬件上限；TMA 把单 SM 拉到 **104 GB/s**，
L2 侧上限 **~8.1 TB/s**（HBM 的两倍）。6 级流水反而更慢 → 不是"在途字节不足"限制。

## 实验六：NCU fabric 计数器（未闭环）

`ncu --query-metrics --chip gh100` 确认存在：
`lts__ltcfabric2lts_cycles_active`、`lts__ltcfabric2ltc_cycles_active`、`lts__t_requests_srcunit_ltcfabric`。

实测 All-Local 与 Bi-Cross 几乎一样（fabric 请求 41,321 vs 总读请求 1.42e9 = 0.003%），
**没有观察到跨区流量差异**。两种可能：计数器语义不是"SM→远端分区"，或 64 MiB 分组时
`b12⊕b13⊕b15⊕b21⊕b22` 已超出验证范围（bit 23–25 未验证）。**这是负面结果，待查。**

## 目录

```
h20-nuca/
  src/       探针源码（.cu）：L2 延迟、NUCA 矩阵、哈希、带宽、TMA
  analysis/  分析脚本（.py）：GF(2) 消元、SVD/聚类、粒度、绘图
  scripts/   Slurm 提交脚本
  figures/   结果图
  results/   test6 原始输出、每 SM 均值、SM 分组
```

## 复现要点

```bash
# 编译（TMA 需要 sm_90a）
nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a src/nuca_atomic.cu -o nuca_atomic -lcudart

# 需要独占 1 张卡；78 个 block 会自动各占一个 SM
sbatch scripts/run_atomic.sbatch

# 分析
python3 analysis/analyze_atomic.py      # 2x2 + PC1
python3 analysis/hash_atomic.py         # GF(2) 拟合
python3 analysis/analyze_pages.py       # 跨页验证
```

## 未完成 / 下一步

- 96 个 L2 slice 的冲突探测（锚点争用 + 扫描）；
- 严格的跨区/本地带宽分离（需要 tensor-map TMA 按分区行步长取样）；
- 写通路带宽（`cp.reduce.async.bulk` / `st.global.wb`）；
- NCU fabric 计数器闭环（先把分组范围收回到已验证的 8 MiB）；
- 多卡/多节点上的迁移（当前全部单卡）。
