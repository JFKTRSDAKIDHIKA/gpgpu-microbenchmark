#!/bin/bash
#SBATCH -J k6
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -o /fact_home/shuoaojia/gpgpu-microbenchmark/h20-nuca/k6-%j.out
#SBATCH -e /fact_home/shuoaojia/gpgpu-microbenchmark/h20-nuca/k6-%j.err
cd /fact_home/shuoaojia/gpgpu-microbenchmark/h20-nuca
cp src/pack_spread2.cu /tmp/ps_k6.cu
sed -i 's/UNROLL = 8, SMEM = 100 \* 1024/UNROLL = 16, SMEM = 120 * 1024/' /tmp/ps_k6.cu
echo "patched constants:"; sed -n '14p' /tmp/ps_k6.cu
/usr/local/cuda-12.8/bin/nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a /tmp/ps_k6.cu -o /tmp/ps_k6 -lcudart || { echo BUILD_FAIL; exit 1; }
/tmp/ps_k6
echo DONE
