#!/bin/bash
#SBATCH -J smport
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o /fact_home/shuoaojia/nuca/smport-%j.out
#SBATCH -e /fact_home/shuoaojia/nuca/smport-%j.err
cd /fact_home/shuoaojia/nuca
echo "NODE=$(hostname)"
for m in 0 1 2; do ./sm_port $m; done
echo "--- ncu peaks ---"
/usr/local/cuda-12.8/bin/ncu --metrics l1tex__m_xbar2l1tex_read_bytes.sum.peak_sustained,l1tex__m_l1tex2xbar_write_bytes.sum.peak_sustained ./sm_port 0 2>&1 | tail -6
echo DONE
