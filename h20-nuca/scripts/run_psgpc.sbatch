#!/bin/bash
#SBATCH -J psgpc
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH -o /fact_home/shuoaojia/nuca/psgpc-%j.out
#SBATCH -e /fact_home/shuoaojia/nuca/psgpc-%j.err
cd /fact_home/shuoaojia/nuca
echo "NODE=$(hostname)  smem=120KB occupancy=1  1024thr unroll16  (each GPC filled with all its SMs)"
for g in 0 1 2 3 4 5 6; do
  ./ps_gpc $g 0 0
done
echo DONE
