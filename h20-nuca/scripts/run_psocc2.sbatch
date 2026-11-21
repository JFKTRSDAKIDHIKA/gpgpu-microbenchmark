#!/bin/bash
#SBATCH -J psocc2
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH -o /fact_home/shuoaojia/nuca/psocc2-%j.out
#SBATCH -e /fact_home/shuoaojia/nuca/psocc2-%j.err
cd /fact_home/shuoaojia/nuca
echo "NODE=$(hostname)  smem=120KB occupancy=1  1024thr unroll16"
for K in 7 8 10; do
  ./ps_occ1 $K 1 0
  ./ps_occ1 $K 0 0
done
echo DONE