#!/bin/bash
#SBATCH -J psocc1
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH -o /fact_home/shuoaojia/nuca/psocc1-%j.out
#SBATCH -e /fact_home/shuoaojia/nuca/psocc1-%j.err
cd /fact_home/shuoaojia/nuca
echo "NODE=$(hostname)  smem=120KB (occupancy 1)  threads=1024  unroll=16"
for K in 1 2 4 6 8 12 16 32; do ./ps_occ1 $K 1 0; done
echo DONE