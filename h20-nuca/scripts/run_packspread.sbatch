#!/bin/bash
#SBATCH -J packspread
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o /fact_home/shuoaojia/nuca/packspread-%j.out
#SBATCH -e /fact_home/shuoaojia/nuca/packspread-%j.err
cd /fact_home/shuoaojia/nuca
echo "NODE=$(hostname)"
for K in 1 2 4 8; do
  ./pack_spread $K 0
  ./pack_spread $K 1
done
echo DONE
