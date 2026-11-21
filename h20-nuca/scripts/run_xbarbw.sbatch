#!/bin/bash
#SBATCH -J xbarbw
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o /fact_home/shuoaojia/nuca/xbarbw-%j.out
#SBATCH -e /fact_home/shuoaojia/nuca/xbarbw-%j.err
cd /fact_home/shuoaojia/nuca
echo "NODE=$(hostname)"
for m in 0 1 2 3; do ./xbar_bw $m; done
echo DONE
