#!/bin/bash
#SBATCH -J tmabw
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o /fact_home/shuoaojia/nuca/tmabw-%j.out
#SBATCH -e /fact_home/shuoaojia/nuca/tmabw-%j.err
cd /fact_home/shuoaojia/nuca
echo "NODE=$(hostname)"
./tma_bw 20000 512
./tma_bw 20000 1024
./tma_bw 5000 8192
echo DONE