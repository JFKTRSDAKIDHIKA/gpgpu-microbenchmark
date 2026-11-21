#!/bin/bash
#SBATCH -J tmabw6
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o /fact_home/shuoaojia/nuca/tmabw6-%j.out
#SBATCH -e /fact_home/shuoaojia/nuca/tmabw6-%j.err
cd /fact_home/shuoaojia/nuca
echo "NODE=$(hostname)"
./tma_bw6 20000 512
echo DONE