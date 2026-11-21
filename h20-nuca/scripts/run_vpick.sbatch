#!/bin/bash
#SBATCH -J vpick
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -o /fact_home/shuoaojia/nuca/vpick-%j.out
#SBATCH -e /fact_home/shuoaojia/nuca/vpick-%j.err
cd /fact_home/shuoaojia/nuca
echo "NODE=$(hostname)"
./verify_pick
echo DONE