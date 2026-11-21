#!/bin/bash
#SBATCH -J smplace
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -o /fact_home/shuoaojia/nuca/smplace-%j.out
#SBATCH -e /fact_home/shuoaojia/nuca/smplace-%j.err
cd /fact_home/shuoaojia/nuca
echo "NODE=$(hostname)"
./sm_place
echo DONE