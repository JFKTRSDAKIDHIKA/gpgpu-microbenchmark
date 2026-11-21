#!/bin/bash
#SBATCH -J nuca_atomic
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -o /fact_home/shuoaojia/nuca/nuca_atomic-%j.out
#SBATCH -e /fact_home/shuoaojia/nuca/nuca_atomic-%j.err
cd /fact_home/shuoaojia/nuca
echo "NODE=$(hostname)"
./nuca_atomic
echo DONE
