#!/bin/bash
#SBATCH -J multism4
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -o /fact_home/shuoaojia/nuca/multism4-%j.out
#SBATCH -e /fact_home/shuoaojia/nuca/multism4-%j.err
cd /fact_home/shuoaojia/nuca
echo "NODE=$(hostname)"
./multism4
echo DONE
