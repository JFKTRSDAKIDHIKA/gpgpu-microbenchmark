#!/bin/bash
#SBATCH -J hashpages
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -o /fact_home/shuoaojia/nuca/hashpages-%j.out
#SBATCH -e /fact_home/shuoaojia/nuca/hashpages-%j.err
cd /fact_home/shuoaojia/nuca
echo "NODE=$(hostname)"
./hash_pages
echo DONE
