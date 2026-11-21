#!/bin/bash
#SBATCH -J chase
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -o /fact_home/shuoaojia/l2lat/chase-%j.out
#SBATCH -e /fact_home/shuoaojia/l2lat/chase-%j.err
cd /fact_home/shuoaojia/l2lat
echo "NODE=$(hostname)"
./chase
echo DONE
