#!/bin/bash
#SBATCH -J l2lat
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -o /fact_home/shuoaojia/l2lat/l2lat-%j.out
#SBATCH -e /fact_home/shuoaojia/l2lat/l2lat-%j.err
cd /fact_home/shuoaojia/l2lat
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES"
./l2lat
echo DONE
