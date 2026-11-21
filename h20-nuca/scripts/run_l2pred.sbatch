#!/bin/bash
#SBATCH -J l2pred
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -o /fact_home/shuoaojia/l2lat/l2pred-%j.out
#SBATCH -e /fact_home/shuoaojia/l2lat/l2pred-%j.err
cd /fact_home/shuoaojia/l2lat
echo "NODE=$(hostname)"
./l2lat_pred
echo DONE
