#!/bin/bash
#SBATCH -J topocl3
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -o /fact_home/shuoaojia/nuca/topocl3-%j.out
#SBATCH -e /fact_home/shuoaojia/nuca/topocl3-%j.err
cd /fact_home/shuoaojia/nuca
echo "NODE=$(hostname)"
./topo_cluster3
echo DONE