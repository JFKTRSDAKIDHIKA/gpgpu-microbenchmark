#!/bin/bash
#SBATCH -J topocl2
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -o /fact_home/shuoaojia/nuca/topocl2-%j.out
#SBATCH -e /fact_home/shuoaojia/nuca/topocl2-%j.err
cd /fact_home/shuoaojia/nuca
echo "NODE=$(hostname)"
./topo_cluster2
echo DONE