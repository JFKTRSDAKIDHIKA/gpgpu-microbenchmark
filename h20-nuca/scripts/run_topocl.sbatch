#!/bin/bash
#SBATCH -J topocl
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o /fact_home/shuoaojia/nuca/topocl-%j.out
#SBATCH -e /fact_home/shuoaojia/nuca/topocl-%j.err
cd /fact_home/shuoaojia/nuca
echo "NODE=$(hostname)"
./topo_cluster
echo DONE
