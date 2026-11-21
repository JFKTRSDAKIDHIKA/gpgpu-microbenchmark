#!/bin/bash
#SBATCH -J dsmem
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -o /fact_home/shuoaojia/nuca/dsmem-%j.out
#SBATCH -e /fact_home/shuoaojia/nuca/dsmem-%j.err
cd /fact_home/shuoaojia/nuca
echo "NODE=$(hostname)"
./topo_dsmem
echo DONE