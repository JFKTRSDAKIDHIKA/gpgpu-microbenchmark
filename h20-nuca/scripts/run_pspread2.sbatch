#!/bin/bash
#SBATCH -J pspread2
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o /fact_home/shuoaojia/nuca/pspread2-%j.out
#SBATCH -e /fact_home/shuoaojia/nuca/pspread2-%j.err
cd /fact_home/shuoaojia/nuca
echo "NODE=$(hostname)"
echo "== single SM port =="
./pack_spread2 1 0 0
./pack_spread2 1 0 1
./pack_spread2 1 0 2
echo "== read scaling, packed vs spread =="
for K in 8 16 32 63; do
  ./pack_spread2 $K 1 0
  ./pack_spread2 $K 0 0
done
echo DONE
