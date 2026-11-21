#!/bin/bash
#SBATCH -J pspair
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH -o /fact_home/shuoaojia/nuca/pspair-%j.out
#SBATCH -e /fact_home/shuoaojia/nuca/pspair-%j.err
cd /fact_home/shuoaojia/nuca
echo "NODE=$(hostname)  smem=120KB occupancy=1  1024thr unroll16  mode=read"
echo "-- single SM baseline --"
./ps_pair 0 0 0
./ps_pair 14 0 0
echo "-- candidate same-TPC pairs (consecutive ids) --"
./ps_pair 0 0 0 1
./ps_pair 14 0 0 15
./ps_pair 28 0 0 29
./ps_pair 12 0 0 13
echo "-- same GPC, different TPC --"
./ps_pair 0 0 0 14
./ps_pair 0 0 0 28
./ps_pair 0 0 0 42
echo "-- different GPC --"
./ps_pair 0 0 0 2
./ps_pair 0 0 0 16
echo DONE
