#!/bin/bash
#SBATCH -J xbarncu
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH -o /fact_home/shuoaojia/nuca/xbarncu-%j.out
#SBATCH -e /fact_home/shuoaojia/nuca/xbarncu-%j.err
cd /fact_home/shuoaojia/nuca
echo "NODE=$(hostname)"
NCU=/usr/local/cuda-12.8/bin/ncu
M=lts__t_requests_srcunit_ltcfabric.sum,lts__ltcfabric2lts_cycles_active.avg.pct_of_peak_sustained_elapsed,lts__ltcfabric2ltc_cycles_active.avg.pct_of_peak_sustained_elapsed,lts__t_requests_srcunit_tex_op_read.sum
for m in 0 2; do
  echo "##### MODE $m"
  $NCU --csv --metrics "$M" ./xbar_bw $m 2>&1 | tail -8
done
echo DONE
