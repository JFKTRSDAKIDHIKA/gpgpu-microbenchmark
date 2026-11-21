#!/bin/bash
#SBATCH -J h20lat
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH -o /fact_home/shuoaojia/nuca/hopper-h20/test6/run-%j.out
#SBATCH -e /fact_home/shuoaojia/nuca/hopper-h20/test6/run-%j.err
cd /fact_home/shuoaojia/nuca/hopper-h20/test6
rm -f lat.csv
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES"
/usr/local/cuda-12.8/bin/nvcc -O3 -gencode arch=compute_90,code=sm_90 \
    -o load load.cu profiler.c -lcudart -lnvidia-ml 2>&1 | head -20 || { echo BUILD_FAIL; exit 1; }
echo BUILD_OK
CUDA_VISIBLE_DEVICES=0 ./load
echo DONE
