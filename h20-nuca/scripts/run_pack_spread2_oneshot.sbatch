#!/bin/bash
#SBATCH -J pspread2
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -o pspread2-%j.out
#SBATCH -e pspread2-%j.err
# Compile + run pack_spread2.cu with arguments from the environment:
#   cd ~/gpgpu-microbenchmark/h20-nuca
#   K=8 PACKED=1 MODE=0 sbatch scripts/run_pack_spread2_oneshot.sbatch
#   K  = number of active SMs (1..63)
#   PACKED = 1 (fill GPC 0 first) | 0 (round-robin over the 7 GPCs)
#   MODE = 0 read | 1 write | 2 mixed 1:1
cd "${SLURM_SUBMIT_DIR:-$PWD}"
NVCC=/usr/local/cuda-12.8/bin/nvcc
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES  K=${K:-1} PACKED=${PACKED:-0} MODE=${MODE:-0}"
$NVCC -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a src/pack_spread2.cu -o pack_spread2 -lcudart || { echo BUILD_FAIL; exit 1; }
./pack_spread2 "${K:-1}" "${PACKED:-0}" "${MODE:-0}"
echo DONE
