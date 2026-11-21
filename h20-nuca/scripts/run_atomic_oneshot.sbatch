#!/bin/bash
#SBATCH -J nuca_atomic
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -o nuca_atomic-%j.out
#SBATCH -e nuca_atomic-%j.err
# Build and run the atomic NUCA probe. Submit from the h20-nuca/ directory:
#   cd ~/gpgpu-microbenchmark/h20-nuca && sbatch scripts/build_and_run_atomic.sbatch
cd "${SLURM_SUBMIT_DIR:-$PWD}"
NVCC=/usr/local/cuda-12.8/bin/nvcc
echo "NODE=$(hostname) GPU=$CUDA_VISIBLE_DEVICES"
$NVCC -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a src/nuca_atomic.cu -o nuca_atomic -lcudart || { echo BUILD_FAIL; exit 1; }
./nuca_atomic
echo DONE
