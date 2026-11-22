#!/bin/bash
#SBATCH -J dsmem
#SBATCH -N 1
#SBATCH -p ai
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -o dsmem-%j.out
#SBATCH -e dsmem-%j.err
# Build + run the DSMEM 8x8 latency matrix.
# Works from either the repo root or the h20-nuca directory:
#   sbatch h20-nuca/scripts/run_dsmem_oneshot.sbatch      (from repo root)
#   sbatch scripts/run_dsmem_oneshot.sbatch               (from h20-nuca)
# NOTE: the login node has no CUDA, so the build happens here, inside the job.

D="${SLURM_SUBMIT_DIR:-$PWD}"
if   [ -d "$D/h20-nuca/src" ]; then cd "$D/h20-nuca"
elif [ -d "$D/src" ];       then cd "$D"
else echo "cannot locate h20-nuca/ (submit dir: $D)"; exit 1; fi

echo "pwd=$(pwd)  NODE=$(hostname)  GPU=$CUDA_VISIBLE_DEVICES"
NVCC=/usr/local/cuda-12.8/bin/nvcc
$NVCC -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a src/topo_dsmem.cu -o topo_dsmem -lcudart || { echo BUILD_FAIL; exit 1; }
./topo_dsmem
echo DONE
